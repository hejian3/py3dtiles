import argparse
from multiprocessing import cpu_count, Process
import os
from pathlib import Path, PurePath
import pickle
import shutil
import struct
import sys
import time
import traceback
from typing import List, Optional, Tuple, Union

import psutil
from pyproj import CRS
import zmq

from py3dtiles.exceptions import SrsInMissingException, WorkerException
from py3dtiles.merger_v2 import merger_v2
from py3dtiles.reader import las_reader, ply_reader, wkb_reader, xyz_reader
from py3dtiles.tilers.b3dm.b3dm_management import B3dmActions, B3dmMetadata, B3dmState
from py3dtiles.tilers.node import node_process
from py3dtiles.tilers.node.shared_node_store import SharedNodeStore
from py3dtiles.tilers.pnts.pnts_management import PntsActions, PntsMetadata, PntsState
from py3dtiles.utils import CommandType, ResponseType, str_to_CRS

TOTAL_MEMORY_MB = int(psutil.virtual_memory().total / (1024 * 1024))
DEFAULT_CACHE_SIZE = int(TOTAL_MEMORY_MB / 10)
CPU_COUNT = cpu_count()

# IPC protocol is not supported on Windows
if os.name == 'nt':
    URI = 'tcp://127.0.0.1:0'
else:
    URI = "ipc:///tmp/py3dtiles1"

READER_MAP = {
    '.xyz': xyz_reader,
    '.las': las_reader,
    '.laz': las_reader,
    '.ply': ply_reader,
    '.wkb': wkb_reader
}


class Worker(Process):
    """
    This class waits from jobs commands from the Zmq socket.
    """
    def __init__(self, activity_graph, pnts_metadata: 'PntsMetadata', b3dm_metadata: 'B3dmMetadata', folder: Path, verbosity, uri):
        super().__init__()
        self.activity_graph = activity_graph
        self.pnts_metadata = pnts_metadata
        self.b3dm_metadata = b3dm_metadata
        self.folder = folder
        self.verbosity = verbosity
        self.uri = uri

        # Socket to receive messages on
        self.context = zmq.Context()
        self.skt = None

    def run(self):
        self.skt = self.context.socket(zmq.DEALER)

        self.skt.connect(self.uri)

        startup_time = time.time()
        idle_time = 0

        if self.activity_graph:
            activity = open(f'activity.{os.getpid()}.csv', 'w')

        # notify we're ready
        self.skt.send_multipart([ResponseType.IDLE.value])

        while True:
            try:
                before = time.time() - startup_time
                self.skt.poll()
                after = time.time() - startup_time

                idle_time += after - before

                message = self.skt.recv_multipart()
                content = message[1:]
                command = content[0]

                delta = time.time() - pickle.loads(message[0])
                if delta > 0.01 and self.verbosity >= 1:
                    print(f'{os.getpid()} / {round(after, 2)} : Delta time: {round(delta, 3)}')

                if command == CommandType.READ_FILE.value:
                    self.execute_read_file(content)
                    command_type = 1
                elif command == CommandType.PROCESS_JOBS.value:
                    self.execute_process_jobs(content)
                    command_type = 2
                elif command == CommandType.WRITE_PNTS.value:
                    self.execute_write_pnts(content)
                    command_type = 3
                elif command == CommandType.SHUTDOWN.value:
                    break  # ack
                else:
                    raise NotImplementedError(f'Unknown command {command}')

                # notify we're idle
                self.skt.send_multipart([ResponseType.IDLE.value])

                if self.activity_graph:
                    print(f'{before}, {command_type}', file=activity)
                    print(f'{before}, 0', file=activity)
                    print(f'{after}, 0', file=activity)
                    print(f'{after}, {command_type}', file=activity)
            except Exception as e:
                traceback.print_exc()
                # usually first arg is the explaining string.
                # let's assume it is always in our context
                self.skt.send_multipart([ResponseType.ERROR.value, e.args[0].encode()])
                # we still print it for stacktraces

        if self.activity_graph:
            activity.close()

        if self.verbosity >= 1:
            print('total: {} sec, idle: {}'.format(
                round(time.time() - startup_time, 1),
                round(idle_time, 1))
            )

        self.skt.send_multipart([ResponseType.HALTED.value])

    def execute_read_file(self, content):
        parameters = pickle.loads(content[1])

        extension = PurePath(parameters['filename']).suffix
        if extension in READER_MAP:
            reader = READER_MAP[extension]
        else:
            raise ValueError(f"The file with {extension} extension can't be read, "
                             f"the available extensions are: {READER_MAP.keys()}")

        reader.run(
            parameters['filename'],
            parameters['offset_scale'],
            parameters['portion'],
            self.skt,
            self.pnts_metadata.transformer
        )

    def execute_write_pnts(self, content):
        PntsActions.write_pnts(self.skt, content[2], content[1], self.folder, self.pnts_metadata.write_rgb)

    def execute_process_jobs(self, content):
        node_process.run(
            content[1:],
            self.pnts_metadata.root_aabb,
            self.pnts_metadata.root_spacing,
            self.pnts_metadata.root_scale[0],
            self.skt,
            self.verbosity
        )


# Manager
class ZmqManager:
    """
    This class sends messages to the workers.
    We can also request general status.
    """
    def __init__(self, number_of_jobs: int, process_args: tuple):
        """
        For the process_args argument, see the init method of Worker
        to get the list of needed parameters.
        """
        self.context = zmq.Context()

        self.number_of_jobs = number_of_jobs

        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(URI)
        # Useful only when TCP is used to get the URI with the opened port
        self.uri = self.socket.getsockopt(zmq.LAST_ENDPOINT)

        self.processes = [
            Worker(*process_args, self.uri)
            for _ in range(number_of_jobs)
        ]
        [p.start() for p in self.processes]

        self.activities = [p.pid for p in self.processes]
        self.idle_clients = []

        self.killing_processes = False
        self.number_processes_killed = 0
        self.time_waiting_an_idle_process = 0

    def send_to_process(self, message):
        if not self.idle_clients:
            raise ValueError("idle_clients is empty")
        self.socket.send_multipart([self.idle_clients.pop(), pickle.dumps(time.time())] + message)

    def send_to_all_idle_processes(self, message):
        if not self.idle_clients:
            raise ValueError("idle_clients is empty")
        for client in self.idle_clients:
            self.socket.send_multipart([client, pickle.dumps(time.time())] + message)
        self.idle_clients.clear()

    def can_queue_more_jobs(self):
        return len(self.idle_clients) != 0

    def add_idle_client(self, client_id):
        if client_id in self.idle_clients:
            raise ValueError(f"The client id {client_id} is already in idle_clients")
        self.idle_clients.append(client_id)

    def are_all_processes_idle(self):
        return len(self.idle_clients) == self.number_of_jobs

    def are_all_processes_killed(self):
        return self.number_processes_killed == self.number_of_jobs

    def kill_all_processes(self):
        self.send_to_all_idle_processes([CommandType.SHUTDOWN.value])
        self.killing_processes = True

    def join_all_processes(self):
        for p in self.processes:
            p.join()

    def terminate_all_processes(self):
        for p in self.processes:
            p.terminate()


def convert(*args, **kwargs):
    converter = _Convert(*args, **kwargs)
    return converter.convert()


class _Convert:
    def __init__(self,
                 files: List[Union[str, Path]],
                 outfolder: Union[str, Path] = './3dtiles',
                 overwrite: bool = False,
                 jobs: int = CPU_COUNT,
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 crs_out: Optional[CRS] = None,
                 crs_in: Optional[CRS] = None,
                 fraction: int = 100,
                 benchmark: Optional[str] = None,
                 rgb: bool = True,
                 graph: bool = False,
                 color_scale: Optional[float] = None,
                 verbose: bool = False):
        """
        :param files: Filenames to process. The file must use the .las, .laz, .xyz or .ply format.
        :param outfolder: The folder where the resulting tileset will be written.
        :param overwrite: Overwrite the ouput folder if it already exists.
        :param jobs: The number of parallel jobs to start. Default to the number of cpu.
        :param cache_size: Cache size in MB. Default to available memory / 10.
        :param crs_out: CRS to convert the output with
        :param crs_in: Set a default input CRS
        :param fraction: Percentage of the pointcloud to process, between 0 and 100.
        :param benchmark: Print summary at the end of the process
        :param rgb: Export rgb attributes.
        :param graph: Produce debug graphs (requires pygal).
        :param color_scale: Force color scale

        :raises SrsInMissingException: if py3dtiles couldn't find srs informations in input files and srs_in is not specified
        :raises SrsInMixinException: if the input files have different CRS

        """
        self.jobs = jobs
        self.cache_size = cache_size

        # allow str directly if only one input
        self.files = [files] if isinstance(files, str) or isinstance(files, Path) else files
        self.files = [Path(file) for file in self.files]

        self.verbose = verbose
        self.graph = graph
        self.benchmark = benchmark
        self.startup = None

        self.pnts_metadata, pnts_portions, self.b3dm_metadata, b3dm_portions = self.get_file_info(color_scale, crs_in)

        if self.pnts_metadata:
            self.pnts_metadata.write_rgb = rgb
            self.pnts_metadata.get_transformer(crs_out)
            self.pnts_metadata.get_rotation_matrix(crs_out)
            self.pnts_metadata.get_root_aabb()

        if self.verbose >= 1:
            self.print_summary()
        if self.graph:
            self.progression_log = open('progression.csv', 'w')

        # create folder
        self.out_folder = Path(outfolder)
        if self.out_folder.is_dir():
            if overwrite:
                shutil.rmtree(self.out_folder, ignore_errors=True)
            else:
                raise FileExistsError(f"Folder '{self.out_folder}' already exists")

        self.out_folder.mkdir()
        self.working_dir = self.out_folder / "tmp"
        self.working_dir.mkdir(parents=True)

        self.zmq_manager = ZmqManager(self.jobs, (self.graph, self.pnts_metadata, self.b3dm_metadata, self.out_folder, self.verbose))

        if self.pnts_metadata:
            node_store = SharedNodeStore(self.working_dir)
            self.pnts_state = PntsState(pnts_portions, node_store, max(1, self.jobs // 2))
            (self.out_folder / "pnts_tiles").mkdir()
        else:
            self.pnts_state = None

        if self.b3dm_metadata:
            self.b3dm_state = B3dmState(b3dm_portions)
        else:
            self.b3dm_state = None

    def get_file_info(self, color_scale, crs_in: CRS) -> Tuple[PntsMetadata, List, B3dmMetadata, List]:
        pnts_metadata = PntsMetadata()
        pnts_metadata.crs_in = crs_in
        pnts_portions = []
        b3dm_metadata = B3dmMetadata()
        b3dm_portions = []

        # read all input files headers and determine the aabb/spacing
        for input_file in self.files:
            extension = input_file.suffix
            if extension in READER_MAP:
                reader = READER_MAP[extension]
            else:
                raise ValueError(f"The file with {extension} extension can't be read, "
                                 f"the available extensions are: {READER_MAP.keys()}")

            file_info = reader.get_metadata(input_file, color_scale=color_scale)

            if file_info['type'] == 'pnts':
                pnts_portions += pnts_metadata.update(input_file, file_info)

            elif file_info['type'] == 'b3dm':
                b3dm_portions += b3dm_metadata.update(input_file, file_info)

        return pnts_metadata, pnts_portions, b3dm_metadata, b3dm_portions

    def convert(self):
        """convert

        Convert pointclouds (xyz, las or laz) to 3dtiles tileset containing pnts node
        """
        self.startup = time.time()

        try:
            while not self.zmq_manager.killing_processes:
                now = time.time() - self.startup

                at_least_one_job_ended = False
                if not self.zmq_manager.can_queue_more_jobs() or self.zmq_manager.socket.poll(timeout=0, flags=zmq.POLLIN):
                    at_least_one_job_ended = self.process_message()

                if self.pnts_state is not None:
                    while self.pnts_state.pnts_to_writing and self.zmq_manager.can_queue_more_jobs():
                        self.zmq_manager.send_to_process(
                            PntsActions.create_command_pnts_to_write(self.pnts_state)
                        )

                    if self.zmq_manager.can_queue_more_jobs():
                        PntsActions.send_points_to_process(self.pnts_state, self.zmq_manager, now)

                    while self.pnts_state.can_add_reading_jobs() and self.zmq_manager.can_queue_more_jobs():
                        self.zmq_manager.send_to_process(
                            PntsActions.create_command_send_file_to_read(self.pnts_state, self.pnts_metadata)
                        )

                if self.b3dm_state is not None and self.b3dm_state.tree is None:
                    if self.zmq_manager.can_queue_more_jobs():
                        B3dmActions.send_build_tree(self.b3dm_state, self.out_folder, now)

                # if at this point we have no work in progress => we're done
                if self.zmq_manager.are_all_processes_idle():
                    self.zmq_manager.kill_all_processes()

                if at_least_one_job_ended:
                    pass
                    # self.print_debug(now)
                    # if self.graph:
                    #     percent = round(100 * self.state.processed_points / self.file_info['point_count'], 3)
                    #     print(f'{time.time() - self.startup}, {percent}', file=self.progression_log)

                if self.pnts_state is not None:
                    self.pnts_state.node_store.control_memory_usage(self.cache_size, self.verbose) # TODO should not only pnts node

            self.zmq_manager.join_all_processes()

            # if self.state.points_in_pnts != self.file_info['pnts']['point_count']:
            #     raise ValueError("!!! Invalid point count in the written .pnts"
            #                      + f"(expected: {self.file_info['pnts']['point_count']}, was: {self.state.points_in_pnts})")

            # if self.verbose >= 1:
            #     print('Writing 3dtiles {}'.format(self.file_info['pnts']['avg_min']))

            final_tilesets = []
            if self.pnts_metadata:
                final_tilesets.append(
                    PntsActions.write_tileset(self.pnts_metadata, self.out_folder)
                )
            if self.b3dm_metadata:
                final_tilesets.append(
                    B3dmActions.write_tileset(self.b3dm_state, self.b3dm_metadata, self.out_folder)
                )

            self.write_tileset(self.out_folder, final_tilesets)
            shutil.rmtree(self.working_dir)

            if self.verbose >= 1:
                print('Done')

            # if self.benchmark:
            #     print('{},{},{},{}'.format(
            #         self.benchmark,
            #         ','.join([f.name for f in self.files]),
            #         self.state.points_in_pnts,
            #         round(time.time() - self.startup, 1)))
        finally:
            self.zmq_manager.terminate_all_processes()

            if self.verbose >= 1:
                print('destroy', round(self.zmq_manager.time_waiting_an_idle_process, 2))

            # pygal chart
            if self.graph:
                self.progression_log.close()
                self.draw_graph()

            self.zmq_manager.context.destroy()

    def process_message(self):
        one_job_ended = False

        # Blocking read but it's fine because either all our child processes are busy
        # or we know that there's something to read (zmq.POLLIN)
        start = time.time()
        message = self.zmq_manager.socket.recv_multipart()

        client_id = message[0]
        result = message[1:]
        return_type = result[0]

        if return_type == ResponseType.IDLE.value:
            self.zmq_manager.add_idle_client(client_id)

            if not self.zmq_manager.can_queue_more_jobs():
                self.zmq_manager.time_waiting_an_idle_process += time.time() - start

        elif return_type == ResponseType.HALTED.value:
            self.zmq_manager.number_processes_killed += 1

        elif return_type == ResponseType.READ.value:
            self.pnts_state.number_of_reading_jobs -= 1
            one_job_ended = True

        elif return_type == ResponseType.PROCESSED.value:
            content = pickle.loads(result[-1])
            self.pnts_state.processed_points += content['total']
            self.pnts_state.points_in_progress -= content['total']

            del self.pnts_state.processing_nodes[content['name']]

            PntsActions.dispatch_processed_nodes(content, self.pnts_state)

            one_job_ended = True

        elif return_type == ResponseType.PNTS_WRITTEN.value:
            self.pnts_state.points_in_pnts += struct.unpack('>I', result[1])[0]
            self.pnts_state.number_of_writing_jobs -= 1

        elif return_type == ResponseType.NEW_TASK.value:
            self.pnts_state.add_tasks_to_process(
                node_name=result[1], data=result[2], point_count=struct.unpack('>I', result[3])[0]
            )

        elif return_type == ResponseType.ERROR.value:
            raise WorkerException(f'An exception occurred in a worker: {result[1].decode()}')

        else:
            raise NotImplementedError(f"The command {return_type} is not implemented")

        return one_job_ended

    def send_file_to_read(self):
        if self.verbose >= 1:
            print(f'Submit next portion {self.pnts_state.point_cloud_file_parts[-1]}')
        file, portion = self.pnts_state.point_cloud_file_parts.pop()
        self.pnts_state.points_in_progress += portion[1] - portion[0]

        self.zmq_manager.send_to_process([CommandType.READ_FILE.value, pickle.dumps({
            'filename': file,
            'offset_scale': (
                -self.pnts_metadata.avg_min,
                self.pnts_metadata.root_scale,
                self.pnts_metadata.rotation_matrix[:3, :3].T if self.pnts_metadata.rotation_matrix is not None else None,
                self.pnts_metadata.color_scale.get(file) if self.pnts_metadata.color_scale is not None else None,
            ),
            'portion': portion,
        })])

        self.pnts_state.number_of_reading_jobs += 1

    def write_tileset(self, out_folder: Path, children_tilesets: List[Path]):
        if False and len(children_tilesets) == 1:
            children_tilesets[0].rename(children_tilesets[0].parent / "tileset.json")
        else:
            merger_v2(children_tilesets, out_folder)


    def print_summary(self):
        print('Summary:')
        # print('  - points to process: {}'.format(self.file_info['point_count']))
        # print(f'  - offset to use: {self.avg_min}')
        # print(f'  - root spacing: {self.root_spacing / self.root_scale[0]}')
        # print(f'  - root aabb: {self.root_aabb}')
        # print(f'  - original aabb: {self.original_aabb}')
        # print(f'  - scale: {self.root_scale}')

    def draw_graph(self):
        import pygal  # type: ignore

        dateline = pygal.XY(x_label_rotation=25, secondary_range=(0, 100))
        for pid in self.zmq_manager.activities:
            activity = []
            activity_csv_path = Path(f'activity.{pid}.csv')
            i = len(self.zmq_manager.activities) - self.zmq_manager.activities.index(pid) - 1
            # activities.index(pid) =
            with activity_csv_path.open() as f:
                content = f.read().split('\n')
                for line in content[1:]:
                    line = line.split(',')
                    if line[0]:
                        ts = float(line[0])
                        value = int(line[1]) / 3.0
                        activity.append((ts, i + value * 0.9))

            activity_csv_path.unlink()
            if activity:
                activity.append((activity[-1][0], activity[0][1]))
                activity.append(activity[0])
                dateline.add(str(pid), activity, show_dots=False, fill=True)

        progression_csv_path = Path('progression.csv')
        with progression_csv_path.open() as f:
            values = []
            for line in f.read().split('\n'):
                if line:
                    line = line.split(',')
                    values += [(float(line[0]), float(line[1]))]
        progression_csv_path.unlink()
        dateline.add('progression', values, show_dots=False, secondary=True,
                     stroke_style={'width': 2, 'color': 'black'})

        dateline.render_to_file('activity.svg')

    def print_debug(self, now):
        pass
        # if self.verbose >= 3:
        #     print('{:^16}|{:^8}|{:^8}'.format('Name', 'Points', 'Seconds'))
        #     for name, v in self.state.processing_nodes.items():
        #         print('{:^16}|{:^8}|{:^8}'.format(
        #             '{} ({})'.format(name.decode('ascii'), v[0]),
        #             v[1],
        #             round(now - v[2], 1)))
        #     print('')
        #     print('Pending:')
        #     print('  - root: {} / {}'.format(
        #         len(self.state.point_cloud_file_parts),
        #         self.state.initial_portion_count))
        #     print('  - other: {} files for {} nodes'.format(
        #         sum([len(f[0]) for f in self.state.node_to_process.values()]),
        #         len(self.state.node_to_process)))
        #     print('')
        #
        # elif self.verbose >= 2:
        #     self.state.print_debug()
        #
        # if self.verbose >= 1:
        #     print('{} % points in {} sec [{} tasks, {} nodes, {} wip]'.format(
        #         round(100 * self.state.processed_points / self.file_info['point_count'], 2),
        #         round(now, 1),
        #         self.jobs - len(self.zmq_manager.idle_clients),
        #         len(self.state.processing_nodes),
        #         self.state.points_in_progress))
        #
        # elif self.verbose >= 0:
        #     percent = round(100 * self.state.processed_points / self.file_info['point_count'], 2)
        #     time_left = (100 - percent) * now / (percent + 0.001)
        #     print(f'\r{percent:>6} % in {round(now)} sec [est. time left: {round(time_left)} sec]', end='',
        #           flush=True)


def init_parser(subparser):
    parser = subparser.add_parser(
        'convert',
        help='Convert input 3D data to a 3dtiles tileset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'files',
        nargs='+',
        help='Filenames to process. The file must use the .las, .laz (lastools must be installed), .xyz or .ply format.')
    parser.add_argument(
        '--out',
        type=str,
        help='The folder where the resulting tileset will be written.',
        default='./3dtiles')
    parser.add_argument(
        '--overwrite',
        help='Delete and recreate the ouput folder if it already exists. WARNING: be careful, there will be no confirmation!',
        action='store_true')
    parser.add_argument(
        '--jobs',
        help='The number of parallel jobs to start. Default to the number of cpu.',
        default=cpu_count(),
        type=int)
    parser.add_argument(
        '--cache_size',
        help='Cache size in MB. Default to available memory / 10.',
        default=int(TOTAL_MEMORY_MB / 10),
        type=int)
    parser.add_argument(
        '--srs_out', help='SRS to convert the output with (numeric part of the EPSG code)', type=str)
    parser.add_argument(
        '--srs_in', help='Override input SRS (numeric part of the EPSG code)', type=str)
    parser.add_argument(
        '--fraction',
        help='Percentage of the pointcloud to process.',
        default=100, type=int)
    parser.add_argument(
        '--benchmark',
        help='Print summary at the end of the process', type=str)
    parser.add_argument(
        '--no-rgb',
        help="Don't export rgb attributes", action='store_true')
    parser.add_argument(
        '--graph',
        help='Produce debug graphes (requires pygal)', action='store_true')
    parser.add_argument(
        '--color_scale',
        help='Force color scale', type=float)

    return parser


def main(args):
    try:
        return convert(args.files,
                       outfolder=args.out,
                       overwrite=args.overwrite,
                       jobs=args.jobs,
                       cache_size=args.cache_size,
                       crs_out=str_to_CRS(args.srs_out),
                       crs_in=str_to_CRS(args.srs_in),
                       fraction=args.fraction,
                       benchmark=args.benchmark,
                       rgb=not args.no_rgb,
                       graph=args.graph,
                       color_scale=args.color_scale,
                       verbose=args.verbose)
    except SrsInMissingException:
        print('No SRS information in input files, you should specify it with --srs_in')
        sys.exit(1)
