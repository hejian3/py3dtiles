import argparse
from collections import namedtuple
import concurrent.futures
import json
from multiprocessing import cpu_count, Process
import os
from pathlib import Path, PurePath
import pickle
import shutil
import struct
import sys
import time
import traceback

import numpy as np
import psutil
from pyproj import CRS, Transformer
import zmq

from py3dtiles import TileContentReader
from py3dtiles.constants import MIN_POINT_SIZE
from py3dtiles.exceptions import WorkerException
from py3dtiles.points.node import Node
from py3dtiles.points.shared_node_store import SharedNodeStore
from py3dtiles.points.task import las_reader, node_process, pnts_writer, xyz_reader
from py3dtiles.points.transformations import (
    angle_between_vectors, inverse_matrix, rotation_matrix, scale_matrix, translation_matrix, vector_product
)
from py3dtiles.points.utils import CommandType, compute_spacing, node_name_to_path, ResponseType
from py3dtiles.utils import SrsInMissingException

TOTAL_MEMORY_MB = int(psutil.virtual_memory().total / (1024 * 1024))
DEFAULT_CACHE_SIZE = int(TOTAL_MEMORY_MB / 10)
CPU_COUNT = cpu_count()

# IPC protocol is not supported on Windows
if os.name == 'nt':
    URI = 'tcp://127.0.0.1:0'
else:
    URI = "ipc:///tmp/py3dtiles1"

OctreeMetadata = namedtuple('OctreeMetadata', ['aabb', 'spacing', 'scale'])


def make_rotation_matrix(z1, z2):
    v0 = z1 / np.linalg.norm(z1)
    v1 = z2 / np.linalg.norm(z2)

    return rotation_matrix(
        angle_between_vectors(v0, v1),
        vector_product(v0, v1))


class Worker(Process):
    """
    This class waits from jobs commands from the Zmq socket.
    """
    def __init__(self, activity_graph, transformer, octree_metadata, folder: Path, write_rgb, verbosity, uri):
        super().__init__()
        self.activity_graph = activity_graph
        self.transformer = transformer
        self.octree_metadata = octree_metadata
        self.folder = folder
        self.write_rgb = write_rgb
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
            activity = open('activity.{}.csv'.format(os.getpid()), 'w')

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
                    print('{} / {} : Delta time: {}'.format(os.getpid(), round(after, 2), round(delta, 3)))

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

        ext = PurePath(parameters['filename']).suffix
        init_reader_fn = las_reader.run if ext in ('.las', '.laz') else xyz_reader.run
        init_reader_fn(
            parameters['filename'],
            parameters['offset_scale'],
            parameters['portion'],
            self.skt,
            self.transformer
        )

    def execute_write_pnts(self, content):
        pnts_writer.run(self.skt, content[2], content[1], self.folder, self.write_rgb)

    def execute_process_jobs(self, content):
        node_process.run(
            content[1:],
            self.octree_metadata,
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


def is_ancestor(node_name, ancestor):
    """
    Example, the tile 22 is ancestor of 22458
    Particular case, the tile 22 is ancestor of 22
    """
    return len(ancestor) <= len(node_name) and node_name[0:len(ancestor)] == ancestor


def is_ancestor_in_list(node_name, ancestors):
    for ancestor in ancestors:
        if not ancestor or is_ancestor(node_name, ancestor):
            return True
    return False


def can_pnts_be_written(node_name, finished_node, input_nodes, active_nodes):
    return (
        is_ancestor(node_name, finished_node)
        and not is_ancestor_in_list(node_name, active_nodes)
        and not is_ancestor_in_list(node_name, input_nodes))


class State:
    def __init__(self, pointcloud_file_portions, max_reading_jobs: int):
        self.processed_points = 0
        self.max_point_in_progress = 60_000_000
        self.points_in_progress = 0
        self.points_in_pnts = 0

        # pointcloud_file_portions is a list of tuple (filename, (start offset, end offset))
        self.point_cloud_file_parts = pointcloud_file_portions
        self.initial_portion_count = len(pointcloud_file_portions)
        self.max_reading_jobs = max_reading_jobs
        self.number_of_reading_jobs = 0
        self.number_of_writing_jobs = 0

        # node_to_process is a dictionary of tasks,
        # each entry is a tile identified by its name (a string of numbers)
        # so for each entry, it is a list of tasks
        # a task is a tuple (list of points, point_count)
        # points is a dictionary {xyz: list of coordinates, color: the associated color}
        self.node_to_process = {}
        # when a node is sent to a process, the item moves to processing_nodes
        # the structure is different. The key remains the node name. But the value is : (len(tasks), point_count, now)
        # these values is for loging
        self.processing_nodes = {}
        # when processing is finished, move the tile name in processed_nodes
        # since the content is at this stage, stored in the node_store,
        # just keep the name of the node.
        # This list will be filled until the writing could be started.
        self.waiting_writing_nodes = []
        # when the node is writing, its name is moved from waiting_writing_nodes to pnts_to_writing
        # the data to write are stored in a node object.
        self.pnts_to_writing = []

    def is_reading_finish(self):
        return not self.point_cloud_file_parts and self.number_of_reading_jobs == 0

    def add_tasks_to_process(self, node_name, data, point_count):
        if point_count <= 0:
            raise ValueError("point_count should be strictly positive, currently", point_count)

        if node_name not in self.node_to_process:
            self.node_to_process[node_name] = ([data], point_count)
        else:
            tasks, count = self.node_to_process[node_name]
            tasks.append(data)
            self.node_to_process[node_name] = (tasks, count + point_count)

    def can_add_reading_jobs(self):
        return (
            self.point_cloud_file_parts
            and self.points_in_progress < self.max_point_in_progress
            and self.number_of_reading_jobs < self.max_reading_jobs
        )

    def print_debug(self):
        print('{:^16}|{:^8}|{:^8}|{:^8}'.format('Step', 'Input', 'Active', 'Inactive'))
        print('{:^16}|{:^8}|{:^8}|{:^8}'.format(
            'Reader',
            len(self.point_cloud_file_parts),
            self.number_of_reading_jobs,
            ''))
        print('{:^16}|{:^8}|{:^8}|{:^8}'.format(
            'Node process',
            len(self.node_to_process),
            len(self.processing_nodes),
            len(self.waiting_writing_nodes)))
        print('{:^16}|{:^8}|{:^8}|{:^8}'.format(
            'Pnts writer',
            len(self.pnts_to_writing),
            self.number_of_writing_jobs,
            ''))


def convert(*args, **kwargs):
    converter = _Convert(*args, **kwargs)
    return converter.convert()


class _Convert:
    def __init__(self,
                 files,
                 outfolder='./3dtiles',
                 overwrite=False,
                 jobs=CPU_COUNT,
                 cache_size=DEFAULT_CACHE_SIZE,
                 srs_out=None,
                 srs_in=None,
                 fraction=100,
                 benchmark=None,
                 rgb=True,
                 graph=False,
                 color_scale=None,
                 verbose=False):
        """
        :param files: Filenames to process. The file must use the .las, .laz or .xyz format.
        :type files: list of str, or str
        :param outfolder: The folder where the resulting tileset will be written.
        :type outfolder: path-like object
        :param overwrite: Overwrite the ouput folder if it already exists.
        :type overwrite: bool
        :param jobs: The number of parallel jobs to start. Default to the number of cpu.
        :type jobs: int
        :param cache_size: Cache size in MB. Default to available memory / 10.
        :type cache_size: int
        :param srs_out: SRS to convert the output with (numeric part of the EPSG code)
        :type srs_out: int or str
        :param srs_in: Override input SRS (numeric part of the EPSG code)
        :type srs_in: int or str
        :param fraction: Percentage of the pointcloud to process, between 0 and 100.
        :type fraction: int
        :param benchmark: Print summary at the end of the process
        :type benchmark: str
        :param rgb: Export rgb attributes.
        :type rgb: bool
        :param graph: Produce debug graphes (requires pygal).
        :type graph: bool
        :param color_scale: Force color scale
        :type color_scale: float

        :raises SrsInMissingException: if py3dtiles couldn't find srs informations in input files and srs_in is not specified

        """
        self.jobs = jobs
        self.cache_size = cache_size
        self.rgb = rgb

        # allow str directly if only one input
        self.files = [files] if isinstance(files, str) or isinstance(files, Path) else files
        self.files = [Path(file) for file in self.files]

        self.verbose = verbose
        self.graph = graph
        self.benchmark = benchmark
        self.startup = None

        self.infos = self.get_infos(color_scale, srs_in, srs_out)

        crs_in, crs_out, transformer = self.get_crs(srs_in, srs_out)
        self.rotation_matrix, self.original_aabb, self.avg_min = self.get_rotation_matrix(srs_in, transformer)
        self.root_aabb, self.root_scale, self.root_spacing = self.get_root_aabb(self.original_aabb)
        octree_metadata = OctreeMetadata(aabb=self.root_aabb, spacing=self.root_spacing, scale=self.root_scale[0])

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

        self.zmq_manager = ZmqManager(self.jobs, (self.graph, transformer, octree_metadata, self.out_folder, self.rgb, self.verbose))
        self.node_store = SharedNodeStore(self.working_dir)
        self.state = State(self.infos['portions'], max(1, self.jobs // 2))

    def get_infos(self, color_scale, srs_in, srs_out):
        # read all input files headers and determine the aabb/spacing
        extensions = set()
        for file in self.files:
            extensions.add(file.suffix)
        if len(extensions) != 1:
            raise ValueError("All files should have the same extension, currently there are", extensions)
        extension = extensions.pop()

        init_reader_fn = las_reader.init if extension in ('.las', '.laz') else xyz_reader.init
        return init_reader_fn(self.files, color_scale=color_scale, srs_in=srs_in, srs_out=srs_out)

    def get_crs(self, srs_in, srs_out):
        if srs_out:
            crs_out = CRS('epsg:{}'.format(srs_out))
            if srs_in:
                crs_in = CRS('epsg:{}'.format(srs_in))
            elif not self.infos['srs_in']:
                raise SrsInMissingException('No SRS information in the provided files')
            else:
                crs_in = CRS(self.infos['srs_in'])

            transformer = Transformer.from_crs(crs_in, crs_out)
        else:
            crs_in = None
            crs_out = None
            transformer = None

        return crs_in, crs_out, transformer

    def get_rotation_matrix(self, srs_out, transformer):
        avg_min = self.infos['avg_min']
        aabb = self.infos['aabb']

        rotation_matrix = None
        if srs_out:

            bl = np.array(list(transformer.transform(
                aabb[0][0], aabb[0][1], aabb[0][2])))
            tr = np.array(list(transformer.transform(
                aabb[1][0], aabb[1][1], aabb[1][2])))
            br = np.array(list(transformer.transform(
                aabb[1][0], aabb[0][1], aabb[0][2])))

            avg_min = np.array(list(transformer.transform(
                avg_min[0], avg_min[1], avg_min[2])))

            x_axis = br - bl

            bl = bl - avg_min
            tr = tr - avg_min

            if srs_out == '4978':
                # Transform geocentric normal => (0, 0, 1)
                # and 4978-bbox x axis => (1, 0, 0),
                # to have a bbox in local coordinates that's nicely aligned with the data
                rotation_matrix = make_rotation_matrix(avg_min, np.array([0, 0, 1]))
                rotation_matrix = np.dot(
                    make_rotation_matrix(x_axis, np.array([1, 0, 0])),
                    rotation_matrix)

                bl = np.dot(bl, rotation_matrix[:3, :3].T)
                tr = np.dot(tr, rotation_matrix[:3, :3].T)

            root_aabb = np.array([
                np.minimum(bl, tr),
                np.maximum(bl, tr)
            ])
        else:
            # offset
            root_aabb = aabb - avg_min

        return rotation_matrix, root_aabb, avg_min

    def get_root_aabb(self, original_aabb):
        base_spacing = compute_spacing(original_aabb)
        if base_spacing > 10:
            root_scale = np.array([0.01, 0.01, 0.01])
        elif base_spacing > 1:
            root_scale = np.array([0.1, 0.1, 0.1])
        else:
            root_scale = np.array([1, 1, 1])

        root_aabb = original_aabb * root_scale
        root_spacing = compute_spacing(root_aabb)
        return root_aabb, root_scale, root_spacing

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

                while self.state.pnts_to_writing and self.zmq_manager.can_queue_more_jobs():
                    self.send_pnts_to_write()

                if self.zmq_manager.can_queue_more_jobs():
                    self.send_points_to_process(now)

                while self.state.can_add_reading_jobs() and self.zmq_manager.can_queue_more_jobs():
                    self.send_file_to_read()

                # if at this point we have no work in progress => we're done
                if self.zmq_manager.are_all_processes_idle():
                    self.zmq_manager.kill_all_processes()

                if at_least_one_job_ended:
                    self.print_debug(now)
                    if self.graph:
                        percent = round(100 * self.state.processed_points / self.infos['point_count'], 3)
                        print('{}, {}'.format(time.time() - self.startup, percent), file=self.progression_log)

                self.node_store.control_memory_usage(self.cache_size, self.verbose)

            self.zmq_manager.join_all_processes()

            if self.state.points_in_pnts != self.infos['point_count']:
                raise ValueError("!!! Invalid point count in the written .pnts"
                                 + f"(expected: {self.infos['point_count']}, was: {self.state.points_in_pnts})")

            if self.verbose >= 1:
                print('Writing 3dtiles {}'.format(self.infos['avg_min']))

            self.write_tileset()
            shutil.rmtree(self.working_dir)

            if self.verbose >= 1:
                print('Done')

            if self.benchmark:
                print('{},{},{},{}'.format(
                    self.benchmark,
                    ','.join([f.name for f in self.files]),
                    self.state.points_in_pnts,
                    round(time.time() - self.startup, 1)))
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
            self.state.number_of_reading_jobs -= 1
            one_job_ended = True

        elif return_type == ResponseType.PROCESSED.value:
            content = pickle.loads(result[-1])
            self.state.processed_points += content['total']
            self.state.points_in_progress -= content['total']

            del self.state.processing_nodes[content['name']]

            self.dispatch_processed_nodes(content)

            one_job_ended = True

        elif return_type == ResponseType.PNTS_WRITTEN.value:
            self.state.points_in_pnts += struct.unpack('>I', result[1])[0]
            self.state.number_of_writing_jobs -= 1

        elif return_type == ResponseType.NEW_TASK.value:
            self.state.add_tasks_to_process(
                node_name=result[1], data=result[2], point_count=struct.unpack('>I', result[3])[0]
            )

        elif return_type == ResponseType.ERROR.value:
            raise WorkerException(f'An exception occurred in a worker: {result[1].decode()}')

        else:
            raise NotImplementedError(f"The command {return_type} is not implemented")

        return one_job_ended

    def dispatch_processed_nodes(self, content):
        if not content['name']:
            return

        self.node_store.put(content['name'], content['save'])
        self.state.waiting_writing_nodes.append(content['name'])

        if not self.state.is_reading_finish():
            return

        # if all nodes aren't processed yet,
        # we should check if linked ancestors are processed
        if self.state.processing_nodes or self.state.node_to_process:
            finished_node = content['name']
            if can_pnts_be_written(
                finished_node, finished_node,
                self.state.node_to_process, self.state.processing_nodes
            ):
                self.state.waiting_writing_nodes.pop(-1)
                self.state.pnts_to_writing.append(finished_node)

                for i in range(len(self.state.waiting_writing_nodes) - 1, -1, -1):
                    candidate = self.state.waiting_writing_nodes[i]

                    if can_pnts_be_written(
                        candidate, finished_node,
                        self.state.node_to_process, self.state.processing_nodes
                    ):
                        self.state.waiting_writing_nodes.pop(i)
                        self.state.pnts_to_writing.append(candidate)

        else:
            for c in self.state.waiting_writing_nodes:
                self.state.pnts_to_writing.append(c)
            self.state.waiting_writing_nodes.clear()

    def send_pnts_to_write(self):
        node_name = self.state.pnts_to_writing.pop()
        data = self.node_store.get(node_name)
        if not data:
            raise ValueError(f'{node_name} has no data')

        self.zmq_manager.send_to_process([CommandType.WRITE_PNTS.value, node_name, data])
        self.node_store.remove(node_name)
        self.state.number_of_writing_jobs += 1

    def send_points_to_process(self, now):
        potentials = sorted(
            # a key (=task) can be in node_to_process and processing_nodes if the node isn't completely processed
            [
                (node, task) for node, task in self.state.node_to_process.items()  # task: [data...], point_count
                if node not in self.state.processing_nodes
            ],
            key=lambda task: -len(task[0]))  # sort by node name size, the root nodes first

        while self.zmq_manager.can_queue_more_jobs() and potentials:
            target_count = 100_000
            job_list = []
            count = 0
            idx = len(potentials) - 1
            while count < target_count and idx >= 0:
                name, (tasks, point_count) = potentials[idx]
                count += point_count
                job_list += [
                    name,
                    self.node_store.get(name),
                    struct.pack('>I', len(tasks)),
                ] + tasks
                del potentials[idx]

                del self.state.node_to_process[name]
                self.state.processing_nodes[name] = (len(tasks), point_count, now)

                if name in self.state.waiting_writing_nodes:
                    self.state.waiting_writing_nodes.pop(self.state.waiting_writing_nodes.index(name))
                idx -= 1

            if job_list:
                self.zmq_manager.send_to_process([CommandType.PROCESS_JOBS.value] + job_list)

    def send_file_to_read(self):
        if self.verbose >= 1:
            print(f'Submit next portion {self.state.point_cloud_file_parts[-1]}')
        file, portion = self.state.point_cloud_file_parts.pop()
        self.state.points_in_progress += portion[1] - portion[0]

        self.zmq_manager.send_to_process([CommandType.READ_FILE.value, pickle.dumps({
            'filename': file,
            'offset_scale': (
                -self.avg_min,
                self.root_scale,
                self.rotation_matrix[:3, :3].T if self.rotation_matrix is not None else None,
                self.infos['color_scale'].get(file) if self.infos['color_scale'] is not None else None,
            ),
            'portion': portion,
        })])

        self.state.number_of_reading_jobs += 1

    def write_tileset(self):
        # compute tile transform matrix
        if self.rotation_matrix is None:
            transform = np.identity(4)
        else:
            transform = inverse_matrix(self.rotation_matrix)
        transform = np.dot(transform, scale_matrix(1.0 / self.root_scale[0]))
        transform = np.dot(translation_matrix(self.avg_min), transform)

        # build fake points
        root_node = Node(''.encode('utf-8'), self.root_aabb, self.root_spacing * 2)
        root_node.children = []
        inv_aabb_size = (1.0 / np.maximum(MIN_POINT_SIZE, self.root_aabb[1] - self.root_aabb[0])).astype(
            np.float32)
        for child in range(8):
            tile_path = node_name_to_path(self.out_folder, str(child).encode('ascii'), '.pnts')
            if tile_path.exists():
                tile_content = TileContentReader.read_file(tile_path)

                fth = tile_content.body.feature_table.header
                xyz = tile_content.body.feature_table.body.positions_arr.view(np.float32).reshape(
                    (fth.points_length, 3))
                if self.rgb:
                    rgb = tile_content.body.feature_table.body.colors_arr.reshape((fth.points_length, 3))
                else:
                    rgb = np.zeros(xyz.shape, dtype=np.uint8)

                root_node.grid.insert(
                    self.root_aabb[0].astype(np.float32),
                    inv_aabb_size,
                    xyz.copy(),
                    rgb)

        pnts_writer.node_to_pnts(''.encode('ascii'), root_node, self.out_folder, self.rgb)

        executor = concurrent.futures.ProcessPoolExecutor()
        root_tileset = Node.to_tileset(executor, ''.encode('ascii'), self.root_aabb, self.root_spacing,
                                       self.out_folder, self.root_scale)
        executor.shutdown()

        root_tileset['transform'] = transform.T.reshape(16).tolist()
        root_tileset['refine'] = 'REPLACE'
        if "children" in root_tileset:
            for child in root_tileset['children']:
                child['refine'] = 'ADD'  # type: ignore

        tileset = {
            'asset': {
                'version': '1.0',
            },
            'geometricError': np.linalg.norm(
                self.root_aabb[1] - self.root_aabb[0]) / self.root_scale[0],
            'root': root_tileset,
        }

        tileset_path = self.out_folder / "tileset.json"
        with tileset_path.open('w') as f:
            f.write(json.dumps(tileset))

    def print_summary(self):
        print('Summary:')
        print('  - points to process: {}'.format(self.infos['point_count']))
        print('  - offset to use: {}'.format(self.avg_min))
        print('  - root spacing: {}'.format(self.root_spacing / self.root_scale[0]))
        print('  - root aabb: {}'.format(self.root_aabb))
        print('  - original aabb: {}'.format(self.original_aabb))
        print('  - scale: {}'.format(self.root_scale))

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
        if self.verbose >= 3:
            print('{:^16}|{:^8}|{:^8}'.format('Name', 'Points', 'Seconds'))
            for name, v in self.state.processing_nodes.items():
                print('{:^16}|{:^8}|{:^8}'.format(
                    '{} ({})'.format(name.decode('ascii'), v[0]),
                    v[1],
                    round(now - v[2], 1)))
            print('')
            print('Pending:')
            print('  - root: {} / {}'.format(
                len(self.state.point_cloud_file_parts),
                self.state.initial_portion_count))
            print('  - other: {} files for {} nodes'.format(
                sum([len(f[0]) for f in self.state.node_to_process.values()]),
                len(self.state.node_to_process)))
            print('')

        elif self.verbose >= 2:
            self.state.print_debug()

        if self.verbose >= 1:
            print('{} % points in {} sec [{} tasks, {} nodes, {} wip]'.format(
                round(100 * self.state.processed_points / self.infos['point_count'], 2),
                round(now, 1),
                self.jobs - len(self.zmq_manager.idle_clients),
                len(self.state.processing_nodes),
                self.state.points_in_progress))

        elif self.verbose >= 0:
            percent = round(100 * self.state.processed_points / self.infos['point_count'], 2)
            time_left = (100 - percent) * now / (percent + 0.001)
            print('\r{:>6} % in {} sec [est. time left: {} sec]'.format(percent, round(now), round(time_left)), end='',
                  flush=True)


def init_parser(subparser):
    parser = subparser.add_parser(
        'convert',
        help='Convert .las files to a 3dtiles tileset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'files',
        nargs='+',
        help='Filenames to process. The file must use the .las, .laz (lastools must be installed) or .xyz format.')
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
                       srs_out=args.srs_out,
                       srs_in=args.srs_in,
                       fraction=args.fraction,
                       benchmark=args.benchmark,
                       rgb=not args.no_rgb,
                       graph=args.graph,
                       color_scale=args.color_scale,
                       verbose=args.verbose)
    except SrsInMissingException:
        print('No SRS information in input files, you should specify it with --srs_in')
        sys.exit(1)
