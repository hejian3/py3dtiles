import concurrent.futures
import json
from pathlib import Path
import pickle
import struct

from lz4 import frame
import numpy as np
from pyproj import CRS, Transformer

from py3dtiles.exceptions import SrsInMissingException, SrsInMixinException
from py3dtiles.tilers.matrix_manipulation import make_rotation_matrix, make_scale_matrix, make_translation_matrix
from py3dtiles.tileset.utils import TileContentReader
from py3dtiles.utils import CommandType, node_name_to_path, ResponseType, str_to_CRS
from .constants import MIN_POINT_SIZE
from .pnts_node import DummyNode, PntsNode
from .pnts_writer import node_to_pnts


def compute_spacing(aabb: np.ndarray) -> float:
    return float(np.linalg.norm(aabb[1] - aabb[0]) / 125)

class PntsMetadata:
    def __init__(self):
        self.num_of_files = 0
        self.aabb = None
        self.write_rgb = False
        self.color_scale = {}
        self.crs_in = None
        self.point_count = 0
        self.avg_min = np.array([0., 0., 0.])
        self.transformer = None
        self.rotation_matrix = None
        self.original_aabb = None
        self.root_aabb = None
        self.root_scale = None
        self.root_spacing = None
        self.transformer = None
        self.rotation_matrix = None

    def update(self, file: Path, file_info: dict) -> list:
        self.num_of_files += 1
        self.point_count += file_info['point_count']

        if self.aabb is None:
            self.aabb = file_info['aabb']
        else:
            self.aabb[0] = np.minimum(self.aabb[0], file_info['aabb'][0])
            self.aabb[1] = np.maximum(self.aabb[1], file_info['aabb'][1])

        self.color_scale[str(file)] = file_info['color_scale']

        file_crs_in = str_to_CRS(file_info['srs_in'])
        if file_crs_in is not None:
            if self.crs_in is None:
                self.crs_in = file_crs_in
            elif self.crs_in != file_crs_in:
                raise SrsInMixinException("All input files should have the same srs in, currently there are a mix of"
                                 f" {self.crs_in} and {file_crs_in}")

        # update the avg_min, since we don't know in advance how many there are pnts files
        self.avg_min = (self.avg_min * (self.num_of_files - 1) + file_info['avg_min']) / self.num_of_files

        return file_info['portions']

    def get_transformer(self, crs_out: CRS) -> None:
        if crs_out:
            if self.crs_in is None:
                raise SrsInMissingException("None file has a input srs specified. Should be provided.")

            self.transformer = Transformer.from_crs(self.crs_in, crs_out)

    def get_rotation_matrix(self, crs_out: CRS) -> None:
        if self.transformer:
            bl = np.array(list(self.transformer.transform(
                self.aabb[0][0], self.aabb[0][1], self.aabb[0][2])))
            tr = np.array(list(self.transformer.transform(
                self.aabb[1][0], self.aabb[1][1], self.aabb[1][2])))
            br = np.array(list(self.transformer.transform(
                self.aabb[1][0], self.aabb[0][1], self.aabb[0][2])))

            self.avg_min = np.array(list(self.transformer.transform(
                self.avg_min[0], self.avg_min[1], self.avg_min[2])))

            x_axis = br - bl

            bl = bl - self.avg_min
            tr = tr - self.avg_min

            if crs_out.to_epsg() == 4978:
                # Transform geocentric normal => (0, 0, 1)
                # and 4978-bbox x axis => (1, 0, 0),
                # to have a bbox in local coordinates that's nicely aligned with the data
                self.rotation_matrix = make_rotation_matrix(self.avg_min, np.array([0, 0, 1]))
                self.rotation_matrix = np.dot(
                    make_rotation_matrix(x_axis, np.array([1, 0, 0])),
                    self.rotation_matrix)

                bl = np.dot(bl, self.rotation_matrix[:3, :3].T)
                tr = np.dot(tr, self.rotation_matrix[:3, :3].T)

            self.original_aabb = np.array([
                np.minimum(bl, tr),
                np.maximum(bl, tr)
            ])
        else:
            # offset
            self.original_aabb = self.aabb - self.avg_min

    def get_root_aabb(self) -> None:
        base_spacing = compute_spacing(self.original_aabb)
        if base_spacing > 10:
            self.root_scale = np.array([0.01, 0.01, 0.01])
        elif base_spacing > 1:
            self.root_scale = np.array([0.1, 0.1, 0.1])
        else:
            self.root_scale = np.array([1, 1, 1])

        self.root_aabb = self.original_aabb * self.root_scale
        self.root_spacing = compute_spacing(self.root_aabb)

    def __bool__(self):
        return self.num_of_files != 0



class PntsState:
    def __init__(self, file_portions: list, node_store, max_reading_jobs: int):
        self.node_store = node_store

        self.processed_points = 0
        self.max_point_in_progress = 60_000_000
        self.points_in_progress = 0
        self.points_in_pnts = 0

        # pointcloud_file_portions is a list of tuple (filename, (start offset, end offset))
        self.point_cloud_file_parts = file_portions
        self.initial_portion_count = len(file_portions)
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


class PntsActions:
    @staticmethod
    def dispatch_processed_nodes(content, pnts_state):
        if not content['name']:
            return

        pnts_state.node_store.put(content['name'], content['save'])
        pnts_state.waiting_writing_nodes.append(content['name'])

        if not pnts_state.is_reading_finish():
            return

        # if all nodes aren't processed yet,
        # we should check if linked ancestors are processed
        if pnts_state.processing_nodes or pnts_state.node_to_process:
            finished_node = content['name']
            if can_pnts_be_written(
                finished_node, finished_node,
                pnts_state.node_to_process, pnts_state.processing_nodes
            ):
                pnts_state.waiting_writing_nodes.pop(-1)
                pnts_state.pnts_to_writing.append(finished_node)

                for i in range(len(pnts_state.waiting_writing_nodes) - 1, -1, -1):
                    candidate = pnts_state.waiting_writing_nodes[i]

                    if can_pnts_be_written(
                        candidate, finished_node,
                        pnts_state.node_to_process, pnts_state.processing_nodes
                    ):
                        pnts_state.waiting_writing_nodes.pop(i)
                        pnts_state.pnts_to_writing.append(candidate)

        else:
            for c in pnts_state.waiting_writing_nodes:
                pnts_state.pnts_to_writing.append(c)
            pnts_state.waiting_writing_nodes.clear()

    @staticmethod
    def create_command_pnts_to_write(pnts_state: PntsState) -> list:
        node_name = pnts_state.pnts_to_writing.pop()
        data = pnts_state.node_store.get(node_name)
        if not data:
            raise ValueError(f'{node_name} has no data')

        pnts_state.node_store.remove(node_name)
        pnts_state.number_of_writing_jobs += 1

        return [CommandType.WRITE_PNTS.value, node_name, data]


    @staticmethod
    def send_points_to_process(pnts_state, zmq_manager, now):
        potentials = sorted(
            # a key (=task) can be in node_to_process and processing_nodes if the node isn't completely processed
            [
                (node, task) for node, task in pnts_state.node_to_process.items()  # task: [data...], point_count
                if node not in pnts_state.processing_nodes
            ],
            key=lambda task: -len(task[0]))  # sort by node name size, the root nodes first

        while zmq_manager.can_queue_more_jobs() and potentials:
            target_count = 100_000
            job_list = []
            count = 0
            idx = len(potentials) - 1
            while count < target_count and idx >= 0:
                name, (tasks, point_count) = potentials[idx]
                count += point_count
                job_list += [
                    name,
                    pnts_state.node_store.get(name),
                    struct.pack('>I', len(tasks)),
                ] + tasks
                del potentials[idx]

                del pnts_state.node_to_process[name]
                pnts_state.processing_nodes[name] = (len(tasks), point_count, now)

                if name in pnts_state.waiting_writing_nodes:
                    pnts_state.waiting_writing_nodes.pop(pnts_state.waiting_writing_nodes.index(name))
                idx -= 1

            if job_list:
                zmq_manager.send_to_process([CommandType.PROCESS_JOBS.value] + job_list)

    @staticmethod
    def create_command_send_file_to_read(pnts_state, pnts_metadata):
        # if self.verbose >= 1:
        #     print(f'Submit next portion {self.pnts_state.point_cloud_file_parts[-1]}')
        file, portion = pnts_state.point_cloud_file_parts.pop()
        pnts_state.points_in_progress += portion[1] - portion[0]

        pnts_state.number_of_reading_jobs += 1

        return [CommandType.READ_FILE.value, pickle.dumps({
            'filename': file,
            'offset_scale': (
                -pnts_metadata.avg_min,
                pnts_metadata.root_scale,
                pnts_metadata.rotation_matrix[:3, :3].T if pnts_metadata.rotation_matrix is not None else None,
                pnts_metadata.color_scale.get(file) if pnts_metadata.color_scale is not None else None,
            ),
            'portion': portion,
        })]

    @staticmethod
    def write_pnts(sender, data, node_name, folder: Path, write_rgb):
        # we can safely write the .pnts file
        if len(data):
            root = pickle.loads(frame.decompress(data))
            # print('write ', node_name.decode('ascii'))
            total = 0
            for name in root:
                node = DummyNode(pickle.loads(root[name]))
                total += node_to_pnts(name, node, folder, write_rgb)[0]

            sender.send_multipart([ResponseType.PNTS_WRITTEN.value, struct.pack('>I', total), node_name])

    @staticmethod
    def write_tileset(pnts_metadata: PntsMetadata, out_folder: Path) -> Path:
        if pnts_metadata.root_aabb is None:
            raise AttributeError()

        # compute tile transform matrix
        if pnts_metadata.rotation_matrix is None:
            transform = np.identity(4)
        else:
            transform = np.linalg.inv(pnts_metadata.rotation_matrix)

        transform = np.dot(transform, make_scale_matrix(1.0 / pnts_metadata.root_scale[0]))
        transform = np.dot(make_translation_matrix(pnts_metadata.avg_min), transform)

        # Create the root tile by sampling (or taking all points?) of child nodes
        root_node = PntsNode(b'', pnts_metadata.root_aabb, pnts_metadata.root_spacing * 2)
        root_node.children = []
        inv_aabb_size = (1.0 / np.maximum(MIN_POINT_SIZE, pnts_metadata.root_aabb[1] - pnts_metadata.root_aabb[0])).astype(
            np.float32)
        for child in range(8):
            tile_path = node_name_to_path(out_folder, str(child).encode('ascii'), '.pnts')
            if tile_path.exists():
                tile_content = TileContentReader.read_file(tile_path)

                fth = tile_content.body.feature_table.header
                xyz = tile_content.body.feature_table.body.positions_arr.view(np.float32).reshape(
                    (fth.points_length, 3))
                if pnts_metadata.write_rgb:
                    rgb = tile_content.body.feature_table.body.colors_arr.reshape((fth.points_length, 3))
                else:
                    rgb = np.zeros(xyz.shape, dtype=np.uint8)

                root_node.grid.insert(
                    pnts_metadata.root_aabb[0].astype(np.float32),
                    inv_aabb_size,
                    xyz.copy(),
                    rgb)

        node_to_pnts(b'', root_node, out_folder, pnts_metadata.write_rgb)

        executor = concurrent.futures.ProcessPoolExecutor()
        root_tileset = PntsNode.to_tileset(executor, b'', pnts_metadata.root_aabb, pnts_metadata.root_spacing,
                                           out_folder, pnts_metadata.root_scale, prune=False)
        executor.shutdown()

        root_tileset['transform'] = transform.T.reshape(16).tolist()
        root_tileset['refine'] = 'REPLACE'  # The root tile is in the "REPLACE" refine mode
        if "children" in root_tileset:
            # And children with the "ADD" refine mode
            # No need to set this property in their children, they will take the parent value if it is not present
            for child in root_tileset['children']:
                child['refine'] = 'ADD'  # type: ignore

        tileset = {
            'asset': {
                'version': '1.0',
            },
            'geometricError': np.linalg.norm(
                pnts_metadata.root_aabb[1] - pnts_metadata.root_aabb[0]) / pnts_metadata.root_scale[0],
            'root': root_tileset,
        }

        tileset_path = out_folder / "tileset_pnts.json"
        with tileset_path.open('w') as f:
            json.dump(tileset, f)

        return tileset_path


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
