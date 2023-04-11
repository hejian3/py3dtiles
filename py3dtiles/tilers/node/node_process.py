import pickle
import time
from typing import Generator, List, Optional, TextIO, Tuple

from py3dtiles.tilers.node.node import Node
from py3dtiles.tilers.node.node_catalog import NodeCatalog
from py3dtiles.utils import OctreeMetadata


class NodeProcess:
    def __init__(
        self,
        node_catalog: NodeCatalog,
        octree_metadata: OctreeMetadata,
        name: bytes,
        tasks: List,
        begin: float,
        log_file: Optional[TextIO],
    ):
        self.node_catalog = node_catalog
        self.octree_metadata = octree_metadata
        self.name = name
        self.tasks = tasks
        self.begin = begin
        self.log_file = log_file
        self.total_point_count = 0

    def _flush(
        self,
        node: Node,
        max_depth: int = 1,
        force_forward: bool = False,
        depth: int = 0,
    ) -> Generator[Tuple, None, None]:
        if depth >= max_depth:
            threshold = 0 if force_forward else 10_000
            if node.get_pending_points_count() > threshold:
                yield from node.dump_pending_points()
            return

        node.flush_pending_points(self.node_catalog, self.octree_metadata.scale)
        if node.children is not None:
            # then flush children
            children = node.children
            # release node
            del node
            for child_name in children:
                yield from self._flush(
                    self.node_catalog.get_node(child_name),
                    max_depth,
                    force_forward,
                    depth + 1,
                )

    def _balance(self, node, max_depth=1, depth=0):
        if depth >= max_depth:
            return 0

        if node.needs_balance():
            node.grid.balance(node.aabb_size, node.aabb[0], node.inv_aabb_size)
            node.dirty = True

        if node.children is not None:
            # then _balance children
            children = node.children
            # release node
            del node
            for child_name in children:
                self._balance(
                    self.node_catalog.get_node(child_name), max_depth, depth + 1
                )

    def infer_depth_from_name(self) -> int:
        halt_at_depth = 0
        if len(self.name) >= 7:
            halt_at_depth = 5
        elif len(self.name) >= 5:
            halt_at_depth = 3
        elif len(self.name) > 2:
            halt_at_depth = 2
        elif len(self.name) >= 1:
            halt_at_depth = 1
        return halt_at_depth

    def run(self) -> Generator[Tuple, None, None]:

        log_enabled = self.log_file is not None

        if log_enabled:
            print(
                f'[>] process_node: "{self.name!r}", {len(self.tasks)}',
                file=self.log_file,
                flush=True,
            )

        node = self.node_catalog.get_node(self.name)

        halt_at_depth = self.infer_depth_from_name()

        for index, task in enumerate(self.tasks):
            if log_enabled:
                print(
                    f"  -> read source [{time.time() - self.begin}]",
                    file=self.log_file,
                    flush=True,
                )

            data = pickle.loads(task)

            point_count = len(data["xyz"])

            if log_enabled:
                print(
                    "  -> insert {} [{} points]/ {} files [{}]".format(
                        index + 1,
                        point_count,
                        len(self.tasks),
                        time.time() - self.begin,
                    ),
                    file=self.log_file,
                    flush=True,
                )

            # insert points in node (no children handling here)
            node.insert(
                self.octree_metadata.scale,
                data["xyz"],
                data["rgb"],
                data["classification"],
                data["intensity"],
                halt_at_depth == 0,
            )

            self.total_point_count += point_count

            if log_enabled:
                print(
                    f"  -> _flush [{time.time() - self.begin}]",
                    file=self.log_file,
                    flush=True,
                )
            # _flush push pending points (= call insert) from level N to level N + 1
            # (_flush is recursive)
            for flush_name, flush_data, flush_point_count in self._flush(
                node,
                halt_at_depth - 1,
                index == len(self.tasks) - 1,
            ):
                self.total_point_count -= flush_point_count
                yield flush_name, flush_data, flush_point_count

        self._balance(node, halt_at_depth - 1)
