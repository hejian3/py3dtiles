import pickle
import time
from typing import Generator, List, Optional, TextIO, Tuple

from py3dtiles.tilers.node.node import Node
from py3dtiles.tilers.node.node_catalog import NodeCatalog
from py3dtiles.utils import OctreeMetadata


def _flush(
    node_catalog: NodeCatalog,
    scale: float,
    node: Node,
    max_depth: int = 1,
    force_forward: bool = False,
    log_file: Optional[TextIO] = None,
    depth: int = 0,
) -> Generator[Tuple[bytes, bytes, int], None, None]:
    if depth >= max_depth:
        threshold = 0 if force_forward else 10_000
        if node.get_pending_points_count() > threshold:
            yield from node.dump_pending_points()
        return

    node.flush_pending_points(node_catalog, scale)
    if node.children is not None:
        # then flush children
        children = node.children
        # release node
        del node
        for child_name in children:
            yield from _flush(
                node_catalog,
                scale,
                node_catalog.get_node(child_name),
                max_depth,
                force_forward,
                log_file,
                depth + 1,
            )


def _balance(node_catalog, node, max_depth=1, depth=0):
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
            _balance(
                node_catalog, node_catalog.get_node(child_name), max_depth, depth + 1
            )


def infer_depth_from_name(name: bytes) -> int:
    halt_at_depth = 0
    if len(name) >= 7:
        halt_at_depth = 5
    elif len(name) >= 5:
        halt_at_depth = 3
    elif len(name) > 2:
        halt_at_depth = 2
    elif len(name) >= 1:
        halt_at_depth = 1
    return halt_at_depth


def run(
    node_catalog: NodeCatalog,
    octree_metadata: OctreeMetadata,
    name: bytes,
    tasks: List[bytes],
    begin: float,
    log_file: Optional[TextIO],
) -> Generator[Tuple[bytes, bytes, int, int], None, None]:

    log_enabled = log_file is not None

    if log_enabled:
        print(f'[>] process_node: "{name!r}", {len(tasks)}', file=log_file, flush=True)

    node = node_catalog.get_node(name)

    total = 0
    halt_at_depth = infer_depth_from_name(name)

    for index, task in enumerate(tasks):
        if log_enabled:
            print(
                f"  -> read source [{time.time() - begin}]", file=log_file, flush=True
            )

        data = pickle.loads(task)

        point_count = len(data["xyz"])

        if log_enabled:
            print(
                "  -> insert {} [{} points]/ {} files [{}]".format(
                    index + 1, point_count, len(tasks), time.time() - begin
                ),
                file=log_file,
                flush=True,
            )

        # insert points in node (no children handling here)
        node.insert(
            octree_metadata.scale,
            data["xyz"],
            data["rgb"],
            data["classification"],
            halt_at_depth == 0,
        )

        total += point_count

        if log_enabled:
            print(f"  -> _flush [{time.time() - begin}]", file=log_file, flush=True)
        # _flush push pending points (= call insert) from level N to level N + 1
        # (_flush is recursive)
        for flush_name, flush_data, flush_point_count in _flush(
            node_catalog,
            octree_metadata.scale,
            node,
            halt_at_depth - 1,
            index == len(tasks) - 1,
            log_file,
        ):
            total -= flush_point_count
            yield flush_name, flush_data, flush_point_count, total

    _balance(node_catalog, node, halt_at_depth - 1)
