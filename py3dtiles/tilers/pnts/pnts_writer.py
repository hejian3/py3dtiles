from __future__ import annotations

from pathlib import Path
import pickle
from typing import Generator, TYPE_CHECKING

import lz4.frame as gzip
import numpy as np
import numpy.typing as npt

import py3dtiles
from py3dtiles.tileset.content import Pnts
from py3dtiles.utils import node_name_to_path

if TYPE_CHECKING:
    from py3dtiles.tilers.node import DummyNode, Node


def points_to_pnts_file(
    name: bytes,
    points: npt.NDArray[np.uint8],
    out_folder: Path,
    include_rgb: bool,
    include_classification: bool,
) -> tuple[int, Path]:

    pnts = Pnts.from_points(points, include_rgb, include_classification)

    node_path = node_name_to_path(out_folder, name, ".pnts")

    if node_path.exists():
        raise FileExistsError(f"{node_path} already written")

    pnts.save_as(node_path)

    return pnts.body.feature_table.nb_points(), node_path


def node_to_pnts(
    name: bytes,
    node: Node | DummyNode,
    out_folder: Path,
    include_rgb: bool,
    include_classification: bool,
) -> int:
    points = py3dtiles.tilers.node.Node.get_points(
        node, include_rgb, include_classification
    )
    point_nb, _ = points_to_pnts_file(
        name, points, out_folder, include_rgb, include_classification
    )
    return point_nb


def run(
    data: bytes,
    folder: Path,
    write_rgb: bool,
    write_classification: bool,
) -> Generator[int, None, None]:
    # we can safely write the .pnts file
    if len(data) > 0:
        root = pickle.loads(gzip.decompress(data))
        total = 0
        for name in root:
            node = py3dtiles.tilers.node.DummyNode(pickle.loads(root[name]))
            total += node_to_pnts(name, node, folder, write_rgb, write_classification)
        yield total
