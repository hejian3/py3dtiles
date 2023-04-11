from __future__ import annotations

from pathlib import Path
import pickle
from typing import Generator, TYPE_CHECKING

import lz4.frame as gzip
import numpy as np
import numpy.typing as npt

import py3dtiles
from py3dtiles.tileset.batch_table import BatchTable
from py3dtiles.tileset.content import Pnts, PntsBody, PntsHeader
from py3dtiles.tileset.feature_table import (
    FeatureTable,
    FeatureTableBody,
    FeatureTableHeader,
    SemanticPoint,
)
from py3dtiles.utils import node_name_to_path

if TYPE_CHECKING:
    from py3dtiles.tilers.node import DummyNode, Node


def points_to_pnts(
    name: bytes,
    points: npt.NDArray[np.uint8],
    out_folder: Path,
    include_rgb: bool,
    include_classification: bool,
    include_intensity: bool
) -> tuple[int, Path | None]:
    count = int(
        len(points)
        / (3 * 4 + (3 if include_rgb else 0) + (1 if include_classification else 0) + (1 if include_intensity else 0))
    )
    if count == 0:
        return 0, None

    ft = FeatureTable()
    ft.header = FeatureTableHeader.from_semantic(
        SemanticPoint.POSITION, SemanticPoint.RGB if include_rgb else None, None, count
    )
    ft.body = FeatureTableBody.from_array(ft.header, points)

    bt = BatchTable()
    if include_classification:
        sdt = np.dtype([("Classification", "u1")])
        offset = count * (3 * 4 + (3 if include_rgb else 0))
        bt.add_property_as_binary(
            "Classification",
            points[offset : offset + count * sdt.itemsize],
            "UNSIGNED_BYTE",
            "SCALAR",
        )

    if include_intensity:
        sdt = np.dtype([("Intensity", "u1")])
        offset = count * (3 * 4 + (3 if include_rgb else 0) + (1 if include_classification else 0))
        bt.add_property_as_binary(
            "Intensity",
            points[offset: offset + count * sdt.itemsize],
            "UNSIGNED_BYTE",
            "SCALAR",
        )

    body = PntsBody()
    body.feature_table = ft
    body.batch_table = bt

    tile = Pnts(PntsHeader(), body)
    tile.sync()

    node_path = node_name_to_path(out_folder, name, ".pnts")

    if node_path.exists():
        raise FileExistsError(f"{node_path} already written")

    tile.save_as(node_path)

    return count, node_path


def node_to_pnts(
    name: bytes,
    node: Node | DummyNode,
    out_folder: Path,
    include_rgb: bool,
    include_classification: bool,
    include_intensity: bool,
) -> int:
    points = py3dtiles.tilers.node.Node.get_points(
        node, include_rgb, include_classification, include_intensity
    )
    point_nb, _ = points_to_pnts(
        name, points, out_folder, include_rgb, include_classification, include_intensity
    )
    return point_nb


def run(
    data: bytes,
    folder: Path,
    write_rgb: bool,
    write_classification: bool,
    write_intensity: bool
) -> Generator[int, None, None]:
    # we can safely write the .pnts file
    if len(data) > 0:
        root = pickle.loads(gzip.decompress(data))
        total = 0
        for name in root:
            node = py3dtiles.tilers.node.DummyNode(pickle.loads(root[name]))
            total += node_to_pnts(name, node, folder, write_rgb, write_classification, write_intensity)
        yield total
