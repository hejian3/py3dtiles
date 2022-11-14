from pathlib import Path
import pickle
import struct
from typing import Tuple, Union

import lz4.frame as gzip
import numpy as np

import py3dtiles
from py3dtiles.tileset.content import PntsBody, PntsHeader
from py3dtiles.tileset.feature_table import FeatureTable, FeatureTableBody, FeatureTableHeader
from py3dtiles.tileset.tile_content import TileContent
from py3dtiles.utils import node_name_to_path, ResponseType


def points_to_pnts(name, points, out_folder: Path, include_rgb) -> Tuple[int, Union[Path, None]]:
    count = int(len(points) / (3 * 4 + (3 if include_rgb else 0)))

    if count == 0:
        return 0, None

    pdt = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
    cdt = np.dtype([('Red', 'u1'), ('Green', 'u1'), ('Blue', 'u1')]) if include_rgb else None

    ft = FeatureTable()
    ft.header = FeatureTableHeader.from_dtype(pdt, cdt, count)
    ft.body = FeatureTableBody.from_array(ft.header, points)

    body = PntsBody()
    body.feature_table = ft

    tile = TileContent()
    tile.body = body
    tile.header = PntsHeader()
    tile.header.sync(body)

    node_path = node_name_to_path(out_folder, name, '.pnts')

    if node_path.exists():
        raise FileExistsError(f"{node_path} already written")

    tile.save_as(node_path)

    return count, node_path


def node_to_pnts(name, node, out_folder: Path, include_rgb):
    points = py3dtiles.tilers.node.Node.get_points(node, include_rgb)
    return points_to_pnts(name, points, out_folder, include_rgb)


def run(sender, data, node_name, folder: Path, write_rgb):
    # we can safely write the .pnts file
    if len(data):
        root = pickle.loads(gzip.decompress(data))
        # print('write ', node_name.decode('ascii'))
        total = 0
        for name in root:
            node = py3dtiles.tilers.node.DummyNode(pickle.loads(root[name]))
            total += node_to_pnts(name, node, folder, write_rgb)[0]

        sender.send_multipart([ResponseType.PNTS_WRITTEN.value, struct.pack('>I', total), node_name])
