from pathlib import Path
import pickle
import struct

import lz4.frame as gzip
import numpy as np

import py3dtiles
from py3dtiles.points.utils import name_to_filename, ResponseType


def points_to_pnts(name, points, out_folder: Path, include_rgb):
    count = int(len(points) / (3 * 4 + (3 if include_rgb else 0)))

    if count == 0:
        return 0, None

    pdt = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
    cdt = np.dtype([('Red', 'u1'), ('Green', 'u1'), ('Blue', 'u1')]) if include_rgb else None

    ft = py3dtiles.feature_table.FeatureTable()
    ft.header = py3dtiles.feature_table.FeatureTableHeader.from_dtype(pdt, cdt, count)
    ft.body = py3dtiles.feature_table.FeatureTableBody.from_array(ft.header, points)

    body = py3dtiles.pnts.PntsBody()
    body.feature_table = ft

    tile = py3dtiles.tile_content.TileContent()
    tile.body = body
    tile.header = py3dtiles.pnts.PntsHeader()
    tile.header.sync(body)

    filename = name_to_filename(out_folder, name, '.pnts')

    if filename.exists():
        raise FileExistsError(f"{filename} already written")

    tile.save_as(filename)

    return count, filename


def node_to_pnts(name, node, out_folder: Path, include_rgb):
    points = py3dtiles.points.node.Node.get_points(node, include_rgb)  # type: ignore
    return points_to_pnts(name, points, out_folder, include_rgb)


def run(sender, data, node_name, folder: Path, write_rgb):
    # we can safely write the .pnts file
    if len(data):
        root = pickle.loads(gzip.decompress(data))
        # print('write ', node_name.decode('ascii'))
        total = 0
        for name in root:
            node = py3dtiles.points.node.DummyNode(pickle.loads(root[name]))  # type: ignore
            total += node_to_pnts(name, node, folder, write_rgb)[0]

        sender.send_multipart([ResponseType.PNTS_WRITTEN.value, struct.pack('>I', total), node_name])
