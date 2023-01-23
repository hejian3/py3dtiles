import math
from pathlib import Path
import pickle
import struct
from typing import Any, Dict

import numpy as np
from plyfile import PlyData

from py3dtiles.utils import ResponseType


def get_metadata(path: Path, color_scale=None, fraction: int = 100) -> Dict[str, Any]:
    """Get metadata in case of a input ply file.
    """
    ply_point_cloud = PlyData.read(path)
    if "vertex" not in [e.name for e in ply_point_cloud.elements]:
        raise KeyError(
            "The ply data does not contain any 'vertex' item. Are you sure the file is valid?"
        )
    ply_vertices = ply_point_cloud["vertex"]
    point_count = ply_vertices.count * fraction // 100
    ply_features = [ply_prop.name for ply_prop in ply_vertices.properties]
    if any(coord not in ply_features for coord in ("x", "y", "z")):
        raise KeyError(
            "At least one of the basic coordinate feature (x, y, z) is missing in the input file."
        )
    data = np.array([ply_vertices["x"], ply_vertices["y"], ply_vertices["z"]]).transpose()
    aabb = np.min(data, axis=0), np.max(data, axis=0)

    pointcloud_file_portions = [(str(path), (0, point_count, point_count))]

    return {
        'portions': pointcloud_file_portions,
        'aabb': aabb,
        'color_scale': color_scale,
        'srs_in': None,
        'point_count': point_count,
        'avg_min': aabb[0],
    }


def run(filename: str, offset_scale, portion, queue, transformer):
    """
    Reads points from a ply file.
    """
    try:
        ply_point_cloud = PlyData.read(filename)
        ply_vertices = ply_point_cloud["vertex"]

        point_count = portion[1] - portion[0]
        step = min(point_count, max(point_count // 10, 100_000))
        indices = [i for i in range(math.ceil(point_count / step))]
        color_scale = offset_scale[3]

        for index in indices:
            start_offset = portion[0] + index * step
            num = min(step, portion[1] - start_offset)

            x = ply_vertices["x"][start_offset:(start_offset + num)]
            y = ply_vertices["y"][start_offset:(start_offset + num)]
            z = ply_vertices["z"][start_offset:(start_offset + num)]
            if transformer:
                x, y, z = transformer.transform(x, y, z)

            x = (x + offset_scale[0][0]) * offset_scale[1][0]
            y = (y + offset_scale[0][1]) * offset_scale[1][1]
            z = (z + offset_scale[0][2]) * offset_scale[1][2]

            coords = np.vstack((x, y, z)).transpose()

            if offset_scale[2] is not None:
                # Apply transformation matrix (because the tile's transform will contain
                # the inverse of this matrix)
                coords = np.dot(coords, offset_scale[2])

            coords = np.ascontiguousarray(coords.astype(np.float32))

            # Read colors
            if 'red' in ply_vertices:
                red = ply_vertices["red"]
                green = ply_vertices["green"]
                blue = ply_vertices["blue"]
            else:
                red = green = blue = np.zeros(num)

            if not color_scale:
                red = red.astype(np.uint8)
                green = green.astype(np.uint8)
                blue = blue.astype(np.uint8)
            else:
                red = (red * color_scale).astype(np.uint8)
                green = (green * color_scale).astype(np.uint8)
                blue = (blue * color_scale).astype(np.uint8)

            colors = np.vstack((red, green, blue)).transpose()
            colors = colors[start_offset:(start_offset + num)]

            queue.send_multipart(
                [
                    ResponseType.NEW_TASK.value,
                    b'',
                    pickle.dumps({'xyz': coords, 'rgb': colors}),
                    struct.pack('>I', len(coords))
                ], copy=False)

        queue.send_multipart([ResponseType.READ.value])

    except Exception as e:
        print(f'Exception while reading points from ply file {filename}')
        raise e
