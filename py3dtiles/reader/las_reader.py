import json
import math
from pathlib import Path
import pickle
import struct
import subprocess

import laspy
import numpy as np

from py3dtiles.utils import ResponseType


def get_metadata(path: Path, color_scale=None, fraction: int = 100) -> dict:
    pointcloud_file_portions = []
    srs_in = None

    filename = str(path)
    with laspy.open(filename) as f:
        point_count = f.header.point_count * fraction // 100

        # read the first points red channel
        if not color_scale:
            if 'red' in f.header.point_format.dimension_names:
                points = next(f.chunk_iterator(10_000))['red']
                if np.max(points) > 255:
                    color_scale = 1.0 / 255
            else:
                # the intensity is then used as color
                color_scale = 1.0 / 255

        _1M = min(point_count, 1_000_000)
        steps = math.ceil(point_count / _1M)
        portions = [(i * _1M, min(point_count, (i + 1) * _1M)) for i in range(steps)]
        for p in portions:
            pointcloud_file_portions += [(filename, p)]

        output = subprocess.check_output(['pdal', 'info', '--summary', filename])
        summary = json.loads(output)['summary']
        if 'srs' in summary:
            srs_in = summary['srs'].get('proj4')

    return {
        'portions': pointcloud_file_portions,
        'aabb': np.array([f.header.mins, f.header.maxs]),
        'color_scale': color_scale,
        'srs_in': srs_in,
        'point_count': point_count,
        'avg_min': np.array(f.header.mins)
    }


def run(filename: str, offset_scale, portion, queue, transformer):
    """
    Reads points from a las file
    """
    try:
        with laspy.open(filename) as f:

            point_count = portion[1] - portion[0]

            step = min(point_count, max(point_count // 10, 100_000))

            indices = [i for i in range(math.ceil(point_count / step))]

            color_scale = offset_scale[3]

            for index in indices:
                start_offset = portion[0] + index * step
                num = min(step, portion[1] - start_offset)

                # read scaled values and apply offset
                f.seek(start_offset)
                points = next(f.chunk_iterator(num))

                x, y, z = points.x, points.y, points.z
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

                # todo: attributes
                if 'red' in f.header.point_format.dimension_names:
                    red = points['red']
                    green = points['green']
                    blue = points['blue']
                else:
                    red = points['intensity']
                    green = points['intensity']
                    blue = points['intensity']

                if not color_scale:
                    red = red.astype(np.uint8)
                    green = green.astype(np.uint8)
                    blue = blue.astype(np.uint8)
                else:
                    red = (red * color_scale).astype(np.uint8)
                    green = (green * color_scale).astype(np.uint8)
                    blue = (blue * color_scale).astype(np.uint8)

                colors = np.vstack((red, green, blue)).transpose()

                queue.send_multipart(
                    [
                        ResponseType.NEW_TASK.value,
                        b'',
                        pickle.dumps({'xyz': coords, 'rgb': colors}),
                        struct.pack('>I', len(coords))
                    ], copy=False)

            queue.send_multipart([ResponseType.READ.value])

    except Exception as e:
        print(f'Exception while reading points from las file {filename}')
        raise e
