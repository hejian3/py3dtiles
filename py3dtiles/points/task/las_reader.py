import json
import math
import pickle
import struct
import subprocess
from typing import Dict, List, Tuple

import laspy
import numpy as np

from py3dtiles.points.utils import ResponseType
from py3dtiles.utils import SrsInMissingException


def get_metadata(filename: str, color_scale, fraction: int = 100, max_batch_size: int = 1_000_000) -> Tuple[List, Dict]:
    with laspy.open(filename) as f:
        point_count = f.header.point_count * fraction // 100

        batch_size = min(point_count, max_batch_size)
        steps = math.ceil(point_count / batch_size)
        portions = [
            (
                filename,
                (i * batch_size, min(point_count, (i + 1) * batch_size))
            )
            for i in range(steps)
        ]

        metadata = {
            'min': np.array(f.header.mins),
            'aabb': np.array([f.header.mins, f.header.maxs]),
            'point_count': point_count
        }

        if color_scale:
            metadata['color_scale'] = color_scale
        # read the first points red channel
        elif 'red' in f.header.point_format.dimension_names:
            points = next(f.chunk_iterator(10_000))['red']
            if np.max(points) > 255:
                metadata['color_scale'] = 1.0 / 255  # ??? there is a case where no color scale defined ???
            else:
                metadata['color_scale'] = None  # not sure about that...
        else:
            # the intensity is then used as color
            metadata['color_scale'] = 1.0 / 255

        output = subprocess.check_output(['pdal', 'info', '--summary', filename])
        summary = json.loads(output)['summary']
        if 'srs' in summary and 'proj4' in summary['srs']:
            metadata['source_srs'] = summary['srs']['proj4']

        return portions, metadata


def run(filename, offset_scale, portion, queue, transformer, verbose):
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
                        ''.encode('ascii'),
                        pickle.dumps({'xyz': coords, 'rgb': colors}),
                        struct.pack('>I', len(coords))
                    ], copy=False)

            queue.send_multipart([ResponseType.READ.value])

    except Exception as e:
        print(f'Exception while reading points from las file {filename}')
        raise e
