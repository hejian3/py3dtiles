import json
import numpy as np
import math
import traceback
import laspy
import struct
import subprocess
from pickle import dumps as pdumps

from py3dtiles.utils import SrsInMissingException


def init(files, color_scale=None, srs_in=None, srs_out=None, fraction=100):
    aabb = None
    total_point_count = 0
    pointcloud_file_portions = []
    avg_min = np.array([0., 0., 0.])
    color_scale_by_file = {}

    for filename in files:
        try:
            with laspy.open(filename) as f:
                avg_min += (np.array(f.header.mins) / len(files))

                if aabb is None:
                    aabb = np.array([f.header.mins, f.header.maxs])
                else:
                    bb = np.array([f.header.mins, f.header.maxs])
                    aabb[0] = np.minimum(aabb[0], bb[0])
                    aabb[1] = np.maximum(aabb[1], bb[1])

                count = f.header.point_count * fraction // 100
                total_point_count += count

                # read the first points red channel
                if color_scale is not None:
                    color_scale_by_file[filename] = color_scale
                elif 'red' in f.header.point_format.dimension_names:
                    points = next(f.chunk_iterator(10000))['red']
                    if np.max(points) > 255:
                        color_scale_by_file[filename] = 1.0 / 255
                else:
                    # the intensity is then used as color
                    color_scale_by_file[filename] = 1.0 / 255

                _1M = min(count, 1000000)
                steps = math.ceil(count / _1M)
                portions = [(i * _1M, min(count, (i + 1) * _1M)) for i in range(steps)]
                for p in portions:
                    pointcloud_file_portions += [(filename, p)]

                if (srs_out is not None and srs_in is None):
                    # NOTE: decode is necessary because in python3.5, json cannot decode bytes. Remove this once 3.5 is EOL
                    output = subprocess.check_output(['pdal', 'info', '--summary', filename]).decode('utf-8')
                    summary = json.loads(output)['summary']
                    if 'srs' not in summary or 'proj4' not in summary['srs'] or not summary['srs']['proj4']:
                        raise SrsInMissingException('\'{}\' file doesn\'t contain srs information. Please use the --srs_in option to declare it.'.format(filename))
                    srs_in = summary['srs']['proj4']
        except Exception as e:
            print('Error opening {filename}. Skipping.'.format(**locals()))
            print(e)
            continue

    return {
        'portions': pointcloud_file_portions,
        'aabb': aabb,
        'color_scale': color_scale_by_file,
        'srs_in': srs_in,
        'point_count': total_point_count,
        'avg_min': avg_min
    }


def run(_id, filename, offset_scale, portion, queue, transformer, verbose):
    '''
    Reads points from a las file
    '''
    try:
        with laspy.open(filename) as f:

            point_count = portion[1] - portion[0]

            step = min(point_count, max((point_count) // 10, 100000))

            indices = [i for i in range(math.ceil((point_count) / step))]

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

                if color_scale is None:
                    red = red.astype(np.uint8)
                    green = green.astype(np.uint8)
                    blue = blue.astype(np.uint8)
                else:
                    red = (red * color_scale).astype(np.uint8)
                    green = (green * color_scale).astype(np.uint8)
                    blue = (blue * color_scale).astype(np.uint8)

                colors = np.vstack((red, green, blue)).transpose()

                queue.send_multipart([
                    ''.encode('ascii'),
                    pdumps({'xyz': coords, 'rgb': colors}),
                    struct.pack('>I', len(coords))], copy=False)

            queue.send_multipart([pdumps({'name': _id, 'total': 0})])
            # notify we're idle
            queue.send_multipart([b''])

    except Exception as e:
        print('Exception while reading points from las file')
        print(e)
        traceback.print_exc()
