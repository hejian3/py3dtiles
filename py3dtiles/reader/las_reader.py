import math
from pathlib import Path
from typing import Generator, Optional, Tuple

import laspy
import numpy as np
from pyproj import Transformer

from py3dtiles.typing import MetadataReaderType, OffsetScaleType, PortionType


def get_metadata(path: Path, fraction: int = 100) -> MetadataReaderType:
    pointcloud_file_portions = []

    filename = str(path)
    with laspy.open(filename) as f:
        point_count = f.header.point_count * fraction // 100

        _1M = min(point_count, 1_000_000)
        steps = math.ceil(point_count / _1M)
        portions = [(i * _1M, min(point_count, (i + 1) * _1M)) for i in range(steps)]
        for p in portions:
            pointcloud_file_portions += [(filename, p)]

        crs_in = f.header.parse_crs()

    return {
        "portions": pointcloud_file_portions,
        "aabb": np.array([f.header.mins, f.header.maxs]),
        "crs_in": crs_in,
        "point_count": point_count,
        "avg_min": np.array(f.header.mins),
    }


def run(
    filename: str,
    offset_scale: OffsetScaleType,
    portion: PortionType,
    transformer: Optional[Transformer],
    color_scale: Optional[float],
) -> Generator[Tuple, None, None]:
    """
    Reads points from a las file
    """
    with laspy.open(filename) as f:

        point_count = portion[1] - portion[0]

        step = min(point_count, max(point_count // 10, 100_000))

        indices = list(range(math.ceil(point_count / step)))

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
            if "red" in f.header.point_format.dimension_names:
                red = points["red"]
                green = points["green"]
                blue = points["blue"]
            else:
                red = points["intensity"]
                green = points["intensity"]
                blue = points["intensity"]

            colors = np.vstack((red, green, blue)).transpose()

            if color_scale is not None:
                colors = np.clip(colors * color_scale, 0, 65535)

            # NOTE las spec says rgb is 16bits by components
            # pnts are 8 bits (by default) by component, hence we divide by 256
            colors = (colors / 256).astype(np.uint8)

            if "classification" in f.header.point_format.dimension_names:
                classification = np.array(
                    points["classification"], dtype=np.uint8
                ).reshape(-1, 1)
            else:
                classification = np.zeros((len(points.x), 1), dtype=np.uint8)

            yield coords, colors, classification
