import math
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from pyproj import Transformer

from py3dtiles.typing import (
    MetadataReaderType,
    OffsetScaleType,
    PortionItemType,
    PortionsType,
)


def get_metadata(path: Path, fraction: int = 100) -> MetadataReaderType:
    """Get metadata in case of a input ply file."""
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

    data = np.array(
        [ply_vertices["x"], ply_vertices["y"], ply_vertices["z"]]
    ).transpose()
    aabb = np.array((np.min(data, axis=0), np.max(data, axis=0)))

    pointcloud_file_portions: PortionsType = [(str(path), (0, point_count))]

    return {
        "portions": pointcloud_file_portions,
        "aabb": aabb,
        "crs_in": None,
        "point_count": point_count,
        "avg_min": aabb[0],
    }


def run(
    filename: str,
    offset_scale: OffsetScaleType,
    portion: PortionItemType,
    transformer: Optional[Transformer],
    color_scale: Optional[float],
) -> Generator[Tuple, None, None]:
    """
    Reads points from a ply file.
    """
    ply_point_cloud = PlyData.read(filename)
    ply_vertices = ply_point_cloud["vertex"]

    point_count = portion[1] - portion[0]
    step = min(point_count, max(point_count // 10, 100_000))
    indices = list(range(math.ceil(point_count / step)))

    for index in indices:
        start_offset = portion[0] + index * step
        num = min(step, portion[1] - start_offset)

        x = ply_vertices["x"][start_offset : (start_offset + num)]
        y = ply_vertices["y"][start_offset : (start_offset + num)]
        z = ply_vertices["z"][start_offset : (start_offset + num)]
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

        # NOTE this code assume all the colors have the same type
        # I think it's a reasonable assumption to make at this point but it's
        # not mandated by the spec!
        if "red" in ply_vertices:
            # val_dtype is of the form: i<nbytes>, u<nbytes>, float<nbytes>
            # see https://github.com/dranjan/python-plyfile/blob/d1f73004ed0a296fc8b9c1fad8139b5d90410639/plyfile.py#L32
            signed = ply_vertices.ply_property("red").val_dtype[0:1] == "i"
            nbytes = int(ply_vertices.ply_property("red").val_dtype[1:])
            # factor is how much we need to divide the data
            # nbytes - 1 is the number of "extra" bytes compared to the 1 byte target
            # but we remove one extra bit if it is signed
            # max, because if the value is *shorter* we don't need to do anything)
            factor = max(1, 2 ** (8 * (nbytes - 1) - (1 if signed else 0)))
            red = ply_vertices["red"][start_offset : (start_offset + num)] / factor
            green = ply_vertices["green"][start_offset : (start_offset + num)] / factor
            blue = ply_vertices["blue"][start_offset : (start_offset + num)] / factor
        else:
            red = green = blue = np.zeros(num)

        raw_colors = np.vstack((red, green, blue)).transpose()

        if color_scale is not None:
            raw_colors = np.clip(raw_colors * color_scale, 0, 255)

        colors = raw_colors.astype(np.uint8)

        if "classification" in ply_vertices:
            classification = np.array(
                ply_vertices["classification"].reshape(-1, 1), dtype=np.uint8
            )
        else:
            classification = np.zeros((coords.shape[0], 1), dtype=np.uint8)

        yield coords, colors, classification


def create_plydata_with_renamed_property(
    plydata: PlyData, old_property_name: str, new_property_name: str
) -> PlyData:
    """Create a new plyfile.PlyData object on the model of the one provided as input, with a modified
    feature name.

    This function may be useful for handling classification feature name in .ply files, knowing
    that this feature is not formalized.

    """
    ply_data = plydata["vertex"].data
    copied_data = ply_data.copy()
    ptype = np.dtype(
        [
            (new_property_name, t[1]) if t[0] == old_property_name else t
            for t in ply_data.dtype.descr
        ]
    )
    pelement = PlyElement.describe(data=copied_data.astype(ptype), name="vertex")
    return PlyData(elements=[pelement])
