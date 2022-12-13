from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pyproj import CRS, Transformer

from .b3dm import B3dm
from .pnts import Pnts

if TYPE_CHECKING:
    from . import TileContent



def convert_to_ecef(x, y, z, epsg_input):
    crs_in = CRS(f'epsg:{epsg_input}')
    crs_out = CRS('epsg:4978')  # ECEF
    transformer = Transformer.from_crs(crs_in, crs_out)
    return transformer.transform(x, y, z)


def str_to_CRS(srs: str | CRS | None) -> CRS | None:
    """
    Convert a string in a pyproj CRS object. The string could be an epsg code or a Proj4 string. If srs is already
    a CRS object, the function returns the same CRS definition.
    """
    if srs is None:
        return None

    try:
        return CRS.from_epsg(int(srs))
    except ValueError:
        return CRS(srs)


class TileContentReader:

    @staticmethod
    def read_file(tile_path: Path) -> TileContent:
        with tile_path.open('rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)

            tile_content = TileContentReader.read_array(arr)
            if tile_content is None or tile_content.header is None:
                raise ValueError(f"The file {tile_path} doesn't contain a valid TileContent data.")

            return tile_content

    @staticmethod
    def read_array(array: np.ndarray) -> TileContent | None:
        magic = ''.join([c.decode('UTF-8') for c in array[0:4].view('c')])
        if magic == 'pnts':
            return Pnts.from_array(array)
        if magic == 'b3dm':
            return B3dm.from_array(array)
        return None
