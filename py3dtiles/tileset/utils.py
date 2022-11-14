from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from py3dtiles.tileset.content import B3dm, Pnts

if TYPE_CHECKING:
    from .tile_content import TileContent


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
