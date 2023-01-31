from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt


class TileContent:
    def __init__(self):
        self.header = None
        self.body = None

    def to_array(self) -> npt.NDArray[np.uint8]:
        self.sync()
        header_arr = self.header.to_array()
        body_arr = self.body.to_array()
        return np.concatenate((header_arr, body_arr))

    def to_hex_str(self):
        arr = self.to_array()
        return " ".join(f"{x:02X}" for x in arr)

    def save_as(self, path: Path):
        tile_arr = self.to_array()
        with path.open('bw') as f:
            f.write(bytes(tile_arr))

    def sync(self):
        """
        Allow to synchronize headers with contents.
        """
        self.header.sync(self.body)


class TileContentType(Enum):
    UNKNOWN = 0
    POINT_CLOUD = 1
    BATCHED_3D_MODEL = 2


class TileContentHeader(ABC):
    def __init__(self):
        self.tile_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0
        self.bt_length = 0  # number of models in the batch

    @staticmethod
    @abstractmethod
    def from_array(array: npt.NDArray) -> TileContentHeader: ...

    @abstractmethod
    def to_array(self) -> npt.NDArray: ...

    @abstractmethod
    def sync(self, body) -> None: ...


class TileContentBody(ABC):
    @abstractmethod
    def to_array(self) -> npt.NDArray: ...
