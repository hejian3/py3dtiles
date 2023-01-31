from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from py3dtiles.typing import BoundingVolumeDictType
from .extendable import Extendable

if TYPE_CHECKING:
    from py3dtiles.tileset.tile import Tile


class BoundingVolume(ABC, Extendable):
    """
    Abstract class used as interface for box, region and sphere
    """
    def __init__(self) -> None:
        super().__init__()

    def is_box(self) -> bool:
        return False

    def is_region(self) -> bool:
        return False

    def is_sphere(self) -> bool:
        return False

    @abstractmethod
    def to_dict(self) -> BoundingVolumeDictType: ...

    @abstractmethod
    def sync_with_children(self, owner: Tile) -> None: ...

    @abstractmethod
    def transform(self, transform: npt.NDArray[np.float64]) -> None: ...

    @abstractmethod
    def add(self, other: BoundingVolume) -> None: ...
