from abc import ABC, abstractmethod

from py3dtiles.tileset.tile import Tile
from py3dtiles.typing import TilesetDictPart
from .extendable import Extendable


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
    def to_dict(self) -> TilesetDictPart: ...

    @abstractmethod
    def sync_with_children(self, owner: Tile) -> None: ...
