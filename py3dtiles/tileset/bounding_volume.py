from abc import ABC

from .extendable import Extendable


class BoundingVolume(ABC, Extendable):
    """
    Abstract class used as interface for box, region and sphere
    """
    def __init__(self):
        super().__init__()

    def is_box(self) -> bool:
        return False

    def is_region(self) -> bool:
        return False

    def is_sphere(self) -> bool:
        return False

    def clone(self: "BoundingVolume") -> "BoundingVolume":
        return self

    def __add__(self, other: "BoundingVolume") -> "BoundingVolume":
        if not self.is_box():
            raise NotImplementedError()
        if not other.is_box():
            raise NotImplementedError()

        return self + other
