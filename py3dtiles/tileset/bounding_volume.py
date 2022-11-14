from .extendable import Extendable


class BoundingVolume(Extendable):
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
