from py3dtiles.typing import ExtensionDictType


class BaseExtension:
    """
    An instance of some ExtensionType.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def to_dict(self) -> ExtensionDictType:
        return {}
