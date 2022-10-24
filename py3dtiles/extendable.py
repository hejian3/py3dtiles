from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .extension import Extension


class Extendable:
    """
    One the 3DTiles notions defined as an abstract data model through
    a schema of the 3DTiles specifications (either core of extensions).
    """

    def __init__(self):
        self._extensions = {}

    def add_extension(self, extension: Extension) -> None:
        if not self.has_extensions():
            self._extensions = dict()
        self._extensions[extension.name] = extension

    def has_extensions(self) -> bool:
        return len(self._extensions) != 0

    def get_extensions(self) -> List[Extension]:
        if not self.has_extensions():
            return list()
        return list(self._extensions.values())

    def get_extension(self, extension_name: str) -> Extension:
        if not self.has_extensions():
            raise AttributeError('No extension present.')
        if extension_name not in self._extensions:
            raise ValueError(f'No extension with name {extension_name}.')
        return self._extensions[extension_name]
