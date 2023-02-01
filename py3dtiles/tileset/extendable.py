from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py3dtiles.tileset.extension import BaseExtension


class Extendable:
    """
    One the 3DTiles notions defined as an abstract data model through
    a schema of the 3DTiles specifications (either core of extensions).
    """

    def __init__(self) -> None:
        self._extensions: dict[str, BaseExtension] = {}

    def add_extension(self, extension: BaseExtension) -> None:
        if not self.has_extensions():
            self._extensions = dict()
        self._extensions[extension.name] = extension

    def has_extensions(self) -> bool:
        return len(self._extensions) != 0

    def get_extensions(self) -> list[BaseExtension]:
        if not self.has_extensions():
            return list()
        return list(self._extensions.values())

    def get_extension(self, extension_name: str) -> BaseExtension:
        if not self.has_extensions():
            raise AttributeError('No extension present.')
        if extension_name not in self._extensions:
            raise ValueError(f'No extension with name {extension_name}.')
        return self._extensions[extension_name]
