from __future__ import annotations

import json
from pathlib import Path

from py3dtiles.typing import AssetDictType, GeometricErrorType, TilesetDictType
from .extendable import Extendable
from .tile import Tile


class TileSet(Extendable):
    def __init__(
        self,
        geometric_error: float = 500,
        root_uri: Path | None = None,
    ) -> None:
        super().__init__()
        self._asset: AssetDictType = {"version": "1.0"}
        self.geometric_error: GeometricErrorType = geometric_error
        self.root_tile = Tile()
        self.root_uri = root_uri

    @classmethod
    def from_file(cls, tileset_uri: Path) -> TileSet:
        with tileset_uri.open() as f:
            tileset_dict = json.load(f)

        tileset = TileSet.from_dict(tileset_dict)
        tileset.root_uri = tileset_uri
        return tileset

    @classmethod
    def from_dict(cls, tileset_dict: TilesetDictType) -> TileSet:
        tileset = cls(geometric_error=tileset_dict["geometricError"])

        tileset._asset = tileset_dict["asset"]
        tileset.root_tile = Tile.from_dict(tileset_dict["root"])

        return tileset

    def add_asset_extras(self, comment: str) -> None:
        """
        :param comment: the comment on original data, pre-processing, possible
                        ownership to be added as asset extra.
        """
        self._asset["extras"] = {
            "$schema": "https://json-schema.org/draft-04/schema",
            "title": "Extras",
            "description": comment,
        }

    def delete_on_disk(
        self, tileset_path: Path, delete_tile_content_tileset: bool = False
    ) -> None:
        """
        Deletes all files linked to the tileset. The uri of the tileset should be defined.

        :param delete_tile_content_tileset: If True, all tilesets present as tile content will be removed as well as their content.
        If False, the linked tilesets in tiles won't be removed.
        """
        tileset_path.unlink()
        self.root_tile.delete_on_disk(tileset_path.parent, delete_tile_content_tileset)

    def write_to_directory(self, tileset_path: Path, overwrite: bool = False) -> None:
        """
        Write (or overwrite), to the directory whose name is provided, the
        TileSet that is:
        - the tileset as a json file and
        - all the tiles content of the Tiles used by the Tileset.
        :param tileset_path: the target directory name
        :param overwrite: delete the tileset (and the content) if already exists
        """
        if tileset_path.exists():
            if overwrite:
                tileset = TileSet.from_file(tileset_path)
                tileset.delete_on_disk(tileset_path, delete_tile_content_tileset=True)
            else:
                raise FileExistsError(f"There is a file at {tileset_path}")

        # Proceed with the writing of the TileSet per se:
        self.write_as_json(tileset_path)

        # Prior to writing the TileSet, the future location of the enclosed
        # Tile's content (set as their respective TileContent uri) must be
        # specified:
        all_tiles = self.root_tile.get_all_children()
        all_tiles.append(self.root_tile)
        for tile in all_tiles:
            if tile.tile_content is not None:
                tile.write_content(tileset_path.parent)

    def write_as_json(self, tileset_path: Path) -> None:
        """
        Write the tileset as a JSON file.
        :param tileset_path: the path where the tileset will be written
        """
        with tileset_path.open("w") as f:
            f.write(self.to_json())

    def to_dict(self) -> TilesetDictType:
        """
        Convert to json string possibly mentioning used schemas
        """
        # Make sure the TileSet is aligned with its children Tiles.
        self.root_tile.sync_bounding_volume_with_children()

        dict_data: TilesetDictType = {
            "root": self.root_tile.to_dict(),
            "asset": self._asset,
            "geometricError": self.geometric_error,
        }

        if self._extensions:
            dict_extensions = {}
            for name, extension in self._extensions.items():
                dict_extensions[name] = extension.to_dict()

            dict_data["extensions"] = dict_extensions

        return dict_data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))
