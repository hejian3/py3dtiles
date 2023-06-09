from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, TYPE_CHECKING

from py3dtiles.typings import AssetDictType, GeometricErrorType, TilesetDictType
from .extendable import Extendable
from .tile import Tile

if TYPE_CHECKING:
    from .content import TileContent


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
    def from_dict(cls, tileset_dict: TilesetDictType, root_uri: Path) -> TileSet:
        tileset = cls(geometric_error=tileset_dict["geometricError"])

        tileset._asset = tileset_dict["asset"]
        tileset.root_tile = Tile.from_dict(tileset_dict["root"])
        tileset.root_uri = root_uri

        return tileset

    @staticmethod
    def from_file(filepath: Path) -> TileSet:
        with open(filepath) as f:
            tileset_dict = json.load(f)
        return TileSet.from_dict(tileset_dict, filepath.parent)

    def get_all_tile_contents(
        self,
    ) -> Generator[TileContent | TileSet | None, None, None]:
        tiles = [self.root_tile] + self.root_tile.get_all_children()
        for tile in tiles:
            yield tile.get_or_fetch_content(self.root_uri)

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

    def write_to_directory(self, directory: Path) -> None:
        """
        Write (or overwrite), to the directory whose name is provided, the
        TileSet that is:
        - the tileset as a json file and
        - all the tiles content of the Tiles used by the Tileset.
        :param directory: the target directory name
        """
        # Create the output directory
        target_dir = directory.expanduser()
        tiles_dir = target_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        # Prior to writing the TileSet, the future location of the enclosed
        # Tile's content (set as their respective TileContent uri) must be
        # specified:
        all_tiles = self.root_tile.get_all_children()
        for index, tile in enumerate(all_tiles):
            tile.content_uri = Path("tiles") / f"{index}.b3dm"

        # Proceed with the writing of the TileSet per se:
        self.write_as_json(target_dir)

        # Terminate with the writing of the tiles content:
        for tile in all_tiles:
            tile.write_content(directory)

    def write_as_json(self, directory: Path) -> None:
        """
        Write the tileset as a JSON file.
        :param directory: the target directory name
        """
        # Make sure the TileSet is aligned with its children Tiles.
        self.root_tile.sync_bounding_volume_with_children()

        tileset_path = directory / "tileset.json"
        with tileset_path.open("w") as f:
            f.write(self.to_json())

    def to_dict(self) -> TilesetDictType:
        """
        Convert to json string possibly mentioning used schemas
        """

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
