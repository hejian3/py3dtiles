from __future__ import annotations

from pathlib import Path

from .extendable import Extendable
from .tile_content import TileContent, TileUri


class Tile(Extendable):

    def __init__(self, geometric_error=500, bounding_volume=None, refine_mode="ADD"):
        super().__init__()
        self.bounding_volume = bounding_volume
        self.geometric_error = geometric_error
        self._refine = ""
        self.set_refine_mode(refine_mode)
        self._content: TileContent | TileUri | None = None
        self._children = []
        # Some possible valid properties left un-delt with viewerRequestVolume
        self._transform = None

    def set_transform(self, transform: list[float] | None) -> None:
        """
        :param transform: a flattened transformation matrix
        :return:
        """
        self._transform = transform

    def get_transform(self) -> list[float]:
        if self._transform is not None:
            return self._transform
        return [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]

    def set_content(self, content: TileContent, force=True) -> None:
        if not force and self._content is not None:
            return
        self._content = content

    def get_content(self) -> TileContent | TileUri | None:
        return self._content

    def set_content_uri(self, uri: Path) -> None:
        self._content = TileUri(uri)

    def set_refine_mode(self, mode: str) -> None:
        if mode != 'ADD' and mode != 'REPLACE':
            raise ValueError(f"Unknown refinement mode {mode}. Should be either 'ADD' or 'REPLACE'.")
        self._refine = mode

    def get_refine_mode(self) -> str:
        return self._refine

    def add_child(self, tile: Tile) -> None:
        self._children.append(tile)

        if tile.bounding_volume is not None:
            if self.bounding_volume is None:
                self.bounding_volume = tile.bounding_volume.clone()
            else:
                self.bounding_volume.add(tile.bounding_volume)

    def has_children(self) -> bool:
        return len(self._children) != 0

    def get_direct_children(self) -> list[Tile]:
        return self._children

    def get_children(self) -> list[Tile]:
        """
        :return: the recursive (across the children tree) list of the children
                 tiles
        """
        descendants = []
        for child in self._children:
            # Add the child...
            descendants.append(child)
            # and if (and only if) they are grand-children then recurse
            if child.has_children():
                descendants += child.get_children()
        return descendants

    def sync_bounding_volume_with_children(self) -> None:
        if self.bounding_volume is None:
            raise AttributeError('This Tile has no bounding volume: exiting.')
        if not self.bounding_volume.is_box():
            raise NotImplementedError("Don't know how to sync non box bounding volume.")

        # We consider that whatever information is present it is the
        # proper one (in other terms: when they are no sub-tiles this tile
        # is a leaf-tile and thus is has no synchronization to do)
        for child in self.get_direct_children():
            child.sync_bounding_volume_with_children()

        # The information that depends on (is defined by) the children
        # nodes is limited to be bounding volume.
        self.bounding_volume.sync_with_children(self)

    def write_content(self, directory: Path) -> None:
        """
        Write (or overwrite) the tile _content_ to the directory specified
        as parameter and withing the relative filename designated by
        the tile's content uri. Note that it is the responsibility of the
        owning TileSet to
        - set those uris
        - to explicitly invoke write_content() (this is to be opposed with
        the Tile attributes which get serialized when recursing on the
        TileSet attributes)
        :param directory: the target directory
        """
        if self._content is None or isinstance(self._content, TileUri):
            raise AttributeError("Tile with no content.")

        file_name = self._content.get_uri()
        if not file_name:
            raise AttributeError("Tile with no content or no uri in content.")

        path_name = directory / file_name

        # Make sure the output directory exists (note that target_dir may
        # be a sub-directory of 'directory' because the uri might hold its
        # own path):
        path_name.parent.mkdir(parents=True, exist_ok=True)

        # Write the tile content of this tile:
        with path_name.open('wb') as f:
            f.write(self._content.to_array())

    def to_dict(self) -> dict:
        dict_data = {}

        if self.bounding_volume is not None:
            dict_data['boundingVolume'] = self.bounding_volume.to_dict()
        else:
            raise AttributeError('Bounding volume is not set')

        if self._refine:
            dict_data['refine'] = self._refine

        if self._transform:
            dict_data['transform'] = self._transform


        dict_data['geometricError'] = self.geometric_error

        if self._children:
            # The children list exists indeed (for technical reasons) yet it
            # happens to be still empty. This would pollute the json output
            # by adding a "children" entry followed by an empty list. In such
            # case just remove that attributes entry:
            dict_data['children'] = [child.to_dict() for child in self._children]

        if self._content is not None:
            # Refer to children related above comment (mutatis mutandis):
            dict_data['content'] = {'uri': self._content.get_uri()}

        return dict_data
