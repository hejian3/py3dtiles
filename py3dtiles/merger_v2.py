from pathlib import Path
from typing import List

import numpy as np

from py3dtiles.tileset.tileset import TileSet


def merger_v2(tileset_paths: List[Path], out_folder: Path):
    if not tileset_paths:
        return

    if len(tileset_paths) == 1:
        tileset_paths[0].rename(tileset_paths[0].parent / "tileset.json")
        return

    global_tileset = TileSet()
    for tileset_path in tileset_paths:
        tileset = TileSet.from_file(tileset_path)

        tileset.root_tile.set_content_uri(tileset_path.relative_to(out_folder)) # todo always relative to ?
        tileset.root_tile.geometric_error = tileset.geometric_error # todo not this

        tileset.root_tile.bounding_volume.transform(tileset.root_tile.get_transform())
        tileset.root_tile.set_transform(None)

        global_tileset.root_tile.add_child(tileset.root_tile)

    for child in global_tileset.root_tile.get_direct_children():
        biggest_geometric_error = max(global_tileset.geometric_error, child.geometric_error)

    global_tileset.geometric_error = biggest_geometric_error
    global_tileset.root_tile.geometric_error = biggest_geometric_error

    global_tileset.root_tile.set_refine_mode("REPLACE")

    global_tileset.write_as_json(out_folder)
