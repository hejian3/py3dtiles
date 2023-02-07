from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from py3dtiles.tileset.content import B3dm, Pnts

if TYPE_CHECKING:
    from .tile_content import TileContent


class TileContentReader:

    @staticmethod
    def read_file(tile_path: Path) -> TileContent:
        with tile_path.open('rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)

            tile_content = TileContentReader.read_array(arr)
            if tile_content is None or tile_content.header is None:
                raise ValueError(f"The file {tile_path} doesn't contain a valid TileContent data.")

            return tile_content

    @staticmethod
    def read_array(array: np.ndarray) -> TileContent | None:
        magic = ''.join([c.decode('UTF-8') for c in array[0:4].view('c')])
        if magic == 'pnts':
            return Pnts.from_array(array)
        if magic == 'b3dm':
            return B3dm.from_array(array)
        return None


def number_of_points_in_tileset(tileset_path: Path) -> int:
    with tileset_path.open() as f:
        tileset = json.load(f)

    nb_points = 0

    children_tileset_info = [(tileset["root"], tileset["root"]["refine"])]
    while children_tileset_info:
        child_tileset, parent_refine = children_tileset_info.pop()
        child_refine = child_tileset["refine"] if child_tileset.get("refine") else parent_refine

        if "content" in child_tileset:
            content = tileset_path.parent / child_tileset["content"]['uri']

            pnts_should_count = "children" not in child_tileset or child_refine == "ADD"
            if content.suffix == '.pnts' and pnts_should_count:
                tile = TileContentReader.read_file(content)
                nb_points += tile.body.feature_table.nb_points()
            elif content.suffix == '.json':
                with content.open() as f:
                    sub_tileset = json.load(f)
                children_tileset_info.append((sub_tileset["root"], child_refine))

        if "children" in child_tileset:
            children_tileset_info += [
                (sub_child_tileset, child_refine) for sub_child_tileset in child_tileset["children"]
            ]

    return nb_points
