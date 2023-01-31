from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import pickle
from typing import Any, Iterator, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from py3dtiles.tilers.pnts import MIN_POINT_SIZE
from py3dtiles.tilers.pnts.pnts_writer import points_to_pnts
from py3dtiles.tileset.feature_table import SemanticPoint
from py3dtiles.tileset.utils import TileContentReader
from py3dtiles.utils import aabb_size_to_subdivision_type, node_from_name, node_name_to_path, SubdivisionType
from .distance import xyz_to_child_index
from .points_grid import Grid

if TYPE_CHECKING:
    from .node_catalog import NodeCatalog


def node_to_tileset(args):
    return Node.to_tileset(None, args[0], args[1], args[2], args[3], args[4])


class DummyNode:
    def __init__(self, _bytes):
        if 'children' in _bytes:
            self.children = _bytes['children']
            self.grid = _bytes['grid']
        else:
            self.children = None
            self.points = _bytes['points']


class Node:
    """docstring for Node"""
    __slots__ = (
        'name', 'aabb', 'aabb_size', 'inv_aabb_size', 'aabb_center',
        'spacing', 'pending_xyz', 'pending_rgb', 'children', 'grid',
        'points', 'dirty')

    def __init__(self, name: bytes, aabb: np.ndarray, spacing: float) -> None:
        super().__init__()
        self.name = name
        self.aabb = aabb.astype(np.float32)
        self.aabb_size = np.maximum(aabb[1] - aabb[0], MIN_POINT_SIZE).astype(np.float32)
        self.inv_aabb_size = (1.0 / self.aabb_size).astype(np.float32)
        self.aabb_center = ((aabb[0] + aabb[1]) * 0.5).astype(np.float32)
        self.spacing = spacing
        self.pending_xyz: list[npt.NDArray] = []
        self.pending_rgb: list[npt.NDArray] = []
        self.children: list[bytes] | None = None
        self.grid = Grid(self)
        self.points: list[tuple[npt.NDArray, npt.NDArray]] = []
        self.dirty = False

    def save_to_bytes(self) -> bytes:
        sub_pickle = {}
        if self.children is not None:
            sub_pickle['children'] = self.children # type: ignore # TODO fix
            sub_pickle['grid'] = self.grid # type: ignore # TODO fix
        else:
            sub_pickle['points'] = self.points # type: ignore # TODO fix

        d = pickle.dumps(sub_pickle)
        return d

    def load_from_bytes(self, byt: bytes) -> None:
        sub_pickle = pickle.loads(byt)
        if 'children' in sub_pickle:
            self.children = sub_pickle['children']
            self.grid = sub_pickle['grid']
        else:
            self.points = sub_pickle['points']

    def insert(self, node_catalog: NodeCatalog, scale: float, xyz: npt.NDArray, rgb: npt.NDArray, make_empty_node: bool = False):
        if make_empty_node:
            self.children = []
            self.pending_xyz += [xyz]
            self.pending_rgb += [rgb]
            return

        # fastpath
        if self.children is None:
            self.points.append((xyz, rgb))
            count = sum([xyz.shape[0] for xyz, rgb in self.points])
            # stop subdividing if spacing is 1mm
            if count >= 20000 and self.spacing > 0.001 * scale:
                self._split(node_catalog, scale)
            self.dirty = True

            return True

        # grid based insertion
        reminder_xyz, reminder_rgb, needs_balance = self.grid.insert(
            self.aabb[0], self.inv_aabb_size, xyz, rgb)

        if needs_balance:
            self.grid.balance(self.aabb_size, self.aabb[0], self.inv_aabb_size)
            self.dirty = True

        self.dirty = self.dirty or (len(reminder_xyz) != len(xyz))

        if len(reminder_xyz) > 0:
            self.pending_xyz += [reminder_xyz]
            self.pending_rgb += [reminder_rgb]

    def needs_balance(self) -> bool:
        if self.children is not None:
            return self.grid.needs_balance()
        return False

    def flush_pending_points(self, catalog: NodeCatalog, scale: float) -> None:
        for name, xyz, rgb in self._get_pending_points():
            catalog.get_node(name).insert(catalog, scale, xyz, rgb)
        self.pending_xyz = []
        self.pending_rgb = []

    def dump_pending_points(self) -> list[tuple[bytes, bytes, int]]:
        result = [
            (name, pickle.dumps({'xyz': xyz, 'rgb': rgb}), len(xyz))
            for name, xyz, rgb in self._get_pending_points()
        ]

        self.pending_xyz = []
        self.pending_rgb = []
        return result

    def get_pending_points_count(self) -> int:
        return sum([xyz.shape[0] for xyz in self.pending_xyz])

    def _get_pending_points(self) -> Iterator[tuple[bytes, np.ndarray, np.ndarray]]:
        if not self.pending_xyz:
            return

        pending_xyz_arr = np.concatenate(self.pending_xyz)
        pending_rgb_arr = np.concatenate(self.pending_rgb)
        t = aabb_size_to_subdivision_type(self.aabb_size)
        if t == SubdivisionType.QUADTREE:
            indices = xyz_to_child_index(
                pending_xyz_arr,
                np.array(
                    [self.aabb_center[0], self.aabb_center[1], self.aabb[1][2]],
                    dtype=np.float32)
            )
        else:
            indices = xyz_to_child_index(pending_xyz_arr, self.aabb_center)

        # unique children list
        childs = np.unique(indices)

        # make sure all children nodes exist
        for child in childs:
            name = '{}{}'.format(self.name.decode('ascii'), child).encode('ascii')
            # create missing nodes, only for remembering they exist.
            # We don't want to serialize them
            # probably not needed...
            if self.children is not None and name not in self.children :
                self.children += [name]
                self.dirty = True
                # print('Added node {}'.format(name))

            mask = np.where(indices - child == 0)
            xyz = pending_xyz_arr[mask]
            if len(xyz) > 0:
                yield name, xyz, pending_rgb_arr[mask]

    def _split(self, node_catalog: NodeCatalog, scale: float) -> None:
        self.children = []
        for xyz, rgb in self.points:
            self.insert(node_catalog, scale, xyz, rgb)
        self.points = []

    def get_point_count(self, node_catalog: NodeCatalog, max_depth: int, depth: int = 0) -> int:
        if self.children is None:
            return sum([xyz.shape[0] for xyz, rgb in self.points])
        else:
            count = self.grid.get_point_count()
            if depth < max_depth:
                for n in self.children:
                    count += node_catalog.get_node(n).get_point_count(
                        node_catalog, max_depth, depth + 1)
            return count

    @staticmethod
    def get_points(data: Node | DummyNode, include_rgb: bool) -> np.ndarray:  # todo remove staticmethod
        if data.children is None:
            points = data.points
            xyz = np.concatenate(tuple([xyz for xyz, rgb in points])).view(np.uint8).ravel()
            if include_rgb:
                rgb = np.concatenate(tuple([rgb for xyz, rgb in points])).ravel()
                result = np.concatenate((xyz, rgb))
                return result
            else:
                return xyz
        else:
            return data.grid.get_points(include_rgb)

    @staticmethod
    def to_tileset(executor: concurrent.futures.ProcessPoolExecutor | None,
                   name: bytes,
                   parent_aabb: np.ndarray,
                   parent_spacing: float,
                   folder: Path,
                   scale: np.ndarray,
                   prune: bool = True) -> dict:
        node = node_from_name(name, parent_aabb, parent_spacing)
        aabb = node.aabb
        tile_path = node_name_to_path(folder, name, '.pnts')
        xyz = np.array(0)
        rgb = np.array(0)

        # Read tile's pnts file, if existing, we'll need it for:
        #   - computing the real AABB (instead of the one based on the octree)
        #   - merging this tile's small (<100 points) children
        if tile_path.exists():
            tile = TileContentReader.read_file(tile_path)

            fth = tile.body.feature_table.header
            xyz = tile.body.feature_table.body.positions_arr
            if fth.colors != SemanticPoint.NONE:
                rgb = tile.body.feature_table.body.colors_arr
            xyz_float = xyz.view(np.float32).reshape((fth.points_length, 3))
            # update aabb based on real values
            aabb = np.array([
                np.amin(xyz_float, axis=0),
                np.amax(xyz_float, axis=0)])

        # geometricError is in meters, so we divide it by the scale
        tileset = {'geometricError': 10 * node.spacing / scale[0]}

        children: list[Any] = [] # todo at the next refacto, fix it
        tile_needs_rewrite = False
        if tile_path.exists():
            tileset['content'] = {'uri': str(tile_path.relative_to(folder))}
        for child in ['0', '1', '2', '3', '4', '5', '6', '7']:
            child_name = '{}{}'.format(
                name.decode('ascii'),
                child
            ).encode('ascii')
            child_tile_path = node_name_to_path(folder, child_name, '.pnts')

            if child_tile_path.exists():
                # See if we should merge this child in tile
                if len(xyz):
                    # Read pnts content
                    tile = TileContentReader.read_file(child_tile_path)

                    fth = tile.body.feature_table.header

                    # If this child is small enough, merge in the current tile.
                    # prune should be set at False is the refine mode is REPLACE.
                    # In some cases, we cannot know which point in the parent tile should be deleted
                    # (for example when 2 points are at the same location)
                    if prune and fth.points_length < 100:
                        xyz = np.concatenate(
                            (xyz,
                             tile.body.feature_table.body.positions_arr))

                        if fth.colors != SemanticPoint.NONE:
                            rgb = np.concatenate(
                                (rgb,
                                 tile.body.feature_table.body.colors_arr))

                        # update aabb
                        xyz_float = tile.body.feature_table.body.positions_arr.view(
                            np.float32).reshape((fth.points_length, 3))

                        aabb[0] = np.amin(
                            [aabb[0], np.min(xyz_float, axis=0)], axis=0)
                        aabb[1] = np.amax(
                            [aabb[1], np.max(xyz_float, axis=0)], axis=0)

                        tile_needs_rewrite = True
                        child_tile_path.unlink()
                        continue

                # Add child to the to-be-processed list if it hasn't been merged
                if executor is not None:
                    children += [(child_name, node.aabb, node.spacing, folder, scale)]
                else:
                    children += [Node.to_tileset(None, child_name, node.aabb, node.spacing, folder, scale)]

        # If we merged at least one child tile in the current tile
        # the pnts file needs to be rewritten.
        if tile_needs_rewrite:
            tile_path.unlink()
            points_to_pnts(name, np.concatenate((xyz, rgb)), folder, len(rgb) != 0)

        center = ((aabb[0] + aabb[1]) * 0.5).tolist()
        half_size = ((aabb[1] - aabb[0]) * 0.5).tolist()
        tileset['boundingVolume'] = {
            'box': [
                center[0], center[1], center[2],
                half_size[0], 0, 0,
                0, half_size[1], 0,
                0, 0, half_size[2]]
        }

        if executor is not None:
            children = [t for t in executor.map(node_to_tileset, children)]

        if children:
            tileset['children'] = children
        else:
            tileset['geometricError'] = 0.0

        if len(name) > 0 and children:
            if len(json.dumps(tileset)) > 100000:
                tile_root = {
                    'asset': {
                        'version': '1.0',
                    },
                    'refine': 'ADD',
                    'geometricError': tileset['geometricError'],
                    'root': tileset
                }
                tileset_name = f"tileset.{name.decode('ascii')}.json"
                tileset_path = folder / tileset_name
                with tileset_path.open('w') as f:
                    json.dump(tile_root, f)
                tileset['content'] = {'uri': tileset_name}
                del tileset['children']

        return tileset
