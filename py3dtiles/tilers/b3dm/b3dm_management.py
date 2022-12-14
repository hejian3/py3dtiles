import json
import math
from pathlib import Path

import numpy as np

from py3dtiles.tileset.content import B3dm, GlTF
from .b3dm_node import B3dmNode, BoundingBox
from .wkb_utils import TriangleSoup


class B3dmState:
    def __init__(self, file_portions: list):
        self.file_portion = file_portions
        self.tree = None

class B3dmMetadata:
    def __init__(self):
        self.offset = (0, 0, 0)
        self.transform = np.array([
            [1, 0, 0, self.offset[0]],
            [0, 1, 0, self.offset[1]],
            [0, 0, 1, self.offset[2]],
            [0, 0, 0, 1]], dtype=float).flatten('F')
        self.has_data = False

    def update(self, file: Path, file_info: dict) -> list:
        self.has_data = True
        return file_info['portions']

    def __bool__(self):
        return self.has_data

# TODO manage offset
# transform = np.array([
#         [1, 0, 0, offset[0]],
#         [0, 1, 0, offset[1]],
#         [0, 0, 1, offset[2]],
#         [0, 0, 0, 1]], dtype=float)
#     transform = transform.flatten('F')
# used by B3dmMetadata.transform

class B3dmActions:
    @staticmethod
    def send_build_tree(b3dm_state: B3dmState, out_folder: Path, now) -> None: # TODO execute this in a process
        wkbs = []
        for file in b3dm_state.file_portion:
            with open(file, 'rb') as f: # TODO use wkb_reader and event better, map_reader
                wkbs.append(f.read())

        geoms = [TriangleSoup.from_wkb_multipolygon(wkb) for wkb in wkbs]
        positions = [ts.get_position_array() for ts in geoms]
        normals = [ts.get_normal_array() for ts in geoms]
        bboxes = [ts.get_bbox() for ts in geoms]
        max_tile_size = 2000
        features_per_tile = 20
        indices = [i for i in range(len(positions))]

        # glTF is Y-up, so to get the bounding boxes in the 3D tiles
        # coordinate system, we have to apply a Y-to-Z transform to the
        # glTF bounding boxes
        z_up_bboxes = []
        for bbox in bboxes:
            bbox_min = bbox[0]
            bbox_max = bbox[1]

            # swap y/z axis and invert z.
            bbox_min, bbox_max = (bbox_min[0], -bbox_max[2], bbox_min[1]), (bbox_max[0], -bbox_min[2], bbox_max[1])
            z_up_bboxes.append((bbox_min, bbox_max))

        # Compute extent
        x_min = y_min = float('inf')
        x_max = y_max = - float('inf')

        for bbox in z_up_bboxes:
            x_min = min(x_min, bbox[0][0])
            y_min = min(y_min, bbox[0][1])
            x_max = max(x_max, bbox[1][0])
            y_max = max(y_max, bbox[1][1])
        extent = BoundingBox([x_min, y_min], [x_max, y_max])
        extent_x = x_max - x_min
        extent_y = y_max - y_min

        # Create quadtree
        tree = B3dmNode()
        # TODO this should be deleted / moved in divide (quadtree)
        for i in range(0, math.floor(extent_x / max_tile_size) + 1):
            for j in range(0, math.floor(extent_y / max_tile_size) + 1):
                tile = tile_extent(extent, max_tile_size, i, j)

                geoms = []
                for idx, box in zip(indices, z_up_bboxes):
                    bbox = BoundingBox(box[0], box[1])

                    if tile.inside(bbox.center()):
                        geoms.append(Feature(idx, bbox))

                if len(geoms) == 0:
                    continue


                if len(geoms) > features_per_tile:
                    node = B3dmNode(geoms[0:features_per_tile])
                    tree.children.append(node)
                    B3dmActions.divide(tile, geoms[features_per_tile:len(geoms)], i * 2,
                           j * 2, max_tile_size / 2., features_per_tile, node)
                else:
                    node = B3dmNode(geoms)
                    tree.children.append(node)

        # Export b3dm & tileset
        tree.compute_bbox()
        nodes = tree.all_nodes()
        identity = np.identity(4).flatten('F')
        tile_folder = out_folder / "b3dm_tiles" # TODO b3dm_tiles should not be hard coded
        tile_folder.mkdir(parents=True, exist_ok=True) # TODO check if there is no collision
        for node in nodes:
            if len(node.features) != 0:
                bin_arrays = []
                for feature in node.features:
                    pos = feature.index
                    bin_arrays.append({
                        'position': positions[pos],
                        'normal': normals[pos],
                        'bbox': [[float(i) for i in j] for j in bboxes[pos]],
                    })
                    # if ids is not None:
                    #     gids.append(ids[pos])
                gltf = GlTF.from_binary_arrays(bin_arrays, identity)
                bt = None
                # if ids is not None:
                #     bt = BatchTable()
                #     bt.add_property_from_array("id", gids)
                b3dm = B3dm.from_glTF(gltf, bt).to_array()
                with (tile_folder / f"{node.id}.b3dm").open('wb') as f:
                    f.write(b3dm)

        b3dm_state.tree = tree

    @staticmethod
    def write_tileset(b3dm_state: B3dmState, b3dm_metadata: B3dmMetadata, out_folder: Path) -> Path:
        tileset_path = out_folder / "b3dm_tileset.json"

        if b3dm_state.tree is None:
            raise AttributeError()
        tileset = b3dm_state.tree.to_tileset(b3dm_metadata.transform)
        with tileset_path.open('w') as f:
            json.dump(tileset, f)

        return tileset_path

    @staticmethod
    def divide(extent, geometries, x_offset, y_offset, tile_size,
               features_per_tile, parent: B3dmNode):
        for i in range(0, 2):
            for j in range(0, 2):
                tile = tile_extent(extent, tile_size, i, j)

                geoms = []
                for g in geometries:
                    if tile.inside(g.box.center()):
                        geoms.append(g)
                if len(geoms) == 0:
                    continue

                if len(geoms) > features_per_tile:
                    node = B3dmNode(geoms[0:features_per_tile])
                    parent.children.append(node)
                    B3dmActions.divide(tile, geoms[features_per_tile:len(geoms)],
                           (x_offset + i) * 2, (y_offset + j) * 2,
                           tile_size / 2., features_per_tile, node)
                else:
                    node = B3dmNode(geoms)
                    parent.children.append(node)


class Feature:
    def __init__(self, index, box):
        self.index = index
        self.box = box


def tile_extent(extent, size, i, j):
    min_extent = [
        extent.min[0] + i * size,
        extent.min[1] + j * size]
    max_extent = [
        extent.min[0] + (i + 1) * size,
        extent.min[1] + (j + 1) * size]
    return BoundingBox(min_extent, max_extent)
