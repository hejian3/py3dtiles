import copy
from pathlib import Path
from typing import List, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import BoundingVolumeMissingException
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content import Pnts
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import TileSet
from py3dtiles.utils import split_aabb

T = TypeVar("T", bound=np.floating)


def quadtree_split(aabb: npt.NDArray[T]) -> List[npt.NDArray[T]]:
    return [
        split_aabb(aabb, 0, True),
        split_aabb(aabb, 2, True),
        split_aabb(aabb, 4, True),
        split_aabb(aabb, 6, True),
    ]


def is_point_inside(
    point: npt.NDArray[np.floating], aabb: npt.NDArray[np.floating]
) -> np.bool_:
    return np.all(aabb[0] <= point) and np.all(point <= aabb[1])


def build_tileset_quadtree(
    aabb: npt.NDArray[np.floating],
    tilesets: List[TileSet],
    bounding_box_centers: List[npt.NDArray[np.floating]],
    inv_base_transform: npt.NDArray[np.floating],
) -> Optional[Tile]:
    insides = [
        (tileset, center)
        for tileset, center in zip(tilesets, bounding_box_centers)
        if is_point_inside(center, aabb)
    ]

    if not insides:
        return None

    quadtree_diag = np.linalg.norm(aabb[1][:2] - aabb[0][:2])
    tileset_insides, center_insides = zip(*insides)

    if len(tileset_insides) == 1 or quadtree_diag < 1:
        # apply transform to boundingVolume
        tileset = tileset_insides[0]
        bvb = copy.deepcopy(tileset.root_tile.bounding_volume)
        if tileset.root_tile.transform is not None:
            bvb.transform(tileset.root_tile.transform)

        tile = Tile(
            geometric_error=tileset.root_tile.geometric_error,
            transform=inv_base_transform.T.flatten(),
            bounding_volume=bvb,
        )
        tile.content_uri = tileset
        return tile
    else:
        children = []

        for quarter in quadtree_split(aabb):
            r = build_tileset_quadtree(
                quarter,
                tileset_insides,
                center_insides,
                inv_base_transform,
            )
            if r is not None:
                children.append(r)

        main_root_tile = tileset_insides[0].root_tile
        union_aabb = copy.deepcopy(main_root_tile.bounding_volume)
        if main_root_tile.transform is not None:
            union_aabb.transform(main_root_tile.transform)

        # take half points from our children
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)

        max_point_count = 50000
        point_count = 0
        for tileset in tileset_insides:
            root_tile_content = tileset.root_tile.get_or_fetch_content()
            point_count += root_tile_content.body.feature_table.header.points_length

        ratio = min(0.5, max_point_count / point_count)

        for tileset in tileset_insides:
            root_tile = tileset.root_tile
            root_tile_content = root_tile.get_or_fetch_content()
            local_transform = root_tile.transform.reshape((4, 4)).T @ inv_base_transform
            _xyz, _rgb = root_tile_content.body.get_points(local_transform)

            select = np.random.choice(_xyz.shape[0], int(_xyz.shape[0] * ratio))
            xyz = np.concatenate((xyz, _xyz[select]))
            if _rgb is not None:
                rgb = np.concatenate((rgb, _rgb[select]))

            ab = copy.deepcopy(root_tile.bounding_volume)
            if root_tile.transform is not None:
                ab.transform(root_tile.transform)
            union_aabb.add(ab)

        pnts = Pnts.from_points(
            np.concatenate((xyz.view(np.uint8).ravel(), rgb.ravel())),
            rgb.shape[0] > 0,
            False,  # TODO: Handle classification in the merging process
        )

        union_aabb.transform(inv_base_transform.T.flatten())

        tile = Tile(
            refine_mode="REPLACE",
            bounding_volume=union_aabb,
            geometric_error=max(
                [tileset.root_tile.geometric_error for tileset in tileset_insides]
            ),
        )
        if pnts is not None:
            tile.tile_content = pnts

        for child in children:
            tile.add_child(child)

        return tile


def merge(tilesets: List[TileSet]) -> TileSet:
    global_bounding_volume = BoundingVolumeBox()
    bounding_box_centers = []

    for tileset in tilesets:
        # apply transformation
        if tileset.root_tile.bounding_volume is None:
            raise BoundingVolumeMissingException(
                "The root tile should have a bounding volume."
            )

        bounding_box = copy.deepcopy(tileset.root_tile.bounding_volume)
        if tileset.root_tile.transform is not None:
            bounding_box.transform(tileset.root_tile.transform)

        global_bounding_volume.add(bounding_box)

        bounding_box_centers.append(bounding_box.get_center())

    corners = global_bounding_volume.get_corners()
    aabb = np.array((corners[0], corners[-1]))

    base_transform = tilesets[0].root_tile.transform.reshape((4, 4))
    inv_base_transform = np.linalg.inv(base_transform)

    # build hierarchical structure
    result = build_tileset_quadtree(
        aabb, tilesets, bounding_box_centers, inv_base_transform
    )

    if result is None:
        raise RuntimeError("Result shouldn't be None")

    result.transform = base_transform.T.flatten()
    tileset = TileSet(geometric_error=float(np.linalg.norm((aabb[1] - aabb[0])[0:3])))
    tileset.root_tile = result

    return tileset


def merge_from_files(
    tileset_paths: List[Path], output_tileset_path: Path, overwrite: bool = True
) -> None:
    if output_tileset_path.exists():
        if overwrite:
            TileSet.from_file(output_tileset_path).delete_on_disk(
                output_tileset_path, delete_tile_content_tileset=False
            )
        else:
            raise FileExistsError(
                f"Destination tileset {output_tileset_path} already exists."
            )

    tilesets = []
    for path in tileset_paths:
        tilesets.append(TileSet.from_file(path))

    tileset = merge(tilesets)

    tileset.root_uri = output_tileset_path
    content = tileset.root_tile.get_or_fetch_content(tileset.root_uri)
    if content is not None:
        if isinstance(content, TileSet):
            raise RuntimeError(
                "The root tile content of the merged tileset shouldn't be a tileset"
            )
        content.save_as(output_tileset_path.parent / "r.pnts")

    tileset.write_as_json(output_tileset_path)


def init_parser(subparser):
    parser = subparser.add_parser(
        "merge", help="Merge several pointcloud tilesets in 1 tileset"
    )
    parser.add_argument("tilesets", nargs="+", help="All tileset paths to merge")
    parser.add_argument(
        "--output-tileset", required=True, help="The path to the output tileset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output folder if it already exists.",
    )

    return parser


def main(args):
    return merge_from_files(
        [Path(file) for file in args.tilesets], Path(args.output_tileset)
    )
