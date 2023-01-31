import json
from pathlib import Path
from typing import List, Union

import numpy as np

from py3dtiles.tilers.pnts.pnts_writer import points_to_pnts
from py3dtiles.tileset.feature_table import SemanticPoint
from py3dtiles.tileset.tile_content import TileContent
from py3dtiles.tileset.utils import TileContentReader
from py3dtiles.typing import TileDictType
from py3dtiles.utils import split_aabb


def _get_root_tile(tileset: dict, root_tile_path: Path) -> TileContent:
    pnts_path = root_tile_path.parent / tileset['root']['content']['uri']
    return TileContentReader.read_file(pnts_path)


def _get_root_transform(tileset: dict) -> np.ndarray:
    transform = np.identity(4)
    if 'transform' in tileset:
        transform = np.array(tileset['transform']).reshape(4, 4).transpose()

    if 'transform' in tileset['root']:
        transform = np.dot(
            transform,
            np.array(tileset['root']['transform']).reshape(4, 4).transpose())

    return transform


def _get_tile_points(tile, tile_transform, out_transform):
    fth = tile.body.feature_table.header

    xyz = tile.body.feature_table.body.positions_arr.view(
        np.float32).reshape((fth.points_length, 3))
    if fth.colors == SemanticPoint.RGB:
        rgb = tile.body.feature_table.body.colors_arr.reshape(
            (fth.points_length, 3)).astype(np.uint8)
    else:
        rgb = None

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    w = np.ones(x.shape[0])

    transform = np.dot(out_transform, tile_transform)

    xyzw = np.dot(np.vstack((x, y, z, w)).transpose(), transform.T)

    return xyzw[:, 0:3].astype(np.float32), rgb


def init(tilset_paths: List[Path]) -> dict:
    aabb = None
    total_point_count = 0
    tilesets = []
    transforms = []

    idx = 0
    for tileset_path in tilset_paths:
        with tileset_path.open() as f:
            tileset = json.load(f)

            tile = _get_root_tile(tileset, tileset_path)
            fth = tile.body.feature_table.header

            # apply transformation
            transform = _get_root_transform(tileset)
            bbox = _aabb_from_3dtiles_bounding_volume(
                tileset['root']['boundingVolume'],
                transform)

            if aabb is None:
                aabb = bbox
            else:
                aabb[0] = np.minimum(aabb[0], bbox[0])
                aabb[1] = np.maximum(aabb[1], bbox[1])

            total_point_count += fth.points_length

            tileset['id'] = idx
            tileset['filename'] = str(tileset_path)
            tileset['center'] = ((bbox[0] + bbox[1]) * 0.5)
            tilesets += [tileset]

            transforms += [transform]

            idx += 1

    return {
        'tilesets': tilesets,
        'aabb': aabb,
        'point_count': total_point_count,
        'transforms': transforms
    }


def quadtree_split(aabb):
    return [
        split_aabb(aabb, 0, True),
        split_aabb(aabb, 2, True),
        split_aabb(aabb, 4, True),
        split_aabb(aabb, 6, True),
    ]


def is_tileset_inside(tileset, aabb):
    return np.all(aabb[0] <= tileset['center']) and np.all(tileset['center'] <= aabb[1])


def _3dtiles_bounding_box_from_aabb(aabb, transform=None):
    if transform is not None:
        aabb = np.dot(aabb, transform.T)
    ab_min = aabb[0]
    ab_max = aabb[1]
    center = (ab_min + ab_max) * 0.5
    half_size = (ab_max - ab_min) * 0.5

    return {
        'box': [
            center[0], center[1], center[2],
            half_size[0], 0, 0,
            0, half_size[1], 0,
            0, 0, half_size[2]
        ]
    }


def _aabb_from_3dtiles_bounding_volume(volume, transform=None):
    center = np.array(volume['box'][0:3])
    h_x_axis = np.array(volume['box'][3:6])
    h_y_axis = np.array(volume['box'][6:9])
    h_z_axis = np.array(volume['box'][9:12])

    amin = (center - h_x_axis - h_y_axis - h_z_axis)
    amax = (center + h_x_axis + h_y_axis + h_z_axis)
    amin.resize((4,))
    amax.resize((4,))
    amin[3] = 1
    amax[3] = 1

    aabb = np.array([amin, amax])

    if transform is not None:
        aabb = np.dot(aabb, transform.T)

    return aabb


def build_tileset_quadtree(out_folder: Path, aabb, tilesets, base_transform, inv_base_transform, name) -> Union[TileDictType, None]:
    insides = [tileset for tileset in tilesets if is_tileset_inside(tileset, aabb)]

    quadtree_diag = np.linalg.norm(aabb[1][:2] - aabb[0][:2])

    if not insides:
        return None
    elif len(insides) == 1 or quadtree_diag < 1:
        # apply transform to boundingVolume
        box = _aabb_from_3dtiles_bounding_volume(
            insides[0]['root']['boundingVolume'],
            _get_root_transform(insides[0]))

        return {
            'transform': inv_base_transform.T.reshape(16).tolist(),
            'geometricError': insides[0]['root']['geometricError'],
            'boundingVolume': _3dtiles_bounding_box_from_aabb(box),
            'content': {
                'uri': str(Path(insides[0]['filename']).relative_to(out_folder))
            }
        }
    else:
        children: List[TileDictType] = []

        sub = 0
        for quarter in quadtree_split(aabb):
            r = build_tileset_quadtree(
                out_folder,
                quarter,
                insides,
                base_transform,
                inv_base_transform,
                name + str(sub)
            )
            sub += 1
            if r is not None:
                children.append(r)

        union_aabb = _aabb_from_3dtiles_bounding_volume(
            insides[0]['root']['boundingVolume'],
            _get_root_transform(insides[0]))
        # take half points from our children
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)

        max_point_count = 50000
        point_count = 0
        for tileset in insides:
            root_tile = _get_root_tile(tileset, Path(tileset['filename']))
            point_count += root_tile.body.feature_table.header.points_length

        ratio = min(0.5, max_point_count / point_count)

        for tileset in insides:
            root_tile = _get_root_tile(tileset, Path(tileset['filename']))
            _xyz, _rgb = _get_tile_points(root_tile, _get_root_transform(tileset), inv_base_transform)
            select = np.random.choice(_xyz.shape[0], int(_xyz.shape[0] * ratio))
            xyz = np.concatenate((xyz, _xyz[select]))
            if _rgb is not None:
                rgb = np.concatenate((rgb, _rgb[select]))

            ab = _aabb_from_3dtiles_bounding_volume(
                tileset['root']['boundingVolume'],
                _get_root_transform(tileset))
            union_aabb[0] = np.minimum(union_aabb[0], ab[0])
            union_aabb[1] = np.maximum(union_aabb[1], ab[1])

        _, pnts_path = points_to_pnts(
            name.encode('ascii'),
            np.concatenate((xyz.view(np.uint8).ravel(), rgb.ravel())),
            out_folder,
            rgb.shape[0] > 0)

        result: TileDictType = {
            'children': children,
            'content': {'uri': str(pnts_path.relative_to(out_folder)) if pnts_path else ""},
            'geometricError': max([t['root']['geometricError'] for t in insides]) / ratio,
            'boundingVolume': _3dtiles_bounding_box_from_aabb(union_aabb, inv_base_transform)
        }

        return result


def extract_content_uris(tileset):
    contents = []
    for key in tileset:
        if key == 'content':
            contents.append(Path(tileset[key]['uri']))
        elif key == 'children':
            for child in tileset['children']:
                contents += extract_content_uris(child)
        elif key == 'root':
            contents += extract_content_uris(tileset['root'])

    return contents


def remove_tileset(tilset_path: Path) -> None:
    with tilset_path.open() as f:
        tileset = json.load(f)

    contents = [
        tilset_path.parent / content
        for content in extract_content_uris(tileset)
    ]

    for content in contents:
        if content.suffix == '.pnts':
            content.unlink()
        elif content.suffix != '.json':
            raise ValueError(f'unknown extension {content.suffix}')

    tilset_path.unlink()


def merge(folder: Union[str, Path], overwrite: bool = False, verbose: int = 0) -> None:
    folder = Path(folder)
    merger_tileset_path = folder / 'tileset.json'
    if merger_tileset_path.exists():
        if overwrite:
            remove_tileset(merger_tileset_path)
        else:
            raise FileExistsError(f'Destination tileset {merger_tileset_path} already exists.')

    tilesets = [
        tileset for tileset in folder.glob('**/tileset.json')
    ]

    if verbose >= 1:
        print(f'Found {len(tilesets)} tilesets to merge')
    if verbose >= 2:
        print(f'Tilesets: {tilesets}')

    infos = init(tilesets)

    aabb = infos['aabb']

    base_transform = infos['transforms'][0]

    inv_base_transform = np.linalg.inv(base_transform)
    print('------------------------')
    # build hierarchical structure
    result = build_tileset_quadtree(folder, aabb, infos['tilesets'], base_transform, inv_base_transform, '')

    if result is None:
        raise ValueError("result is None")  # todo better message

    result['transform'] = base_transform.T.reshape(16).tolist()
    tileset = {
        'asset': {
            'version': '1.0'
        },
        'refine': 'REPLACE',
        'geometricError': np.linalg.norm((aabb[1] - aabb[0])[0:3]),
        'root': result
    }

    output_tileset_path = folder / "tileset.json"
    with output_tileset_path.open('w') as f:
        json.dump(tileset, f)


def init_parser(subparser):
    parser = subparser.add_parser('merge', help='Merge several pointcloud tilesets in 1 tileset')
    parser.add_argument(
        'folder',
        help='Folder that contains tileset folders inside (the merged tileset will be inside folder)'
    )
    parser.add_argument(
        '--overwrite',
        action="store_true",
        help='Overwrite the output folder if it already exists.'
    )

    return parser


def main(args):
    return merge(args.folder, args.overwrite, args.verbose)
