from pathlib import Path

from py3dtiles.tileset.tile_content import TileContent
from py3dtiles.tileset.utils import TileContentReader


def print_pnts_info(tile: TileContent):
    if tile.header:
        th = tile.header
        print("Tile Header")
        print("-----------")
        print("Magic Value: ", th.magic_value)
        print("Version: ", th.version)
        print("Tile byte length: ", th.tile_byte_length)
        print("Feature table json byte length: ", th.ft_json_byte_length)
        print("Feature table bin byte length: ", th.ft_bin_byte_length)
    else:
        print("Tile with no header")

    if tile.body:
        fth = tile.body.feature_table.header
        print("")
        print("Feature Table Header")
        print("--------------------")
        print(fth.to_json())

        # first point data
        if fth.points_length > 0:
            print("")
            print("First point")
            print("-----------")
            f = tile.body.feature_table.feature(0)
            d = f.positions
            d.update(f.colors)
            print(d)
    else:
        print("Tile with no body")


def print_b3dm_info(tile: TileContent):
    if tile.header:
        th = tile.header
        print("Tile Header")
        print("-----------")
        print("Magic Value: ", th.magic_value)
        print("Version: ", th.version)
        print("Tile byte length: ", th.tile_byte_length)
        print("Feature table json byte length: ", th.ft_json_byte_length)
        print("Feature table bin byte length: ", th.ft_bin_byte_length)
        print("Batch table json byte length: ", th.bt_json_byte_length)
        print("Batch table bin byte length: ", th.bt_bin_byte_length)
    else:
        print("Tile with no header")

    if tile.body:
        gltfh = tile.body.glTF.header
        print("")
        print("glTF Header")
        print("-----------")
        print(gltfh)
    else:
        print("Tile with no body")


def main(args):
    try:
        tile = TileContentReader.read_file(args.file)
    except ValueError as e:
        print(f"Error when reading the file {args.file}")
        raise e

    magic = tile.header.magic_value

    if magic == b"pnts":
        print_pnts_info(tile)
    elif magic == b"b3dm":
        print_b3dm_info(tile)
    else:
        raise RuntimeError("Unsupported format " + str(magic))


def init_parser(subparser):
    # arg parse
    parser = subparser.add_parser('info', help='Extract information from a 3DTiles file')

    parser.add_argument('file', type=Path)

    return parser
