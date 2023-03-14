from pathlib import Path

from py3dtiles.tileset.content import B3dm, Pnts
from py3dtiles.tileset.tile_content_reader import read_file


def print_pnts_info(tile: Pnts) -> None:
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


def print_b3dm_info(tile: B3dm) -> None:
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
        gltf_header = tile.body.gltf.header
        print("")
        print("glTF Header")
        print("-----------")
        print(gltf_header)
    else:
        print("Tile with no body")


def main(args):
    try:
        tile_content = read_file(args.file)
    except ValueError as e:
        print(f"Error when reading the file {args.file}")
        raise e

    if isinstance(tile_content, Pnts):
        print_pnts_info(tile_content)
    elif isinstance(tile_content, B3dm):
        print_b3dm_info(tile_content)
    else:
        with args.file.open("b") as f:
            magic = f.read(4)
        raise RuntimeError(f"Unsupported format {magic!r}")


def init_parser(subparser):
    # arg parse
    parser = subparser.add_parser(
        "info", help="Extract information from a 3DTiles file"
    )

    parser.add_argument("file", type=Path)

    return parser
