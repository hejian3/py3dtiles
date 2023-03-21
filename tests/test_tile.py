import copy
from pathlib import Path
import shutil

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from py3dtiles.convert import convert
from py3dtiles.exceptions import InvalidTilesetError, TilerException
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content import (
    Pnts,
    PntsBody,
    PntsHeader,
)
from py3dtiles.tileset.tile import Tile

DATA_DIRECTORY = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_dir():
    tmp_dir = Path("tmp/")
    tmp_dir.mkdir(exist_ok=True)
    convert(DATA_DIRECTORY / "simple.xyz", outfolder=tmp_dir, overwrite=True)
    yield tmp_dir
    print("cleanup")
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestTileContentManagement:
    def test_init_without_data(self, tmp_dir):
        tile = Tile()

        assert tile.tile_content is None
        assert tile.content_uri is None
        with pytest.raises(
            RuntimeError, match="Cannot load a tile without a content_uri"
        ):
            tile.get_or_fetch_content(tmp_dir)

    def test_with_tile_content(self, tmp_dir):
        tile = Tile()

        pnts = Pnts(PntsHeader(), PntsBody())
        tile.tile_content = pnts

        assert tile.get_or_fetch_content(None) == pnts
        assert tile.content_uri is None

    def test_with_path_content(self, tmp_dir):
        tile_path = Path("r.pnts")
        tile = Tile(content_uri=tile_path)

        assert tile.tile_content is None
        assert isinstance(tile.get_or_fetch_content(tmp_dir), Pnts)
        assert isinstance(tile.tile_content, Pnts)

    def test_write(self, tmp_dir):
        tile = Tile()
        pnts = Pnts(PntsHeader(), PntsBody())
        tile.tile_content = pnts

        with pytest.raises(
            TilerException, match="tile.content_uri is null, cannot write tile content"
        ):
            tile.write_content(tmp_dir)

        tile.content_uri = Path("rr.pnts")
        with pytest.raises(
            ValueError, match="No root_uri given and tile.content_uri is not absolute"
        ):
            tile.write_content(None)

        tile.write_content(tmp_dir)
        assert (tmp_dir / tile.content_uri).exists()

        tile.tile_content = None
        with pytest.raises(
            TilerException,
            match="The tile has no tile content. A tile content should be added in the tile.",
        ):
            tile.write_content(tmp_dir)


class TestTile:
    def test_constructor(self):
        tile = Tile()
        assert tile.bounding_volume is None
        assert tile.geometric_error == 500
        assert tile._refine == "ADD"
        assert tile.children == []
        assert_array_equal(tile.transform, np.identity(4).reshape(-1))

        bounding_volume = BoundingVolumeBox()
        bounding_volume.set_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        tile = Tile(geometric_error=200, bounding_volume=bounding_volume)
        assert tile.bounding_volume == bounding_volume
        assert tile.geometric_error == 200
        assert tile._refine == "ADD"
        assert tile.children == []
        assert_array_equal(tile.transform, np.identity(4).reshape(-1))

    def test_transform(self):
        tile = Tile()

        assert_array_equal(tile.transform, np.identity(4).reshape(-1))

        tile.transform = np.array(
            [
                1.0001,
                0.0,
                0.0,
                0.0,
                0.0,
                1.001,
                0.0,
                0.0,
                0.0,
                0.0,
                1.01,
                0.0,
                0.0,
                0.0,
                0.0,
                1.1,
            ]
        )

        assert_array_equal(
            tile.transform,
            np.array(
                [
                    1.0001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.1,
                ]
            ),
        )

    def test_refine_mode(self):
        tile = Tile()
        assert tile.get_refine_mode() == "ADD"

        with pytest.raises(InvalidTilesetError):
            tile.set_refine_mode("replace")

        tile.set_refine_mode("REPLACE")
        assert tile.get_refine_mode() == "REPLACE"

    def test_children(self):
        tile1 = Tile()

        assert tile1.children == []
        assert tile1.get_all_children() == []

        tile11 = Tile()
        tile1.add_child(tile11)
        assert tile1.children == [tile11]
        assert tile1.get_all_children() == [tile11]

        tile12 = Tile()
        tile1.add_child(tile12)
        assert tile1.children == [tile11, tile12]
        assert tile1.get_all_children() == [tile11, tile12]

        tile111 = Tile()
        tile11.add_child(tile111)

        assert tile1.children == [tile11, tile12]
        assert tile1.get_all_children() == [tile11, tile111, tile12]
        assert tile11.get_all_children() == [tile111]

    def test_to_dict(self):
        tile = Tile()

        with pytest.raises(InvalidTilesetError, match="Bounding volume is not set"):
            tile.to_dict()

        bounding_volume = BoundingVolumeBox()
        bounding_volume.set_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        tile.bounding_volume = bounding_volume

        # most simple case
        assert tile.to_dict() == {
            "boundingVolume": {
                "box": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                ]
            },
            "geometricError": 500,
            "refine": "ADD",
        }

        tile.geometric_error = 3.14159
        tile.to_dict()  # just test if no error

        child_tile = Tile()
        child_tile.geometric_error = 21
        child_tile.bounding_volume = copy.deepcopy(bounding_volume)
        tile.add_child(child_tile)

        # cannot test now
        # child.set_content()

        assert tile.to_dict() == {
            "boundingVolume": {
                "box": [
                    1.0,
                    2.0,
                    3.0,
                    21.0,
                    0.0,
                    0.0,
                    0.0,
                    24.0,
                    0.0,
                    0.0,
                    0.0,
                    27.0,
                ]
            },
            "geometricError": 3.14159,
            "refine": "ADD",
            "children": [
                {
                    "boundingVolume": {
                        "box": [
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            8.0,
                            9.0,
                            10.0,
                            11.0,
                            12.0,
                        ]
                    },
                    "geometricError": 21,
                    "refine": "ADD",
                }
            ],
        }
