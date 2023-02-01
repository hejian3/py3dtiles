import unittest

import numpy as np
from numpy.testing import assert_array_equal

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tile_content import TileContent


class TestTile(unittest.TestCase):

    def test_constructor(self):
        tile = Tile()
        self.assertIsNone(tile.bounding_volume)
        self.assertEqual(tile.geometric_error, 500)
        self.assertEqual(tile._refine, "ADD")
        self.assertIsNone(tile._content)
        self.assertListEqual(tile._children, [])
        assert_array_equal(tile.transform, np.identity(4).reshape(-1))

        bounding_volume = BoundingVolumeBox()
        bounding_volume.set_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        tile = Tile(geometric_error=200, bounding_volume=bounding_volume)
        self.assertIs(tile.bounding_volume, bounding_volume)
        self.assertEqual(tile.geometric_error, 200)
        self.assertEqual(tile._refine, "ADD")
        self.assertIsNone(tile._content)
        self.assertListEqual(tile._children, [])
        assert_array_equal(tile.transform, np.identity(4).reshape(-1))

    def test_transform(self):
        tile = Tile()

        assert_array_equal(
            tile.transform,
            np.identity(4).reshape(-1)
        )

        tile.transform = np.array([
            1.0001, 0.0, 0.0, 0.0,
            0.0, 1.001, 0.0, 0.0,
            0.0, 0.0, 1.01, 0.0,
            0.0, 0.0, 0.0, 1.1
        ])

        assert_array_equal(
            tile.transform,
            np.array([1.0001, 0.0, 0.0, 0.0,
             0.0, 1.001, 0.0, 0.0,
             0.0, 0.0, 1.01, 0.0,
             0.0, 0.0, 0.0, 1.1])
        )

    def test_content(self):
        tile = Tile()

        self.assertIsNone(tile.get_content())

        tile_content = TileContent()
        tile.set_content(tile_content)
        self.assertIs(tile.get_content(), tile_content)

        new_tile_content = TileContent()
        tile.set_content(new_tile_content, force=False)
        self.assertIs(tile.get_content(), tile_content)

        new_tile_content = TileContent()
        tile.set_content(new_tile_content)
        self.assertIs(tile.get_content(), new_tile_content)

    def test_content_uri(self):
        pass

    def test_refine_mode(self):
        tile = Tile()

        self.assertEqual(tile.get_refine_mode(), "ADD")

        with self.assertRaises(ValueError):
            tile.set_refine_mode("replace")
        self.assertEqual(tile.get_refine_mode(), "ADD")

        tile.set_refine_mode("REPLACE")
        self.assertEqual(tile.get_refine_mode(), "REPLACE")

    def test_children(self):
        tile1 = Tile()

        self.assertFalse(tile1.has_children())
        self.assertListEqual(tile1.get_direct_children(), [])
        self.assertListEqual(tile1.get_children(), [])

        tile11 = Tile()
        tile1.add_child(tile11)
        self.assertTrue(tile1.has_children())
        self.assertListEqual(tile1.get_direct_children(), [tile11])
        self.assertListEqual(tile1.get_children(), [tile11])

        tile12 = Tile()
        tile1.add_child(tile12)
        self.assertTrue(tile1.has_children())
        self.assertListEqual(tile1.get_direct_children(), [tile11, tile12])
        self.assertListEqual(tile1.get_children(), [tile11, tile12])

        tile111 = Tile()
        tile11.add_child(tile111)

        self.assertTrue(tile1.has_children())
        self.assertTrue(tile11.has_children())
        self.assertFalse(tile111.has_children())
        self.assertFalse(tile12.has_children())

        self.assertListEqual(tile1.get_direct_children(), [tile11, tile12])
        self.assertListEqual(tile1.get_children(), [tile11, tile111, tile12])
        self.assertListEqual(tile11.get_children(), [tile111])

    def test_to_dict(self):
        tile = Tile()

        with self.assertRaises(AttributeError):
            tile.to_dict()

        bounding_volume = BoundingVolumeBox()
        bounding_volume.set_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        tile.bounding_volume = bounding_volume

        # most simple case
        self.assertDictEqual(
            tile.to_dict(),
            {
                "boundingVolume": {"box": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
                "geometricError": 500,
                "refine": "ADD",
            }
        )

        tile.geometric_error = 3.14159
        tile.to_dict()  # just test if no error

        child_tile = Tile()
        child_tile.geometric_error = 21
        child_tile.bounding_volume = bounding_volume
        tile.add_child(child_tile)

        # cannot test now
        # child.set_content()

        self.assertDictEqual(
            tile.to_dict(),
            {
                "boundingVolume": {"box": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
                "geometricError": 3.14159,
                "refine": "ADD",
                "children": [
                    {
                        "boundingVolume": {
                            "box": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
                        "geometricError": 21,
                        "refine": "ADD",
                    }
                ]
            }
        )


if __name__ == "__main__":
    unittest.main()
