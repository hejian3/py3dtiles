import unittest

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.extension import BaseExtension
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import TileSet


class Test_TileSet(unittest.TestCase):

    @classmethod
    def build_sample(cls):
        """
        Programmatically define a tileset sample encountered in the
        TileSet json header specification cf
        https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/specification#tileset-json
        :return: the sample as TileSet object.
        """
        tile_set = TileSet()
        bounding_volume = BoundingVolumeBox()
        bounding_volume.set_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        root_tile = Tile(geometric_error=3.14159, bounding_volume=bounding_volume)
        # Setting the mode to the default mode does not really change things.
        # The following line is thus just here ot test the "callability" of
        # set_refine_mode():
        root_tile.set_refine_mode('ADD')
        tile_set.root_tile = root_tile

        extension = BaseExtension('Test')
        tile_set.add_extension(extension)

        return tile_set

    def test_constructor(self):
        tile_set = TileSet()
        self.assertDictEqual(tile_set._asset, {"version": "1.0"})
        self.assertDictEqual(tile_set._extensions, {})
        self.assertEqual(tile_set.geometric_error, 500)
        self.assertIsNotNone(tile_set.root_tile)

    def test_to_dict(self):
        self.assertDictEqual(
            self.build_sample().to_dict(),
            {
                "root": {
                    "boundingVolume": {"box": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
                    "geometricError": 3.14159,
                    "refine": "ADD",
                },
                "extensions": {
                    'Test': {}},
                "geometricError": 500.0, "asset": {"version": "1.0"},
            }
        )


if __name__ == "__main__":
    unittest.main()
