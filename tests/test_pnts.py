from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from py3dtiles.tileset.content import Pnts, PntsHeader
from py3dtiles.tileset.feature_table import FeatureTableHeader, SemanticPoint
from py3dtiles.tileset.tile_content_reader import read_file


class TestTileContentReader(unittest.TestCase):
    def test_read(self):
        tile = read_file(Path("tests/fixtures/pointCloudRGB.pnts"))

        self.assertEqual(tile.header.version, 1.0)
        self.assertEqual(tile.header.tile_byte_length, 15176)
        self.assertEqual(tile.header.ft_json_byte_length, 148)
        self.assertEqual(tile.header.ft_bin_byte_length, 15000)
        self.assertEqual(tile.header.bt_json_byte_length, 0)
        self.assertEqual(tile.header.bt_bin_byte_length, 0)

        feature_table = tile.body.feature_table
        pt_color = feature_table.get_feature_color_at(0)
        if pt_color is None:
            raise RuntimeError("pt_color should not be None")
        assert_array_equal(
            np.array([44, 243, 209], dtype=np.uint8),
            pt_color,
        )


class TestTileBuilder(unittest.TestCase):
    def test_build_without_colors(self):
        tread = read_file(Path("tests/fixtures/pointCloudRGB.pnts"))
        feature_0_position = tread.body.feature_table.get_feature_position_at(0)

        # create features
        positions = []
        for i in range(tread.body.feature_table.header.points_length):
            feature_position = tread.body.feature_table.get_feature_position_at(i)
            positions.append(feature_position)
        position_array = np.array(positions).flatten()

        # create a tile
        feature_table_header = FeatureTableHeader.from_semantic(
            SemanticPoint.POSITION,
            None,
            None,
            len(positions),
        )
        t = Pnts.from_features(feature_table_header, position_array)

        # configure the tile
        rtc = (1215012.8828876738, -4736313.051199594, 4081605.22126042)
        t.body.feature_table.header.rtc = rtc

        # get an array
        tile_arr = t.to_array()
        t2 = Pnts.from_array(tile_arr)
        self.assertEqual(t2.header.version, 1.0)

        # Test the tile byte length
        self.assertEqual(t2.header.tile_byte_length, 12152)
        self.assertEqual(
            t2.header.tile_byte_length % 8, 0
        )  # tile bytes must be 8-byte aligned

        # Test the feature table byte lengths
        json_feature_table_end = PntsHeader.BYTE_LENGTH + t2.header.ft_json_byte_length
        self.assertEqual(json_feature_table_end % 8, 0)
        self.assertEqual(t2.header.ft_json_byte_length, 124)
        bin_feature_table_end = json_feature_table_end + t2.header.ft_bin_byte_length
        self.assertEqual(bin_feature_table_end % 8, 0)
        self.assertEqual(t2.header.ft_bin_byte_length, 12000)

        self.assertEqual(t2.header.bt_json_byte_length, 0)
        self.assertEqual(t2.header.bt_bin_byte_length, 0)

        feature_table = t.body.feature_table

        assert_array_equal(feature_0_position, feature_table.get_feature_position_at(0))

    def test_build(self):
        tread = read_file(Path("tests/fixtures/pointCloudRGB.pnts"))

        # create features
        positions = []
        colors = []
        for i in range(tread.body.feature_table.header.points_length):
            (
                feature_position,
                feature_color,
                _,
            ) = tread.body.feature_table.get_feature_at(i)
            positions.append(feature_position)
            colors.append(feature_color)
        position_array = np.array(positions).flatten()
        color_array = np.array(colors).flatten()

        # create a tile
        feature_table_header = FeatureTableHeader.from_semantic(
            SemanticPoint.POSITION, SemanticPoint.RGB, None, len(positions)
        )
        t = Pnts.from_features(feature_table_header, position_array, color_array)

        # configure the tile
        rtc = (1215012.8828876738, -4736313.051199594, 4081605.22126042)
        t.body.feature_table.header.rtc = rtc

        # get an array
        tile_arr = t.to_array()
        t2 = Pnts.from_array(tile_arr)
        self.assertEqual(t2.header.version, 1.0)

        # Test the tile byte length
        self.assertEqual(t2.header.tile_byte_length, 15176)
        self.assertEqual(
            t2.header.tile_byte_length % 8, 0
        )  # tile bytes must be 8-byte aligned

        # Test the feature table byte lengths
        json_feature_table_end = PntsHeader.BYTE_LENGTH + t2.header.ft_json_byte_length
        self.assertEqual(json_feature_table_end % 8, 0)
        self.assertEqual(t2.header.ft_json_byte_length, 148)
        bin_feature_table_end = json_feature_table_end + t2.header.ft_bin_byte_length
        self.assertEqual(bin_feature_table_end % 8, 0)
        self.assertEqual(t2.header.ft_bin_byte_length, 15000)

        self.assertEqual(t2.header.bt_json_byte_length, 0)
        self.assertEqual(t2.header.bt_bin_byte_length, 0)

        feature_table = t.body.feature_table
        pt_color = feature_table.get_feature_color_at(0)
        if pt_color is None:
            raise RuntimeError("pt_color should not be None")
        assert_array_equal(
            np.array([44, 243, 209], dtype=np.uint8),
            pt_color,
        )
