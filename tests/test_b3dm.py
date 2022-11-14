from filecmp import cmp
import json
from pathlib import Path
import unittest

import numpy as np

from py3dtiles.tilers.b3dm.wkb_utils import TriangleSoup
from py3dtiles.tileset.content import B3dm, GlTF
from py3dtiles.tileset.utils import TileContentReader


class TestTileContentReader(unittest.TestCase):

    def test_read(self):
        tile = TileContentReader.read_file(Path('tests/fixtures/dragon_low.b3dm'))

        self.assertEqual(tile.header.version, 1.0)
        self.assertEqual(tile.header.tile_byte_length, 47246)
        self.assertEqual(tile.header.ft_json_byte_length, 20)
        self.assertEqual(tile.header.ft_bin_byte_length, 0)
        self.assertEqual(tile.header.bt_json_byte_length, 0)
        self.assertEqual(tile.header.bt_bin_byte_length, 0)

        with open('tests/fixtures/dragon_low_gltf_header.json') as f:
            gltf_header = json.loads(f.read())
        self.assertDictEqual(gltf_header, tile.body.glTF.header)

    def test_read_and_write(self):
        tile_content = TileContentReader.read_file(Path('tests/fixtures/buildings.b3dm'))

        self.assertEqual(tile_content.header.tile_byte_length, 6176)
        self.assertEqual(tile_content.header.ft_json_byte_length, 0)
        self.assertEqual(tile_content.header.ft_bin_byte_length, 0)
        self.assertEqual(tile_content.header.bt_json_byte_length, 64)
        self.assertEqual(tile_content.header.bt_bin_byte_length, 0)
        self.assertEqual(tile_content.body.batch_table.header, {"id": ["BATIMENT0000000240853073", "BATIMENT0000000240853157"]})
        self.assertEqual(len(tile_content.body.glTF.to_array()), 6084)
        self.assertEqual(tile_content.body.glTF.header['asset']['version'], '2.0')

        path_name = Path('tests/output_tests/buildings.b3dm')
        path_name.parent.mkdir(parents=True, exist_ok=True)

        tile_content.save_as(path_name)
        self.assertTrue(cmp('tests/fixtures/buildings.b3dm', path_name))


class TestTileContentBuilder(unittest.TestCase):

    def test_build(self):
        with open('tests/fixtures/building.wkb', 'rb') as f:
            wkb = f.read()
        ts = TriangleSoup.from_wkb_multipolygon(wkb)
        positions = ts.get_position_array()
        normals = ts.get_normal_array()
        box = [[-8.74748499994166, -7.35523200035095, -2.05385796777344],
               [8.8036420000717, 7.29930999968201, 2.05386103222656]]
        arrays = [{
            'position': positions,
            'normal': normals,
            'bbox': box
        }]

        transform = np.array([
            [1, 0, 0, 1842015.125],
            [0, 1, 0, 5177109.25],
            [0, 0, 1, 247.87364196777344],
            [0, 0, 0, 1]], dtype=float)
        # translation : 1842015.125, 5177109.25, 247.87364196777344
        transform = transform.flatten('F')
        glTF = GlTF.from_binary_arrays(arrays, transform)
        t = B3dm.from_glTF(glTF)

        # get an array
        t.to_array()
        self.assertEqual(t.header.version, 1.0)
        self.assertEqual(t.header.tile_byte_length, 2952)
        self.assertEqual(t.header.ft_json_byte_length, 0)
        self.assertEqual(t.header.ft_bin_byte_length, 0)
        self.assertEqual(t.header.bt_json_byte_length, 0)
        self.assertEqual(t.header.bt_bin_byte_length, 0)

        # t.save_as("/tmp/py3dtiles_test_build_1.b3dm")


class TestTexturedTileBuilder(unittest.TestCase):

    def test_build(self):
        with open('tests/fixtures/square.wkb', 'rb') as f:
            wkb = f.read()
        with open('tests/fixtures/squareUV.wkb', 'rb') as f:
            wkbuv = f.read()
        ts = TriangleSoup.from_wkb_multipolygon(wkb, [wkbuv])
        positions = ts.get_position_array()
        normals = ts.get_normal_array()
        uvs = ts.get_data_array(0)
        box = [[0, 0, 0],
               [10, 10, 0]]
        arrays = [{
            'position': positions,
            'normal': normals,
            'uv': uvs,
            'bbox': box
        }]

        transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=float)
        transform = transform.flatten('F')
        glTF = GlTF.from_binary_arrays(arrays, transform, texture_uri='squaretexture.jpg')
        t = B3dm.from_glTF(glTF)

        # get an array
        t.to_array()
        self.assertEqual(t.header.version, 1.0)
        self.assertEqual(t.header.tile_byte_length, 1556)
        self.assertEqual(t.header.ft_json_byte_length, 0)
        self.assertEqual(t.header.ft_bin_byte_length, 0)
        self.assertEqual(t.header.bt_json_byte_length, 0)
        self.assertEqual(t.header.bt_bin_byte_length, 0)

        # t.save_as("/tmp/py3dtiles_test_build_1.b3dm")
