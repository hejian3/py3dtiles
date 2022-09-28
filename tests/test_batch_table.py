import json
import unittest

import numpy as np

from py3dtiles.tileset.batch_table import BatchTable
from py3dtiles.tileset.content import PntsHeader


class Test_Batch(unittest.TestCase):

    @staticmethod
    def build_bt_sample():
        """
        Programmatically define the reference sample encountered in the
        Bath Table specification cf
        https://github.com/AnalyticalGraphicsInc/3d-tiles/blob/master/specification/TileFormats/BatchTable/README.md#json-header
        :return: the sample as BatchTable object.
        """
        bt = BatchTable()

        bt.add_property_from_array("id", ["unique id", "another unique id"])
        bt.add_property_from_array("displayName", ["Building name", "Another building name"])
        bt.add_property_from_array("yearBuilt", [1999, 2015])
        bt.add_property_from_array("address", [{"street": "Main Street", "houseNumber": "1"},
                                               {"street": "Main Street", "houseNumber": "2"}])
        return bt

    def test_json_encoding(self):
        bt_dict = json.loads(Test_Batch.build_bt_sample().to_array().tobytes().decode('utf-8'))
        bt_dict_reference = {
            "id": ["unique id", "another unique id"],
            "displayName": ["Building name", "Another building name"],
            "yearBuilt": [1999, 2015],
            "address": [{"street": "Main Street", "houseNumber": "1"}, {"street": "Main Street", "houseNumber": "2"}]
        }

        self.assertDictEqual(bt_dict, bt_dict_reference)

    def test_export_import_empty_batch_table(self):
        header = PntsHeader()
        batch_table = BatchTable.from_array(header, np.array(()))
        self.assertDictEqual(batch_table.header, {})

        batch_table_array = batch_table.to_array()
        self.assertEqual(batch_table_array.size, 0)
