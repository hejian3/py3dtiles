import json
import unittest

from py3dtiles import BatchTable


class Test_Batch(unittest.TestCase):

    @classmethod
    def build_bt_sample(cls):
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
        json_bt = json.loads(self.build_bt_sample().to_array().tobytes().decode('utf-8'))
        json_reference = json.loads('{"id":["unique id","another unique id"],\
                                    "displayName":["Building name","Another building name"],\
                                    "yearBuilt":[1999,2015],\
                                    "address":[{"street":"Main Street","houseNumber":"1"},\
                                    {"street":"Main Street","houseNumber":"2"}]}')
        if not json_bt.items() == json_reference.items():
            self.fail()
