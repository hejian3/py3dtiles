import json
import unittest
import numpy as np

from py3dtiles.tileset.batch_table import BatchTable, BatchTableHeader, BatchTableBody


class TestBatchTableHeader(unittest.TestCase):

    def test_header_as_array_of_values(self):
        input_data = {
            "id": ["unique id", "another unique id"],
            "displayName": ["Building name", "Another building name"],
            "yearBuilt": [1999, 2015],
            "address": [{"street": "Main Street", "houseNumber": "1"}, {"street": "Main Street", "houseNumber": "2"}]
        }
        bth = BatchTableHeader(input_data)

        if not json.loads(bth.to_array().tobytes().decode('utf-8')).items() == input_data.items():
            self.fail()


class TestBatchTableBody(unittest.TestCase):

    def test_empty_body(self):
        btb = BatchTableBody({})
        self.assertTrue(np.array_equal(btb.to_array(), np.array([], np.uint8)))

    def test_non_empty_body(self):
        btb = BatchTableBody({'property_1': {'data': np.array([1, 2, 3], np.ubyte),
                                             'componentType': 'UNSIGNED_BYTE',
                                             'byteOffset': 0},
                              'property_2': {'data': np.array([4, 5, 6], np.ubyte),
                                             'componentType': 'UNSIGNED_BYTE',
                                             'byteOffset': 3},
                              'property_3': {'data': np.array([7, 8, 9], np.ubyte),
                                             'componentType': 'UNSIGNED_BYTE',
                                             'byteOffset': 6}})
        btb_array = btb.to_array()
        self.assertEqual(btb_array.nbytes, 9)
        self.assertTrue(np.array_equal(btb.to_array(), np.arange(1, 10, dtype=np.uint8)))

    def test_wrong_offset(self):
        btb = BatchTableBody({'property_1': {'data': np.array([1, 2, 3], np.ubyte),
                                             'componentType': 'UNSIGNED_BYTE',
                                             'byteOffset': 0},
                              'property_2': {'data': np.array([4, 5, 6], np.ubyte),
                                             'componentType': 'UNSIGNED_BYTE',
                                             'byteOffset': 2},
                              'property_3': {'data': np.array([7, 8, 9], np.ubyte),
                                             'componentType': 'UNSIGNED_BYTE',
                                             'byteOffset': 6}})
        with self.assertRaises(AssertionError):
            btb_array = btb.to_array()

class TestBatchTable(unittest.TestCase):

    def test_add_property_from_array(self):
        bt = BatchTable()
        bt.add_property_from_array('property_1', [1, 2, 3])
        self.assertTrue(set(bt.header.data.keys()) == {'property_1'})
        self.assertEqual(bt.header.data['property_1'], [1, 2, 3])
        self.assertEqual(bt.body.data, {})

    def test_add_binary_property_from_array(self):
        bt = BatchTable()
        bt.add_binary_property_from_array('property_1', np.array([1, 2, 3], np.uint8),
                                          'UNSIGNED_BYTE',
                                          'SCALAR')
        self.assertTrue(set(bt.header.data.keys()) == {'property_1'})
        self.assertTrue(set(bt.body.data.keys()) == {'property_1'})
        self.assertEqual(bt.header.data['property_1'],
                         {"byteOffset": 0,
                          "componentType": 'UNSIGNED_BYTE',
                          "type": 'SCALAR'})
        self.assertTrue(np.array_equal(bt.body.data['property_1']['data'], np.array([1, 2, 3], np.uint8)))
        self.assertEqual(bt.body.data['property_1']['componentType'], 'UNSIGNED_BYTE')
        self.assertEqual(bt.body.data['property_1']['byteOffset'], 0)

        bt.add_binary_property_from_array('property_2', np.array([4, 5, 6], np.uint8),
                                          'UNSIGNED_BYTE',
                                          'SCALAR')
        self.assertTrue(set(bt.header.data.keys()) == {'property_1', 'property_2'})
        self.assertTrue(set(bt.body.data.keys()) == {'property_1', 'property_2'})
        self.assertEqual(bt.header.data['property_2'],
                         {"byteOffset": 3,
                          "componentType": 'UNSIGNED_BYTE',
                          "type": 'SCALAR'})
        self.assertTrue(np.array_equal(bt.body.data['property_2']['data'], np.array([4, 5, 6], np.uint8)))
        self.assertEqual(bt.body.data['property_2']['componentType'], 'UNSIGNED_BYTE')
        self.assertEqual(bt.body.data['property_2']['byteOffset'], 3)

    def test_to_array_with_empty_body(self):
        bt = BatchTable()
        bt.add_property_from_array('property_1', [1, 2, 3])
        self.assertTrue(np.array_equal(bt.to_array(), bt.header.to_array()))

    def test_to_array_with_non_empty_body(self):
        bt = BatchTable()
        bt.add_binary_property_from_array('property_1', np.array([1, 2, 3], np.uint8),
                                          'UNSIGNED_BYTE',
                                          'SCALAR')
        bt.add_binary_property_from_array('property_2', np.array([4, 5, 6], np.uint8),
                                          'UNSIGNED_BYTE',
                                          'SCALAR')
        self.assertTrue(np.array_equal(bt.to_array(), np.concatenate((bt.header.to_array(), bt.body.to_array()))))
