import json
import unittest

import numpy as np
import numpy.typing as npt

from py3dtiles.tileset.batch_table import (
    BatchTable,
    BatchTableBody,
    BatchTableHeader,
    BatchTableHeaderDataType,
    ComponentNumpyType,
)
from py3dtiles.tileset.content import PntsHeader


class TestBatchTableHeader(unittest.TestCase):
    def test_header_as_array_of_values(self) -> None:
        input_data: BatchTableHeaderDataType = {
            "id": ["unique id", "another unique id"],
            "displayName": ["Building name", "Another building name"],
            "yearBuilt": [1999, 2015],
            "address": [
                {"street": "Main Street", "houseNumber": "1"},
                {"street": "Main Street", "houseNumber": "2"},
            ],
        }
        bth = BatchTableHeader(input_data)

        if (
            json.loads(bth.to_array().tobytes().decode("utf-8")).items()
            != input_data.items()
        ):
            self.fail()


class TestBatchTableBody(unittest.TestCase):
    def test_empty_body(self) -> None:
        btb = BatchTableBody()
        np.testing.assert_equal(btb.to_array(), np.array([], np.uint8))

    def test_non_empty_body(self) -> None:
        btb = BatchTableBody(
            [
                np.array([1, 2, 3], np.uint8),
                np.array([4, 5, 6], np.uint8),
                np.array([7, 8, 9], np.uint8),
            ]
        )
        btb_array = btb.to_array()
        self.assertEqual(btb_array.nbytes, 16)
        np.testing.assert_equal(
            btb.to_array(),
            np.concatenate(
                (np.arange(1, 10, dtype=np.uint8), np.array([32] * 7, dtype=np.uint8))
            ),
        )


class TestBatchTable(unittest.TestCase):
    @staticmethod
    def get_batch_table_header_dict_only_binary() -> BatchTableHeaderDataType:
        return {
            "binProperty1": {
                "byteOffset": 0,
                "componentType": "UNSIGNED_BYTE",
                "type": "SCALAR",
            },
            "binProperty2": {
                "byteOffset": 4 * 1 * 1,
                "componentType": "INT",
                "type": "VEC2",
            },
        }

    @staticmethod
    def get_batch_table_header_dict_only_json() -> BatchTableHeaderDataType:
        return {
            "property1": [1, 2, 3, 4],
            "property2": [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]],
        }

    def test_add_property_as_json(self) -> None:
        bt = BatchTable()
        bt.add_property_as_json("property_1", [1, 2, 3])
        self.assertTrue(set(bt.header.data.keys()) == {"property_1"})
        self.assertEqual(bt.header.data["property_1"], [1, 2, 3])
        self.assertEqual(bt.body.data, [])

    def test_add_property_as_binary(self) -> None:
        bt = BatchTable()

        bt.add_property_as_binary(
            "property_1", np.array([1, 2, 3], np.uint8), "UNSIGNED_BYTE", "SCALAR"
        )
        self.assertTrue(set(bt.header.data.keys()) == {"property_1"})
        self.assertEqual(
            bt.header.data["property_1"],
            {"byteOffset": 0, "componentType": "UNSIGNED_BYTE", "type": "SCALAR"},
        )
        self.assertEqual(len(bt.body.data), 1)
        np.testing.assert_equal(bt.body.data[0], np.array([1, 2, 3], np.uint8))

        bt.add_property_as_binary(
            "property_2", np.array([4, 5, 6], np.float32), "FLOAT", "SCALAR"
        )
        self.assertTrue(set(bt.header.data.keys()) == {"property_1", "property_2"})
        self.assertEqual(
            bt.header.data["property_2"],
            {"byteOffset": 3, "componentType": "FLOAT", "type": "SCALAR"},
        )
        self.assertEqual(len(bt.body.data), 2)
        np.testing.assert_equal(bt.body.data[1], np.array([4, 5, 6], np.float32))

    def test_to_array_with_empty_body(self) -> None:
        bt = BatchTable()
        bt.add_property_as_json("property_1", [1, 2, 3])
        self.assertTrue(np.array_equal(bt.to_array(), bt.header.to_array()))

    def test_to_array_with_non_empty_body(self) -> None:
        bt = BatchTable()
        bt.add_property_as_binary(
            "property_1", np.array([1, 2, 3], np.uint8), "UNSIGNED_BYTE", "SCALAR"
        )
        bt.add_property_as_binary(
            "property_2", np.array([4, 5, 6], np.uint8), "UNSIGNED_BYTE", "SCALAR"
        )
        self.assertTrue(
            np.array_equal(
                bt.to_array(),
                np.concatenate((bt.header.to_array(), bt.body.to_array())),
            )
        )

    def test_from_array_only_json(self) -> None:
        # build a minimal batch table
        batch_table_dict = self.get_batch_table_header_dict_only_json()
        batch_table_header = BatchTableHeader(batch_table_dict)
        batch_table_json = batch_table_header.to_array()

        # and tile content
        pnts_header = PntsHeader()
        pnts_header.bt_json_byte_length = len(batch_table_json)

        # import the array
        batch_table = BatchTable.from_array(pnts_header, batch_table_json)

        self.assertListEqual(batch_table.body.data, [])
        self.assertDictEqual(batch_table.header.data, batch_table_dict)

    def test_from_array_only_binary(self) -> None:
        # build a minimal batch table
        # json part
        batch_table_header_dict = self.get_batch_table_header_dict_only_binary()
        batch_table_header = BatchTableHeader(batch_table_header_dict)
        batch_table_json = batch_table_header.to_array()

        # binary part
        batch_table_body_binary: list[npt.NDArray[ComponentNumpyType]] = [
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.int32),
        ]
        batch_table_body = BatchTableBody(batch_table_body_binary)

        # merge the 2 parts
        batch_table_array = np.concatenate(
            (batch_table_json, batch_table_body.to_array())
        )

        # and tile content
        pnts_header = PntsHeader()
        pnts_header.bt_json_byte_length = len(batch_table_json)

        batch_table = BatchTable.from_array(pnts_header, batch_table_array, 4)

        # import the array
        self.assertDictEqual(batch_table.header.data, batch_table_header_dict)

        self.assertEqual(len(batch_table.body.data), 2)
        np.testing.assert_equal(batch_table_body_binary[0], batch_table.body.data[0])
        np.testing.assert_equal(batch_table_body_binary[1], batch_table.body.data[1])

    def test_from_array_mix_of_json_and_binary(self) -> None:
        # build a minimal batch table
        # json part
        batch_table_header_dict = {
            **self.get_batch_table_header_dict_only_json(),
            **self.get_batch_table_header_dict_only_binary(),
        }
        batch_table_header = BatchTableHeader(batch_table_header_dict)

        # binary part
        batch_table_body_binary: list[npt.NDArray[ComponentNumpyType]] = [
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.int32),
        ]
        batch_table_body = BatchTableBody(batch_table_body_binary)
        batch_table_json = batch_table_header.to_array()

        batch_table_array = np.concatenate(
            (batch_table_json, batch_table_body.to_array())
        )

        # and tile content
        pnts_header = PntsHeader()
        pnts_header.bt_json_byte_length = len(batch_table_json)

        # import the array
        batch_table = BatchTable.from_array(pnts_header, batch_table_array, 4)

        self.assertDictEqual(batch_table.header.data, batch_table_header_dict)

        self.assertEqual(len(batch_table.body.data), 2)
        np.testing.assert_equal(batch_table_body_binary[0], batch_table.body.data[0])
        np.testing.assert_equal(batch_table_body_binary[1], batch_table.body.data[1])

    def test_from_array_with_wrong_offset(self) -> None:
        # build a minimal batch table
        # json part
        batch_table_header_dict = self.get_batch_table_header_dict_only_binary()
        batch_table_header = BatchTableHeader(batch_table_header_dict)
        batch_table_json = batch_table_header.to_array()

        # binary part
        batch_table_body_binary: list[npt.NDArray[ComponentNumpyType]] = [
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.int32),
        ]
        batch_table_body = BatchTableBody(batch_table_body_binary)

        # merge the 2 parts
        batch_table_array = np.concatenate(
            (batch_table_json, batch_table_body.to_array())
        )

        # and tile content
        pnts_header = PntsHeader()
        pnts_header.bt_json_byte_length = len(batch_table_json)

        # import the array (wrong batch_len)
        with self.assertRaisesRegex(
            ValueError, "The byte offset is 4 but the byte offset computed is 3"
        ):
            BatchTable.from_array(pnts_header, batch_table_array, 3)

        # import the array (wrong byteOffset)
        batch_table_header.data["binProperty2"]["byteOffset"] = 6  # type: ignore [call-overload]
        batch_table_json = batch_table_header.to_array()
        batch_table_array = np.concatenate(
            (batch_table_json, batch_table_body.to_array())
        )

        with self.assertRaisesRegex(
            ValueError, "The byte offset is 6 but the byte offset computed is 4"
        ):
            BatchTable.from_array(pnts_header, batch_table_array, 4)
