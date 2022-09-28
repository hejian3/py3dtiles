from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tile_content import TileContentHeader


class BatchTable:
    """
    Only the JSON header has been implemented for now. According to the batch
    table documentation, the binary body is useful for storing long arrays of
    data (better performances)
    """

    def __init__(self):
        self.header = {}

    def add_property_from_array(self, property_name, array):
        self.header[property_name] = array

    # returns batch table as binary
    def to_array(self):
        # return an empty array if there is no data in header
        # instead of returning an array with only "{}" as data
        if not self.header:
            return np.array([], dtype=np.uint8)

        # convert dict to json string
        bt_json = json.dumps(self.header, separators=(',', ':'))
        # header must be 8-byte aligned (refer to batch table documentation)
        if len(bt_json) % 8 != 0:
            bt_json += ' ' * (8 - len(bt_json) % 8)
        # returns an array of binaries representing the batch table
        return np.frombuffer(bt_json.encode(), dtype=np.uint8)

    @staticmethod
    def from_array(header: TileContentHeader, array: np.ndarray) -> BatchTable:
        if header.bt_bin_byte_length != 0:
            raise NotImplementedError("BatchTable support only data from JSON header")

        batch_table = BatchTable()

        if header.bt_json_byte_length == 0:
            return batch_table

        json_array_part = array[:header.bt_json_byte_length]

        raw_json = json.loads(json_array_part.tobytes().decode('utf-8'))
        batch_table.header = raw_json

        return batch_table
