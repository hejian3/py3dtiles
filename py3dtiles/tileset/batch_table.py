from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt

from py3dtiles.tileset.tile_content import TileContentHeader

COMPONENT_TYPE_NUMPY_MAPPING = {
    "BYTE": np.int8,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.int16,
    "UNSIGNED_SHORT": np.uint16,
    "INT": np.int32,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
}

TYPE_LENGTH_MAPPING = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


class BatchTableHeader:

    def __init__(self, data=None):
        self.data = data or {}

    def to_array(self):
        if not self.data:
            return np.empty((0,), dtype=np.uint8)

        json_str = json.dumps(self.data, separators=(',', ':'))
        if len(json_str) % 8 != 0:
            json_str += ' ' * (8 - len(json_str) % 8)
        return np.frombuffer(json_str.encode('utf-8'), dtype=np.uint8)


class BatchTableBody:
    def __init__(self, data=None):
        self.data = data or []

    def to_array(self):
        if not self.data:
            return np.empty((0,), dtype=np.uint8)

        if self.nbytes % 8 != 0:
            padding = ' ' * (8 - self.nbytes % 8)
            padding = np.frombuffer(padding.encode('utf-8'), dtype=np.uint8)
            self.data.append(padding)

        return np.concatenate([data.view(np.ubyte) for data in self.data], dtype=np.uint8)

    @property
    def nbytes(self):
        return sum([data.nbytes for data in self.data])


class BatchTable:
    """
    Only the JSON header has been implemented for now. According to the batch
    table documentation, the binary body is useful for storing long arrays of
    data (better performances)
    """

    def __init__(self):
        self.header = BatchTableHeader()
        self.body = BatchTableBody()

    def add_property_as_json(self, property_name, array):
        self.header.data[property_name] = array

    def add_property_as_binary(self, property_name, array, component_type, property_type):
        if array.dtype != COMPONENT_TYPE_NUMPY_MAPPING[component_type]:
            raise RuntimeError("The dtype of array should be the same as component_type")

        self.header.data[property_name] = {
            "byteOffset": self.body.nbytes,
            "componentType": component_type,
            "type": property_type
        }

        transformed_array = array.reshape(-1)
        self.body.data.append(transformed_array)

    def get_binary_property(self, property_name_to_fetch):
        binary_property_index = 0
        # The order in self.header.data is the same as in self.body.data
        # We should filter properties added as json.
        for property_name, property_definition in self.header.data.items():
            if isinstance(property_definition, list): # If it is a list, it means that it is a json property
                continue
            elif property_name_to_fetch == property_name:
                return self.body.data[binary_property_index]
            else:
                binary_property_index += 1
        else:
            raise ValueError(f"The property {property_name_to_fetch} is not found")

    def to_array(self):
        batch_table_header_array = self.header.to_array()
        batch_table_body_array = self.body.to_array()

        return np.concatenate((batch_table_header_array, batch_table_body_array))

    @staticmethod
    def from_array(tile_header: TileContentHeader, array: npt.NDArray[np.ubyte], batch_len: int | None = None) -> BatchTable:
        batch_table = BatchTable()
        # separate batch table header
        batch_table_header_length = tile_header.bt_json_byte_length
        batch_table_body_array = array[batch_table_header_length:]
        batch_table_header_array = array[0:batch_table_header_length]

        jsond = json.loads(batch_table_header_array.tobytes().decode('utf-8') or '{}')
        batch_table.header.data = jsond

        previous_byte_offset = 0
        for property_definition in batch_table.header.data.values():
            if isinstance(property_definition, list):
                continue

            if batch_len is None:  # todo once feature table is supported in B3dm, remove this exception
                raise ValueError("batch_len shouldn't be None if there are binary property in the batch table array")

            if previous_byte_offset != property_definition["byteOffset"]:
                raise ValueError("Bad byteOffset")

            numpy_type = COMPONENT_TYPE_NUMPY_MAPPING[property_definition["componentType"]]
            end_byte_offset = property_definition["byteOffset"] + (
                np.dtype(numpy_type).itemsize
                * TYPE_LENGTH_MAPPING[property_definition["type"]]
                * batch_len
            )
            batch_table.body.data.append(
                batch_table_body_array[property_definition["byteOffset"]:end_byte_offset].view(numpy_type)
            )
            previous_byte_offset = end_byte_offset

        return batch_table
