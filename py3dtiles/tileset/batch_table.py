from __future__ import annotations

import numpy as np
import json

COMPONENT_TYPE_NUMPY_MAPPING = {
    "BYTE": np.byte,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.short,
    "UNSIGNED_SHORT": np.ushort,
    "INT": np.int32,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.single,
    "DOUBLE": np.double
}


class BatchTableHeader:

    def __init__(self, d):
        self.data = d

    def to_array(self):
        json_str = json.dumps(self.data, separators=(',', ':'))
        if len(json_str) % 8 != 0:
            json_str += ' ' * (8 - len(json_str) % 8)
        return np.frombuffer(json_str.encode('utf-8'), dtype=np.uint8)


class BatchTableBody:

    def __init__(self, data):
        self.data = data

    def to_array(self):
        if not self.data:
            return np.array([], np.uint8)
        body_array = None
        for property, property_data in sorted(self.data.items(), key=lambda x: x[1]['byteOffset']):
            assert property_data['byteOffset'] == (body_array.nbytes if body_array is not None else 0), 'Mismatch between expected offset and array byte size'
            array = property_data['data'].view(COMPONENT_TYPE_NUMPY_MAPPING[property_data['componentType']])
            if body_array is not None:
                body_array = np.concatenate((body_array, array))
            else:
                body_array = array
        return body_array


class BatchTable:
    """
    Only the JSON header has been implemented for now. According to the batch
    table documentation, the binary body is useful for storing long arrays of
    data (better performances)
    """

    def __init__(self):
        self.header = BatchTableHeader({})
        self.body = BatchTableBody({})

    def add_property_from_array(self, property_name, array):
        self.header.data[property_name] = array

    def add_binary_property_from_array(self, property_name, array, component_type, property_type):
        self.header.data[property_name] = {
            "byteOffset": self.to_array().nbytes - self.header.to_array().nbytes,
            "componentType": component_type,
            "type": property_type
        }

        self.body.data[property_name] = {'data': array,
                                         'componentType': self.header.data[property_name]['componentType'],
                                         'byteOffset': self.header.data[property_name]['byteOffset']}

    def to_array(self):
        batch_table_header_array = self.header.to_array()
        if not self.body.data:
            return batch_table_header_array

        batch_table_body_array = self.body.to_array()
        return np.concatenate((batch_table_header_array, batch_table_body_array))

    @staticmethod
    def from_array(header: TileContentHeader, array: np.ndarray) -> BatchTable:
        batch_table = BatchTable()
        # separate batch table header
        batch_table_header_length = header.bt_json_byte_length
        batch_table_header_array = array[0:batch_table_header_length]
        jsond = json.loads(batch_table_header_array.tobytes().decode('utf-8') or '{}')
        batch_table.header = BatchTableHeader(jsond)
        if all([type(a) == dict for a in jsond.values()]):
            body_data = {}
            for key in jsond:
                body_data[key] = {'data': array[(batch_table_header_length + jsond[key]['byteOffset']):],
                                  'componentType': jsond[key]['componentType'],
                                  'byteOffset': jsond[key]['byteOffset']}
            bt.body = BatchTableBody(body_data)
        return batch_table
