import json

import numpy as np


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
        # convert dict to json string
        bt_json = json.dumps(self.header, separators=(',', ':'))
        # header must be 8-byte aligned (refer to batch table documentation)
        if len(bt_json) % 8 != 0:
            bt_json += ' ' * (8 - len(bt_json) % 8)
        # returns an array of binaries representing the batch table
        return np.frombuffer(bt_json.encode(), dtype=np.uint8)
