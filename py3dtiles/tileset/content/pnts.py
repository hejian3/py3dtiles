from __future__ import annotations

import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from py3dtiles.tileset.batch_table import BatchTable
from py3dtiles.tileset.feature_table import Feature, FeatureTable
from py3dtiles.tileset.tile_content import TileContent, TileContentBody, TileContentHeader, TileContentType


class Pnts(TileContent):

    @staticmethod
    def from_features(pd_type: npt.DTypeLike, cd_type: npt.DTypeLike, features: list[Feature]) -> Pnts:
        """
        Creates a Pnts from features defined by pd_type and cd_type.
        """

        ft = FeatureTable.from_features(pd_type, cd_type, features)

        tb = PntsBody()
        tb.feature_table = ft

        th = PntsHeader()

        t = Pnts()
        t.body = tb
        t.header = th

        return t

    @staticmethod
    def from_array(array: npt.NDArray) -> Pnts:
        """
        Creates a Pnts from an array
        """

        # build tile header
        h_arr = array[0:PntsHeader.BYTE_LENGTH]
        h = PntsHeader.from_array(h_arr)

        if h.tile_byte_length != len(array):
            raise RuntimeError("Invalid byte length in header")

        # build tile body
        b_len = h.ft_json_byte_length + h.ft_bin_byte_length
        b_arr = array[PntsHeader.BYTE_LENGTH:PntsHeader.BYTE_LENGTH + b_len]
        b = PntsBody.from_array(h, b_arr)

        # build TileContent with header and body
        t = Pnts()
        t.header = h
        t.body = b

        return t


class PntsHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self) -> None:
        super().__init__()
        self.type = TileContentType.POINT_CLOUD
        self.magic_value = b"pnts"
        self.version = 1

    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Returns the header as a numpy array.
        """
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array([self.version,
                                self.tile_byte_length,
                                self.ft_json_byte_length,
                                self.ft_bin_byte_length,
                                self.bt_json_byte_length,
                                self.bt_bin_byte_length], dtype=np.uint32)

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

    def sync(self, body: PntsBody) -> None:
        """
        Synchronizes headers with the Pnts body.
        """

        # extract arrays
        feature_table_header_array = body.feature_table.header.to_array()
        feature_table_body_array = body.feature_table.body.to_array()
        batch_table_header_array = body.batch_table.to_array()  # for now, there is only json part in the batch table
        batch_table_body_array: list[Any] = []  # body.batch_table.body.to_array()

        # sync the tile header with feature table contents
        self.tile_byte_length = (
            len(feature_table_header_array) + len(feature_table_body_array)
            + len(batch_table_header_array) + len(batch_table_body_array)
            + PntsHeader.BYTE_LENGTH
        )
        self.ft_json_byte_length = len(feature_table_header_array)
        self.ft_bin_byte_length = len(feature_table_body_array)
        self.bt_json_byte_length = len(batch_table_header_array)
        self.bt_bin_byte_length = len(batch_table_body_array)

    @staticmethod
    def from_array(array: npt.NDArray) -> PntsHeader:
        """
        Create a PntsHeader from an array
        """

        h = PntsHeader()

        if len(array) != PntsHeader.BYTE_LENGTH:
            raise RuntimeError("Invalid header length")

        h.version = struct.unpack("i", array[4:8].tobytes())[0]
        h.tile_byte_length = struct.unpack("i", array[8:12].tobytes())[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16].tobytes())[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20].tobytes())[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24].tobytes())[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28].tobytes())[0]

        return h


class PntsBody(TileContentBody):
    def __init__(self) -> None:
        self.feature_table = FeatureTable()
        self.batch_table = BatchTable()

    def to_array(self) -> npt.NDArray:
        """
        Returns the body as a numpy array.
        """
        feature_table_array = self.feature_table.to_array()
        batch_table_array = self.batch_table.to_array()
        return np.concatenate((feature_table_array, batch_table_array))

    @staticmethod
    def from_array(header: PntsHeader, array: npt.NDArray) -> PntsBody:
        """
        Creates a PntsBody from an array and the header
        """

        # build feature table
        feature_table_size = header.ft_json_byte_length + header.ft_bin_byte_length
        feature_table_array = array[0:feature_table_size]
        feature_table = FeatureTable.from_array(header, feature_table_array)

        # build batch table
        batch_table_size = header.bt_json_byte_length + header.bt_bin_byte_length
        batch_table_array = array[feature_table_size:feature_table_size + batch_table_size]
        batch_table = BatchTable.from_array(header, batch_table_array)

        # build tile body with feature table
        body = PntsBody()
        body.feature_table = feature_table
        body.batch_table = batch_table

        return body
