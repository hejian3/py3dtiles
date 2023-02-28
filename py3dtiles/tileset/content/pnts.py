from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

from py3dtiles.tileset.batch_table import BatchTable
from py3dtiles.tileset.feature_table import (
    FeatureTable,
    FeatureTableHeader,
)
from .tile_content import (
    TileContent,
    TileContentBody,
    TileContentHeader,
)


class Pnts(TileContent):
    def __init__(self, header: PntsHeader, body: PntsBody) -> None:
        super().__init__()
        self.header: PntsHeader = header
        self.body: PntsBody = body
        self.sync()

    def sync(self) -> None:
        """
        Synchronizes headers with the Pnts body.
        """
        self.header.ft_json_byte_length = len(self.body.feature_table.header.to_array())
        self.header.ft_bin_byte_length = sum(
            len(array) for array in self.body.feature_table.body.to_array()
        )
        self.header.bt_json_byte_length = len(self.body.batch_table.header.to_array())
        self.header.bt_bin_byte_length = len(self.body.batch_table.body.to_array())

        self.header.tile_byte_length = (
            PntsHeader.BYTE_LENGTH
            + self.header.ft_json_byte_length
            + self.header.ft_bin_byte_length
            + self.header.bt_json_byte_length
            + self.header.bt_bin_byte_length
        )

    @staticmethod
    def from_features(
        feature_table_header: FeatureTableHeader,
        position_array: npt.NDArray[np.float32 | np.uint8],
        color_array: npt.NDArray[np.uint8 | np.uint16] | None = None,
        normal_position: npt.NDArray[np.float32 | np.uint8] | None = None,
    ) -> Pnts:
        """
        Creates a Pnts from features defined by pd_type and cd_type.
        """
        pnts_body = PntsBody()
        pnts_body.feature_table = FeatureTable.from_features(
            feature_table_header, position_array, color_array, normal_position
        )

        pnts = Pnts(PntsHeader(), pnts_body)
        pnts.sync()

        return pnts

    @staticmethod
    def from_array(array: npt.NDArray) -> Pnts:
        """
        Creates a Pnts from an array
        """

        # build tile header
        h_arr = array[0 : PntsHeader.BYTE_LENGTH]
        pnts_header = PntsHeader.from_array(h_arr)

        if pnts_header.tile_byte_length != len(array):
            raise RuntimeError(
                f"Invalid byte length in header, this tile has a length of {len(array)} but the length in the header is {pnts_header.tile_byte_length}"
            )

        # build tile body
        b_len = (
            pnts_header.ft_json_byte_length
            + pnts_header.ft_bin_byte_length
            + pnts_header.bt_json_byte_length
            + pnts_header.bt_bin_byte_length
        )
        b_arr = array[PntsHeader.BYTE_LENGTH : PntsHeader.BYTE_LENGTH + b_len]
        pnts_body = PntsBody.from_array(pnts_header, b_arr)

        # build the tile with header and body
        return Pnts(pnts_header, pnts_body)


class PntsHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self) -> None:
        super().__init__()
        self.magic_value = b"pnts"
        self.version = 1

    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Returns the header as a numpy array.
        """
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array(
            [
                self.version,
                self.tile_byte_length,
                self.ft_json_byte_length,
                self.ft_bin_byte_length,
                self.bt_json_byte_length,
                self.bt_bin_byte_length,
            ],
            dtype=np.uint32,
        )

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

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
        feature_table_array = array[:feature_table_size]
        feature_table = FeatureTable.from_array(header, feature_table_array)

        # build batch table
        batch_table_size = header.bt_json_byte_length + header.bt_bin_byte_length
        batch_table_array = array[
            feature_table_size : feature_table_size + batch_table_size
        ]
        batch_table = BatchTable.from_array(
            header, batch_table_array, feature_table.nb_points()
        )

        # build tile body with feature table
        body = PntsBody()
        body.feature_table = feature_table
        body.batch_table = batch_table

        return body
