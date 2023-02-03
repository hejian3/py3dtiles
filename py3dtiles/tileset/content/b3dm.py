from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

from py3dtiles.tileset.batch_table import BatchTable
from .gltf import GlTF
from .tile_content import (
    TileContent,
    TileContentBody,
    TileContentHeader,
)


class B3dm(TileContent):
    def __init__(self, header: B3dmHeader, body: B3dmBody) -> None:
        super().__init__()

        self.header: B3dmHeader = header
        self.body: B3dmBody = body

    @staticmethod
    def from_glTF(gltf, bt=None):
        """
        Parameters
        ----------
        gltf : GlTF
            glTF object representing a set of objects

        bt : Batch Table (optional)
            BatchTable object containing per-feature metadata

        Returns
        -------
        tile : TileContent
        """

        b3dm_body = B3dmBody()
        b3dm_body.glTF = gltf
        b3dm_body.batch_table = bt

        b3dm_header = B3dmHeader()
        b3dm_header.sync(b3dm_body)

        return B3dm(b3dm_header, b3dm_body)

    @staticmethod
    def from_array(array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        t : TileContent
        """

        # build tile header
        h_arr = array[0 : B3dmHeader.BYTE_LENGTH]
        b3dm_header = B3dmHeader.from_array(h_arr)

        if b3dm_header.tile_byte_length != len(array):
            raise RuntimeError("Invalid byte length in header")

        # build tile body
        b_arr = array[B3dmHeader.BYTE_LENGTH : b3dm_header.tile_byte_length]
        b3dm_body = B3dmBody.from_array(b3dm_header, b_arr)

        # build tile with header and body
        return B3dm(b3dm_header, b3dm_body)


class B3dmHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self):
        super().__init__()
        self.magic_value = b"b3dm"
        self.version = 1

    def to_array(self):
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

    def sync(self, body):
        """
        Allow to synchronize headers with contents.
        """

        # extract array
        glTF_arr = body.glTF.to_array()

        # sync the tile header with feature table contents
        self.tile_byte_length = len(glTF_arr) + B3dmHeader.BYTE_LENGTH
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0

        if body.batch_table is not None:
            bth_arr = body.batch_table.to_array()

            self.tile_byte_length += len(bth_arr)
            self.bt_json_byte_length = len(bth_arr)

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> B3dmHeader:
        h = B3dmHeader()

        if len(array) != B3dmHeader.BYTE_LENGTH:
            raise RuntimeError("Invalid header length")

        h.version = struct.unpack("i", array[4:8].tobytes())[0]
        h.tile_byte_length = struct.unpack("i", array[8:12].tobytes())[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16].tobytes())[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20].tobytes())[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24].tobytes())[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28].tobytes())[0]

        return h


class B3dmBody(TileContentBody):
    def __init__(self):
        self.batch_table = BatchTable()
        self.glTF = GlTF()

    def to_array(self):
        # TODO : export feature table
        array = self.glTF.to_array()
        if self.batch_table is not None:
            array = np.concatenate((self.batch_table.to_array(), array))
        return array

    @staticmethod
    def from_glTF(glTF):
        """
        Parameters
        ----------
        th : TileContentHeader

        glTF : GlTF

        Returns
        -------
        b : TileContentBody
        """

        # build tile body
        b = B3dmBody()
        b.glTF = glTF

        return b

    @staticmethod
    def from_array(b3dm_header: B3dmHeader, array: npt.NDArray[np.uint8]) -> B3dmBody:
        # build feature table
        ft_len = b3dm_header.ft_json_byte_length + b3dm_header.ft_bin_byte_length

        # build batch table
        bt_len = b3dm_header.bt_json_byte_length + b3dm_header.bt_bin_byte_length

        # build glTF
        glTF_len = (
            b3dm_header.tile_byte_length - ft_len - bt_len - B3dmHeader.BYTE_LENGTH
        )
        glTF_arr = array[ft_len + bt_len : ft_len + bt_len + glTF_len]
        glTF = GlTF.from_array(glTF_arr)

        # build tile body with batch table
        b = B3dmBody()
        b.glTF = glTF
        if b3dm_header.bt_json_byte_length > 0:
            b.batch_table = BatchTable.from_array(
                b3dm_header, array[0 : b3dm_header.bt_json_byte_length]
            )

        return b
