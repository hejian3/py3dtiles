import struct

import numpy as np

from .feature_table import FeatureTable
from .tile_content import TileContent, TileContentBody, TileContentHeader, TileContentType


class Pnts(TileContent):

    @classmethod
    def from_features(cls, pd_type, cd_type, features):
        """
        Parameters
        ----------
        dtype : numpy.dtype
            Numpy description of a single feature

        features : Feature[]

        Returns
        -------
        tile : TileContent
        """

        ft = FeatureTable.from_features(pd_type, cd_type, features)

        tb = PntsBody()
        tb.feature_table = ft

        th = PntsHeader()

        t = cls()
        t.body = tb
        t.header = th

        return t

    @classmethod
    def from_array(cls, array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        t : TileContent
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
        t = cls()
        t.header = h
        t.body = b

        return t


class PntsHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self):
        self.type = TileContentType.POINT_CLOUD
        self.magic_value = b"pnts"
        self.version = 1
        self.tile_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0

    def to_array(self):
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array([self.version,
                                self.tile_byte_length,
                                self.ft_json_byte_length,
                                self.ft_bin_byte_length,
                                self.bt_json_byte_length,
                                self.bt_bin_byte_length], dtype=np.uint32)

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

    def sync(self, body):
        """
        Allow to synchronize headers with contents.
        """

        # extract array
        fth_arr = body.feature_table.header.to_array()
        ftb_arr = body.feature_table.body.to_array()

        # sync the tile header with feature table contents
        self.tile_byte_length = (len(fth_arr) + len(ftb_arr)
                                 + PntsHeader.BYTE_LENGTH)
        self.ft_json_byte_length = len(fth_arr)
        self.ft_bin_byte_length = len(ftb_arr)

    @classmethod
    def from_array(cls, array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        h : TileContentHeader
        """

        h = cls()

        if len(array) != cls.BYTE_LENGTH:
            raise RuntimeError("Invalid header length")

        h.magic_value = "pnts"
        h.version = struct.unpack("i", array[4:8])[0]
        h.tile_byte_length = struct.unpack("i", array[8:12])[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16])[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20])[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24])[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28])[0]

        h.type = TileContentType.POINT_CLOUD

        return h


class PntsBody(TileContentBody):
    def __init__(self):
        self.feature_table = FeatureTable()
        # TODO : self.batch_table = BatchTable()

    def to_array(self):
        return self.feature_table.to_array()

    @classmethod
    def from_array(cls, th, array):
        """
        Parameters
        ----------
        th : TileContentHeader

        array : numpy.array

        Returns
        -------
        b : TileContentBody
        """

        # build feature table
        ft_len = th.ft_json_byte_length + th.ft_bin_byte_length
        ft_arr = array[0:ft_len]
        ft = FeatureTable.from_array(th, ft_arr)

        # build batch table
        # bt_len = th.bt_json_byte_length + th.bt_bin_byte_length
        # bt_arr = array[ft_len:ft_len+ba_len]
        # bt = BatchTable.from_array(th, bt_arr)

        # build tile body with feature table
        b = cls()
        b.feature_table = ft

        return b
