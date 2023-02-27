from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

from py3dtiles.tileset.batch_table import BatchTable
from py3dtiles.tileset.feature_table import (
    Feature,
    FeatureTable,
    FeatureTableBody,
    FeatureTableHeader,
    SemanticPoint,
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

    def sync(self) -> None:
        """
        Synchronizes headers with the Pnts body.
        """

        # extract arrays
        feature_table_header_array = self.body.feature_table.header.to_array()
        feature_table_body_array = self.body.feature_table.body.to_array()
        batch_table_header_array = self.body.batch_table.header.to_array()
        batch_table_body_array = self.body.batch_table.body.to_array()

        # sync the tile header with feature table contents
        self.header.tile_byte_length = (
            len(feature_table_header_array)
            + len(feature_table_body_array)
            + len(batch_table_header_array)
            + len(batch_table_body_array)
            + PntsHeader.BYTE_LENGTH
        )
        self.header.ft_json_byte_length = len(feature_table_header_array)
        self.header.ft_bin_byte_length = len(feature_table_body_array)
        self.header.bt_json_byte_length = len(batch_table_header_array)
        self.header.bt_bin_byte_length = len(batch_table_body_array)

    @staticmethod
    def from_features(
        pd_type: npt.DTypeLike, cd_type: npt.DTypeLike, features: list[Feature]
    ) -> Pnts:
        """
        Creates a Pnts from features defined by pd_type and cd_type.
        """

        ft = FeatureTable.from_features(pd_type, cd_type, features)

        pnts_body = PntsBody()
        pnts_body.feature_table = ft

        return Pnts(PntsHeader(), pnts_body)

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

    @staticmethod
    def from_points(
        points: npt.NDArray[np.uint8], include_rgb: bool, include_classification: bool
    ) -> Pnts:
        if len(points) == 0:
            raise ValueError("The argument points cannot be empty.")

        point_size = (
            3 * 4 + (3 if include_rgb else 0) + (1 if include_classification else 0)
        )

        if len(points) % point_size != 0:
            raise ValueError(
                f"The length of points array is {len(points)} but the point size is {point_size}."
                f"There is a rest of {len(points) % point_size}"
            )

        count = len(points) // point_size

        pdt = np.dtype([("X", "<f4"), ("Y", "<f4"), ("Z", "<f4")])
        cdt = (
            np.dtype([("Red", "u1"), ("Green", "u1"), ("Blue", "u1")])
            if include_rgb
            else None
        )

        ft = FeatureTable()
        ft.header = FeatureTableHeader.from_dtype(pdt, cdt, count)
        ft.body = FeatureTableBody.from_array(ft.header, points)
        bt = BatchTable()
        if include_classification:
            sdt = np.dtype([("Classification", "u1")])
            offset = count * (3 * 4 + (3 if include_rgb else 0))
            bt.add_property_as_binary(
                "Classification",
                points[offset : offset + count * sdt.itemsize],
                "UNSIGNED_BYTE",
                "SCALAR",
            )

        body = PntsBody()
        body.feature_table = ft
        body.batch_table = bt

        pnts = Pnts(PntsHeader(), body)
        pnts.sync()

        return pnts


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

    def get_points(
        self, transform: npt.NDArray[np.number] | None
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        fth = self.feature_table.header

        xyz = self.feature_table.body.positions_arr.view(np.float32).reshape((-1, 3))
        if fth.colors == SemanticPoint.RGB:
            rgb = self.feature_table.body.colors_arr.reshape((-1, 3))
        else:
            rgb = None

        if transform is not None:
            transform = transform.reshape((4, 4))
            xyzw = np.vstack((xyz, np.ones(xyz.shape[0], dtype=xyz.dtype))).transpose()
            xyz = np.dot(xyzw, transform.T)[:, :3]

        return xyz, rgb

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
