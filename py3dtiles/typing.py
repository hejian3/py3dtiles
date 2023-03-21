from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    Union,
)

import numpy as np
import numpy.typing as npt
from pyproj import CRS

if TYPE_CHECKING:
    from typing_extensions import NotRequired

# Tileset types

ExtensionDictType = Dict[str, Any]
ExtraDictType = Dict[str, Any]
GeometricErrorType = float
PropertyType = Dict[str, Any]
RefineType = Literal["ADD", "REPLACE"]
TransformDictType = List[float]


class ThreeDDictBase(TypedDict):
    extensions: NotRequired[dict[str, ExtensionDictType]]
    extras: NotRequired[ExtraDictType]


class BoundingVolumeBoxDictType(ThreeDDictBase):
    box: list[float]


class BoundingVolumeRegionDictType(ThreeDDictBase):
    region: list[float]


class BoundingVolumeSphereDictType(ThreeDDictBase):
    sphere: list[float]


BoundingVolumeDictType = Union[
    BoundingVolumeBoxDictType,
    BoundingVolumeRegionDictType,
    BoundingVolumeSphereDictType,
]


class ContentType(ThreeDDictBase):
    boundingVolume: NotRequired[BoundingVolumeDictType]
    uri: str


class PropertyDictType(ThreeDDictBase):
    maximum: float
    minimum: float


class AssetDictType(ThreeDDictBase):
    version: str
    tilesetVersion: NotRequired[str]


class TileDictType(ThreeDDictBase):
    boundingVolume: BoundingVolumeDictType
    geometricError: GeometricErrorType
    viewerRequestVolume: NotRequired[BoundingVolumeDictType]
    refine: NotRequired[RefineType]
    transform: NotRequired[TransformDictType]
    content: NotRequired[ContentType]
    children: NotRequired[list[TileDictType]]


class TilesetDictType(ThreeDDictBase):
    asset: AssetDictType
    geometricError: GeometricErrorType
    root: TileDictType
    properties: NotRequired[PropertyType]
    extensionsUsed: NotRequired[list[str]]


# Tile content types

BatchTableHeaderDataType = Dict[str, Union[List[Any], Dict[str, Any]]]

FeatureTableHeaderDataType = Dict[
    str,
    Union[
        int,  # points_length
        Dict[str, int],  # byte offsets
        Tuple[float, float, float],  # rtc
        List[float],  # quantized_volume_offset and quantized_volume_scale
        Tuple[int, int, int, int],  # constant_rgba
    ],
]

# Tiler types

PortionItemType = Tuple[int, ...]
PortionsType = List[Tuple[str, PortionItemType]]


class MetadataReaderType(TypedDict):
    portions: PortionsType
    aabb: npt.NDArray[np.double]
    crs_in: CRS | None
    point_count: int
    avg_min: npt.NDArray[np.double]


OffsetScaleType = Tuple[
    npt.NDArray[np.double],
    npt.NDArray[np.double],
    Optional[npt.NDArray[np.double]],
    Optional[float],
]
