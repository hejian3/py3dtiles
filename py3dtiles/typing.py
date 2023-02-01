from __future__ import annotations

from typing import Any, Dict, List, Literal, TYPE_CHECKING, TypedDict, Union

if TYPE_CHECKING:
    from typing_extensions import NotRequired

ExtensionDictType = Dict[str, Any]
ExtraDictType = Dict[str, Any]
GeometricErrorType = float
PropertyType = Dict[str, Any]
RefineType = Literal['ADD', 'REPLACE']
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

BoundingVolumeDictType = Union[BoundingVolumeBoxDictType, BoundingVolumeRegionDictType, BoundingVolumeSphereDictType]

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
