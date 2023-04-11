from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit  # type: ignore [attr-defined]
from numba.typed import List
import numpy as np

from py3dtiles.exceptions import TilerException
from py3dtiles.utils import aabb_size_to_subdivision_type, SubdivisionType
from .distance import is_point_far_enough, xyz_to_key

if TYPE_CHECKING:
    from .node import Node


@njit(fastmath=True, cache=True)
def _insert(
    cells_xyz,
    cells_rgb,
    cells_classification,
    cells_intensity,
    aabmin,
    inv_aabb_size,
    cell_count,
    xyz,
    rgb,
    classification,
    intensity,
    spacing,
    shift,
    force=False,
):
    keys = xyz_to_key(xyz, cell_count, aabmin, inv_aabb_size, shift)
    if force:
        # allocate this one once and for all
        for k in np.unique(keys):
            idx = np.where(keys - k == 0)
            cells_xyz[k] = np.concatenate((cells_xyz[k], xyz[idx]))
            cells_rgb[k] = np.concatenate((cells_rgb[k], rgb[idx]))
            cells_classification[k] = np.concatenate(
                (cells_classification[k], classification[idx])
            )
            cells_intensity[k] = np.concatenate(
                (cells_intensity[k], intensity[idx])
            )
    else:
        notinserted = np.full(len(xyz), False)
        needs_balance = False
        for i in range(len(xyz)):
            k = keys[i]
            if cells_xyz[k].shape[0] == 0 or is_point_far_enough(
                cells_xyz[k], xyz[i], spacing
            ):
                cells_xyz[k] = np.concatenate((cells_xyz[k], xyz[i].reshape(1, 3)))
                cells_rgb[k] = np.concatenate((cells_rgb[k], rgb[i].reshape(1, 3)))

                cells_classification[k] = np.concatenate(
                    (cells_classification[k], classification[i].reshape(1, 1))
                )
                cells_intensity[k] = np.concatenate(
                    (cells_intensity[k], intensity[i].reshape(1, 1))
                )

                if cell_count[0] < 8:
                    needs_balance = needs_balance or cells_xyz[k].shape[0] > 200000
            else:
                notinserted[i] = True

        return (
            xyz[notinserted],
            rgb[notinserted],
            classification[notinserted],
            intensity[notinserted],
            needs_balance,
        )


class Grid:
    """docstring for Grid"""

    __slots__ = (
        "cell_count",
        "cells_xyz",
        "cells_rgb",
        "cells_classification",
        "cells_intensity",
        "spacing",
    )

    def __init__(self, node: Node, initial_count: int = 3) -> None:
        self.cell_count = np.array(
            [initial_count, initial_count, initial_count], dtype=np.int32
        )
        self.spacing = node.spacing * node.spacing

        self.cells_xyz = List()
        self.cells_rgb = List()
        self.cells_classification = List()
        self.cells_intensity = List()
        for _ in range(self.max_key_value):
            self.cells_xyz.append(np.zeros((0, 3), dtype=np.float32))
            self.cells_rgb.append(np.zeros((0, 3), dtype=np.uint8))
            self.cells_classification.append(np.zeros((0, 1), dtype=np.uint8))
            self.cells_intensity.append(np.zeros((0, 1), dtype=np.uint8))

    def __getstate__(self) -> dict:
        return {
            "cell_count": self.cell_count,
            "spacing": self.spacing,
            "cells_xyz": list(self.cells_xyz),
            "cells_rgb": list(self.cells_rgb),
            "cells_classification": list(self.cells_classification),
            "cells_intensity": list(self.cells_intensity),
        }

    def __setstate__(self, state: dict) -> None:
        self.cell_count = state["cell_count"]
        self.spacing = state["spacing"]
        self.cells_xyz = List(state["cells_xyz"])
        self.cells_rgb = List(state["cells_rgb"])
        self.cells_classification = List(state["cells_classification"])
        self.cells_intensity = List(state["cells_intensity"])

    @property
    def max_key_value(self) -> int:
        return 1 << (
            2 * int(self.cell_count[0]).bit_length()
            + int(self.cell_count[2]).bit_length()
        )

    def insert(
        self,
        aabmin: np.ndarray,
        inv_aabb_size: np.ndarray,
        xyz: np.ndarray,
        rgb: np.ndarray,
        classification: np.ndarray,
        intensity: np.ndarray,
        force: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        return _insert(
            self.cells_xyz,
            self.cells_rgb,
            self.cells_classification,
            self.cells_intensity,
            aabmin,
            inv_aabb_size,
            self.cell_count,
            xyz,
            rgb,
            classification,
            intensity,
            self.spacing,
            int(self.cell_count[0] - 1).bit_length(),
            force,
        )

    def needs_balance(self) -> bool:
        if self.cell_count[0] < 8:
            for cell in self.cells_xyz:
                if cell.shape[0] > 100000:
                    return True
        return False

    def balance(
        self, aabb_size: np.ndarray, aabmin: np.ndarray, inv_aabb_size: np.ndarray
    ) -> None:
        t = aabb_size_to_subdivision_type(aabb_size)
        self.cell_count[0] += 1
        self.cell_count[1] += 1
        if t != SubdivisionType.QUADTREE:
            self.cell_count[2] += 1
        if self.cell_count[0] > 8:
            raise TilerException(
                f"The first value of the attribute cell count must be lower or equal to 8, "
                f"currently, it is {self.cell_count[0]}"
            )

        old_cells_xyz = self.cells_xyz
        old_cells_rgb = self.cells_rgb
        old_cells_classification = self.cells_classification
        old_cells_intensity = self.cells_intensity
        self.cells_xyz = List()
        self.cells_rgb = List()
        self.cells_classification = List()
        self.cells_intensity = List()
        for _ in range(self.max_key_value):
            self.cells_xyz.append(np.zeros((0, 3), dtype=np.float32))
            self.cells_rgb.append(np.zeros((0, 3), dtype=np.uint8))
            self.cells_classification.append(np.zeros((0, 1), dtype=np.uint8))
            self.cells_intensity.append(np.zeros((0, 1), dtype=np.uint8))

        for cellxyz, cellrgb, cellclassification, cellsintensity in zip(
            old_cells_xyz, old_cells_rgb, old_cells_classification, old_cells_intensity
        ):
            self.insert(
                aabmin, inv_aabb_size, cellxyz, cellrgb, cellclassification, cellsintensity, True
            )

    def get_points(self, include_rgb: bool, include_classification: bool, include_intensity: bool) -> np.ndarray:
        xyz = []
        rgb = []
        classification = []
        intensity = []
        for i in range(len(self.cells_xyz)):
            xyz.append(self.cells_xyz[i].view(np.uint8).ravel())
            if include_rgb:
                rgb.append(self.cells_rgb[i].ravel())
            if include_classification:
                classification.append(self.cells_classification[i].ravel())
            if include_intensity:
                intensity.append(self.cells_intensity[i].ravel())

        return np.concatenate((*xyz, *rgb, *classification, *intensity))

    def get_point_count(self) -> int:
        pt = 0
        for i in range(len(self.cells_xyz)):
            pt += self.cells_xyz[i].shape[0]
        return pt
