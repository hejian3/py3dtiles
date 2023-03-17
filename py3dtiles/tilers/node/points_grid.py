from __future__ import annotations

from typing import Any, TYPE_CHECKING

from numba import njit  # type: ignore [attr-defined]
from numba.typed import List
import numpy as np
import numpy.typing as npt

from py3dtiles.utils import aabb_size_to_subdivision_type, SubdivisionType
from .distance import is_point_far_enough, xyz_to_key

if TYPE_CHECKING:
    from .node import Node


@njit(fastmath=True, cache=True)  # type: ignore [misc]
def _insert(
    cells_xyz: List[npt.NDArray[np.float32]],
    cells_rgb: List[npt.NDArray[np.uint8]],
    cells_classification: List[npt.NDArray[np.uint8]],
    aabmin: npt.NDArray[np.float32],
    inv_aabb_size: npt.NDArray[np.float32],
    cell_count: npt.NDArray[np.int32],
    xyz: npt.NDArray[np.float32],
    rgb: npt.NDArray[np.uint8],
    classification: npt.NDArray[np.uint8],
    spacing: float,
    shift: int,
    force: bool = False,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint8], bool
] | None:
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
        return None
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
                if cell_count[0] < 8:
                    needs_balance = needs_balance or cells_xyz[k].shape[0] > 200000
            else:
                notinserted[i] = True

        return (
            xyz[notinserted],
            rgb[notinserted],
            classification[notinserted],
            needs_balance,
        )


class Grid:
    """docstring for Grid"""

    __slots__ = (
        "cell_count",
        "cells_xyz",
        "cells_rgb",
        "cells_classification",
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
        for _ in range(self.max_key_value):
            self.cells_xyz.append(np.zeros((0, 3), dtype=np.float32))
            self.cells_rgb.append(np.zeros((0, 3), dtype=np.uint8))
            self.cells_classification.append(np.zeros((0, 1), dtype=np.uint8))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "cell_count": self.cell_count,
            "spacing": self.spacing,
            "cells_xyz": list(self.cells_xyz),
            "cells_rgb": list(self.cells_rgb),
            "cells_classification": list(self.cells_classification),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cell_count = state["cell_count"]
        self.spacing = state["spacing"]
        self.cells_xyz = List(state["cells_xyz"])
        self.cells_rgb = List(state["cells_rgb"])
        self.cells_classification = List(state["cells_classification"])

    @property
    def max_key_value(self) -> int:
        return 1 << (
            2 * int(self.cell_count[0]).bit_length()
            + int(self.cell_count[2]).bit_length()
        )

    def insert(
        self,
        aabmin: npt.NDArray[np.float32],
        inv_aabb_size: npt.NDArray[np.float32],
        xyz: npt.NDArray[np.float32],
        rgb: npt.NDArray[np.uint8],
        classification: npt.NDArray[np.uint8],
        force: bool = False,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint8], bool
    ]:
        return _insert(
            self.cells_xyz,
            self.cells_rgb,
            self.cells_classification,
            aabmin,
            inv_aabb_size,
            self.cell_count,
            xyz,
            rgb,
            classification,
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
        self,
        aabb_size: npt.NDArray[np.float32],
        aabmin: npt.NDArray[np.float32],
        inv_aabb_size: npt.NDArray[np.float32],
    ) -> None:
        t = aabb_size_to_subdivision_type(aabb_size)
        self.cell_count[0] += 1
        self.cell_count[1] += 1
        if t != SubdivisionType.QUADTREE:
            self.cell_count[2] += 1
        if self.cell_count[0] > 8:
            raise ValueError(
                f"The first value of the attribute cell count should be lower or equal to 8,"
                f"actual it is {self.cell_count[0]}"
            )

        old_cells_xyz = self.cells_xyz
        old_cells_rgb = self.cells_rgb
        old_cells_classification = self.cells_classification
        self.cells_xyz = List()
        self.cells_rgb = List()
        self.cells_classification = List()
        for _ in range(self.max_key_value):
            self.cells_xyz.append(np.zeros((0, 3), dtype=np.float32))
            self.cells_rgb.append(np.zeros((0, 3), dtype=np.uint8))
            self.cells_classification.append(np.zeros((0, 1), dtype=np.uint8))

        for cellxyz, cellrgb, cellclassification in zip(
            old_cells_xyz, old_cells_rgb, old_cells_classification
        ):
            self.insert(
                aabmin, inv_aabb_size, cellxyz, cellrgb, cellclassification, True
            )

    def get_points(
        self, include_rgb: bool, include_classification: bool
    ) -> npt.NDArray[np.uint8]:
        xyz = []
        rgb = []
        classification = []
        for i in range(len(self.cells_xyz)):
            xyz.append(self.cells_xyz[i].view(np.uint8).ravel())
            if include_rgb:
                rgb.append(self.cells_rgb[i].ravel())
            if include_classification:
                classification.append(self.cells_classification[i].ravel())

        return np.concatenate((*xyz, *rgb, *classification))

    def get_point_count(self) -> int:
        pt = 0
        for i in range(len(self.cells_xyz)):
            pt += self.cells_xyz[i].shape[0]
        return pt
