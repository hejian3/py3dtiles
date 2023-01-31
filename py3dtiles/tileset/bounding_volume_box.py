from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .bounding_volume import BoundingVolume

if TYPE_CHECKING:
    from .tile import Tile

# In order to prevent the appearance of ghost newline characters ("\n")
# when printing a numpy.array (mainly self._box in this file):
np.set_printoptions(linewidth=500)


class BoundingVolumeBox(BoundingVolume):
    """
    A box bounding volume as defined in the 3DTiles specifications i.e. an
    array of 12 numbers that define an oriented bounding box:
    - The first three elements define the x, y, and z values for the
    center of the box.
    - The next three elements (with indices 3, 4, and 5) define the x axis
    direction and half-length.
    - The next three elements (with indices 6, 7, and 8) define the y axis
    direction and half-length.
    - The last three elements (indices 9, 10, and 11) define the z axis
    direction and half-length.

    Note that, by default, a box bounding volume doesn't need to be aligned
    with the coordinate axis. Still in general, computing the box bounding
    volume of two box bounding volumes won't necessarily yield a box that is
    aligned with the coordinate axis (although this computation might require
    some fitting algorithm e.g. the principal component analysis method.
    Yet in sake of simplification (and numerical efficiency), when asked to
    "add" (i.e. to find the enclosing box of) two (or more) box bounding
    volumes this class resolves to compute the "canonical" fitting/enclosing
    box i.e. a box that is parallel to the coordinate axis.
    """

    def __init__(self):
        super().__init__()
        self._box = None

    def clone(self) -> BoundingVolumeBox:
        new = BoundingVolumeBox()
        new._box = copy.copy(self._box)
        return new

    def get_center(self) -> np.ndarray:
        if self._box is None:
            raise AttributeError('Bounding Volume Box is not defined.')

        return self._box[0: 3]

    def translate(self, offset: npt.ArrayLike) -> None:
        """
        Translate the box center with the given offset "vector"
        :param offset: the 3D vector by which the box should be translated
        """
        if self._box is None:
            raise AttributeError('Bounding Volume Box is not defined.')

        for i in range(0, 3):
            self._box[i] += offset[i]

    def transform(self, transform: npt.ArrayLike) -> None:
        """
        Apply the provided transformation matrix (4x4) to the box
        :param transform: transformation matrix (4x4) to be applied
        """
        rotation = np.array([transform[0:3],
                             transform[4:7],
                             transform[8:11]])

        center = self._box[0: 3]
        x_half_axis = self._box[3: 6]
        y_half_axis = self._box[6: 9]
        z_half_axis = self._box[9:12]

        # Apply the rotation part to each element
        new_center = rotation.dot(center)
        new_x_half_axis = rotation.dot(x_half_axis)
        new_y_half_axis = rotation.dot(y_half_axis)
        new_z_half_axis = rotation.dot(z_half_axis)
        self._box = np.concatenate((new_center,
                                    new_x_half_axis,
                                    new_y_half_axis,
                                    new_z_half_axis))
        offset = np.array(transform[12:15])
        self.translate(offset)

    def is_box(self) -> bool:
        return True

    def set_from_list(self, box_list: npt.ArrayLike) -> BoundingVolumeBox:
        box = np.array(box_list, dtype=float)

        valid, reason = BoundingVolumeBox.is_valid(box)
        if not valid:
            raise ValueError(reason)
        self._box = box

        return self

    def set_from_array(self, box_array: np.ndarray) -> BoundingVolumeBox:
        box = box_array.astype(float)

        valid, reason = BoundingVolumeBox.is_valid(box)
        if not valid:
            raise ValueError(reason)
        self._box = box

        return self

    def set_from_points(self, points: list) -> BoundingVolumeBox:
        box = BoundingVolumeBox.get_box_array_from_point(points)

        valid, reason = BoundingVolumeBox.is_valid(box)
        if not valid:
            raise ValueError(reason)
        self._box = box

        return self

    def set_from_mins_maxs(self, mins_maxs: npt.ArrayLike) -> BoundingVolumeBox:
        """
        :param mins_maxs: the list [x_min, y_min, z_min, x_max, y_max, z_max]
                          that is the boundaries of the box along each
                          coordinate axis
        """
        self._box = BoundingVolumeBox.get_box_array_from_mins_maxs(mins_maxs)

        return self

    def get_corners(self) -> list:
        """
        :return: the corners (3D points) of the box as a list
        """
        if self._box is None:
            raise AttributeError('Bounding Volume Box is not defined.')

        center = self._box[0: 3: 1]
        x_half_axis = self._box[3: 6: 1]
        y_half_axis = self._box[6: 9: 1]
        z_half_axis = self._box[9:12: 1]

        x_axis = x_half_axis * 2
        y_axis = y_half_axis * 2
        z_axis = z_half_axis * 2

        # The eight cornering points of the box
        tmp = np.subtract(center, x_half_axis)
        tmp = np.subtract(tmp, y_half_axis)

        o = np.subtract(tmp, z_half_axis)
        ox = np.add(o, x_axis)
        oy = np.add(o, y_axis)
        oxy = np.add(o, np.add(x_axis, y_axis))

        oz = np.add(o, z_axis)
        oxz = np.add(oz, x_axis)
        oyz = np.add(oz, y_axis)
        oxyz = np.add(oz, np.add(x_axis, y_axis))

        return [o, ox, oy, oxy, oz, oxz, oyz, oxyz]

    def get_canonical_as_array(self) -> np.ndarray:
        """
        :return: the smallest enclosing box (as an array) that is parallel
                 to the coordinate axis
        """
        return BoundingVolumeBox.get_box_array_from_point(self.get_corners())

    def add(self, other: BoundingVolumeBox) -> None:
        """
        Compute the 'canonical' bounding volume fitting this bounding volume
        together with the added bounding volume. Again (refer above to the
        class definition) the computed fitting bounding volume is generically
        not the smallest one (due to its alignment with the coordinate axis).
        :param other: another box bounding volume to be added with this one
        """
        if self._box is None:
            # Then it is safe to overwrite
            self._box = other._box
            return

        corners = self.get_corners() + other.get_corners()
        self.set_from_points(corners)

    def sync_with_children(self, owner: Tile) -> None:
        # We reset to some dummy state of this Bounding Volume Box so we
        # can add up in place the boxes of the owner's children
        # If there is no child, no modifications are done.
        for child in owner.get_direct_children():
            bounding_volume = copy.deepcopy(child.bounding_volume)
            bounding_volume.transform(child.get_transform())
            if not bounding_volume.is_box():
                raise AttributeError("All children should also have a box as bounding volume"
                                     "if the parent has a bounding box")
            self.add(bounding_volume)

    def to_dict(self) -> dict:
        if self._box is None:
            raise AttributeError('Bounding Volume Box is not defined.')

        return {'box': list(self._box)}

    @staticmethod
    def get_box_array_from_mins_maxs(mins_maxs: npt.ArrayLike) -> np.ndarray: # todo return smallest obb, not aabb
        """
        :param mins_maxs: the list [x_min, y_min, z_min, x_max, y_max, z_max]
                          that is the boundaries of the box along each
                          coordinate axis
        :return: the smallest box (as an array, as opposed to a
                BoundingVolumeBox instance) that encloses the given list of
                (3D) points and that is parallel to the coordinate axis.
        """
        x_min = mins_maxs[0]
        x_max = mins_maxs[3]
        y_min = mins_maxs[1]
        y_max = mins_maxs[4]
        z_min = mins_maxs[2]
        z_max = mins_maxs[5]
        new_center = np.array([(x_min + x_max) / 2,
                               (y_min + y_max) / 2,
                               (z_min + z_max) / 2])
        new_x_half_axis = np.array([(x_max - x_min) / 2, 0, 0])
        new_y_half_axis = np.array([0, (y_max - y_min) / 2, 0])
        new_z_half_axis = np.array([0, 0, (z_max - z_min) / 2])

        return np.concatenate((new_center,
                               new_x_half_axis,
                               new_y_half_axis,
                               new_z_half_axis))

    @staticmethod
    def get_box_array_from_point(points: list[list[float]]) -> np.ndarray:
        """
        :param points: a list of 3D points
        :return: the smallest box (as an array, as opposed to a
                BoundingVolumeBox instance) that encloses the given list of
                (3D) points and that is parallel to the coordinate axis.
        """
        return BoundingVolumeBox.get_box_array_from_mins_maxs(
            [min(c[0] for c in points),
             min(c[1] for c in points),
             min(c[2] for c in points),
             max(c[0] for c in points),
             max(c[1] for c in points),
             max(c[2] for c in points)])

    @staticmethod
    def is_valid(box) -> tuple[bool, str]:
        if box is None:
            return False, 'Bounding Volume Box is not defined.'
        if not box.ndim == 1:
            return False, 'Bounding Volume Box has wrong dimensions.'
        if not box.shape[0] == 12:
            return False, 'Warning: Bounding Volume Box must have 12 elements.'
        return True, ""
