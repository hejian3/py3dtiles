from typing import List
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox


# fmt: off
DUMMY_MATRIX = [
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
]
# fmt: on


class TestBoundingVolumeBox(unittest.TestCase):
    @classmethod
    def build_box_sample(cls) -> BoundingVolumeBox:
        bounding_volume_box = BoundingVolumeBox()
        bounding_volume_box.set_from_list(DUMMY_MATRIX)
        return bounding_volume_box

    def test_constructor(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        self.assertTrue(bounding_volume_box._box.ndim == 0)

    def test_set_from_list(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        bounding_volume_box.set_from_list(DUMMY_MATRIX)
        assert_array_equal(bounding_volume_box._box, np.array(DUMMY_MATRIX))

    def test_set_from_invalid_list(self) -> None:
        bounding_volume_box = BoundingVolumeBox()

        # Empty list
        bounding_volume_list: List[float] = []
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(bounding_volume_list)
        self.assertTrue(bounding_volume_box._box.ndim == 0)

        # Too small list
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(DUMMY_MATRIX[:-1])
        self.assertTrue(bounding_volume_box._box.ndim == 0)

        # Too long list
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(DUMMY_MATRIX + [13])
        self.assertTrue(bounding_volume_box._box.ndim == 0)

        # Not only number
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(DUMMY_MATRIX[:-1] + ["a"])
        self.assertTrue(bounding_volume_box._box.ndim == 0)

        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(DUMMY_MATRIX[:-1] + [[1]])
        self.assertTrue(bounding_volume_box._box.ndim == 0)

    def test_set_from_points(self) -> None:
        pass

    def test_set_from_invalid_points(self) -> None:
        # what if I give only one point ?
        pass

    def test_is_only_box(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        self.assertTrue(bounding_volume_box.is_box())
        self.assertFalse(bounding_volume_box.is_region())
        self.assertFalse(bounding_volume_box.is_sphere())

    def test_get_center(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.get_center()

        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()
        assert_array_equal(bounding_volume_box.get_center(), [1, 2, 3])

    def test_translate(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.translate(np.array([-1, -2, -3]))

        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()
        assert_array_equal(bounding_volume_box.get_center(), [1, 2, 3])

        bounding_volume_box.translate(np.array([-1, -2, -3]))
        # Should move only the center
        # fmt: off
        expected_result = [
            0, 0, 0, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
        ]
        # fmt: on
        assert_array_equal(bounding_volume_box._box, expected_result)

    def test_transform(self) -> None:
        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()

        # Assert box hasn't change after transformation with identity matrix
        transformer = np.identity(4).reshape(-1)
        bounding_volume_box.transform(transformer)
        assert_array_equal(bounding_volume_box._box, DUMMY_MATRIX)

        # Assert box is translated by [10, 10, 10] on X,Y, Z axis
        transformer[12:15] = 10
        bounding_volume_box.transform(transformer)
        # fmt: off
        expected_result = [
            11, 12, 13, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
        ]
        # fmt: on
        assert_array_equal(bounding_volume_box._box, expected_result)

        # Assert box is reversed
        transformer = -np.identity(4).reshape(-1)
        bounding_volume_box.transform(transformer)
        # fmt: off
        expected_result = [
            -11, -12, -13, -4,
            -5, -6, -7, -8,
            -9, -10, -11, -12,
        ]
        # fmt: on
        assert_array_equal(bounding_volume_box._box, expected_result)

    def test_get_corners(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.get_corners()

        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()
        assert_array_equal(
            bounding_volume_box.get_corners(),
            [  # almost a kindness test
                [-20, -22, -24],
                [-12, -12, -12],
                [-6, -6, -6],
                [2, 4, 6],
                [0, 0, 0],
                [8, 10, 12],
                [14, 16, 18],
                [22, 26, 30],
            ],
        )

    def test_get_canonical_as_array(self) -> None:
        pass

    def test_to_dict(self) -> None:
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.to_dict()

        self.assertDictEqual(
            TestBoundingVolumeBox.build_box_sample().to_dict(),
            {"box": DUMMY_MATRIX},
        )


if __name__ == "__main__":
    unittest.main()
