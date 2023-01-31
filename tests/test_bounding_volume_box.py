import unittest

import numpy as np
from numpy.testing import assert_array_equal

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox


class TestBoundingVolumeBox(unittest.TestCase):

    @classmethod
    def build_box_sample(cls):
        bounding_volume_box = BoundingVolumeBox()
        bounding_volume_box.set_from_list([
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ])
        return bounding_volume_box

    def test_constructor(self):
        bounding_volume_box = BoundingVolumeBox()
        self.assertIs(bounding_volume_box._box, None)

    def test_set_from_list(self):
        bounding_volume_box = BoundingVolumeBox()
        bounding_volume_list = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ]
        bounding_volume_box.set_from_list(bounding_volume_list)
        assert_array_equal(bounding_volume_box._box, np.array(bounding_volume_list))

    def test_set_from_invalid_list(self):
        bounding_volume_box = BoundingVolumeBox()

        # Empty list
        bounding_volume_list = []
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(bounding_volume_list)
        self.assertIs(bounding_volume_box._box, None)

        # Too small list
        bounding_volume_list = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11
        ]
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(bounding_volume_list)
        self.assertIs(bounding_volume_box._box, None)

        # Too long list
        bounding_volume_list = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12, 13
        ]
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(bounding_volume_list)
        self.assertIs(bounding_volume_box._box, None)

        # Not only number
        bounding_volume_list = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 'a'
        ]
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(bounding_volume_list)
        self.assertIs(bounding_volume_box._box, None)

        bounding_volume_list = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, [1]
        ]
        with self.assertRaises(ValueError):
            bounding_volume_box.set_from_list(bounding_volume_list)
        self.assertIs(bounding_volume_box._box, None)

    def test_set_from_points(self):
        pass

    def test_set_from_invalid_points(self):
        # what if I give only one point ?
        pass

    def test_is_only_box(self):
        bounding_volume_box = BoundingVolumeBox()
        self.assertTrue(bounding_volume_box.is_box())
        self.assertFalse(bounding_volume_box.is_region())
        self.assertFalse(bounding_volume_box.is_sphere())

    def test_get_center(self):
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.get_center()

        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()
        assert_array_equal(bounding_volume_box.get_center(), [1, 2, 3])

    def test_translate(self):
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.translate(np.array([-1, -2, -3]))

        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()
        assert_array_equal(bounding_volume_box.get_center(), [1, 2, 3])

        bounding_volume_box.translate(np.array([-1, -2, -3]))
        # Should move only the center
        assert_array_equal(
            bounding_volume_box._box,
            [
                0, 0, 0, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
            ]
        )

    def test_transform(self):
        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()

        # Assert box hasn't change after transformation with identity matrix
        bounding_volume_box.transform(np.array([1, 0, 0, 0,
                                       0, 1, 0, 0,
                                       0, 0, 1, 0,
                                       0, 0, 0, 1]))
        assert_array_equal(
            bounding_volume_box._box,
            [
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
            ]
        )

        # Assert box is translated by [10, 10, 10] on X,Y, Z axis
        bounding_volume_box.transform(np.array([1, 0, 0, 0,
                                       0, 1, 0, 0,
                                       0, 0, 1, 0,
                                       10, 10, 10, 1]))
        assert_array_equal(
            bounding_volume_box._box,
            [
                11, 12, 13,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
            ]
        )

        # Assert box is reversed
        bounding_volume_box.transform(np.array([-1, 0, 0, 0,
                                       0, -1, 0, 0,
                                       0, 0, -1, 0,
                                       0, 0, 0, -1]))
        assert_array_equal(
            bounding_volume_box._box,
            [
                -11, -12, -13,
                -4, -5, -6,
                -7, -8, -9,
                -10, -11, -12
            ]
        )

    def test_get_corners(self):
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.get_corners()

        bounding_volume_box = TestBoundingVolumeBox.build_box_sample()
        assert_array_equal(bounding_volume_box.get_corners(), [  # almost a kindness test
            [-20, -22, -24],
            [-12, -12, -12],
            [-6, -6, -6],
            [2, 4, 6],
            [0, 0, 0],
            [8, 10, 12],
            [14, 16, 18],
            [22, 26, 30]])

    def test_get_canonical_as_array(self):
        pass

    def test_to_dict(self):
        bounding_volume_box = BoundingVolumeBox()
        with self.assertRaises(AttributeError):
            bounding_volume_box.to_dict()

        self.assertDictEqual(
            TestBoundingVolumeBox.build_box_sample().to_dict(),
            {'box': [
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
            ]
            }
        )


if __name__ == "__main__":
    unittest.main()
