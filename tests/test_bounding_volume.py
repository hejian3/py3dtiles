import unittest

from py3dtiles.bounding_volume import BoundingVolume


class TestBoundingVolume(unittest.TestCase):

    def test_constructor(self):
        bounding_volume = BoundingVolume()
        self.assertDictEqual(bounding_volume._extensions, {})

    def test_is_not_box_not_region_not_sphere(self):
        bounding_volume = BoundingVolume()
        self.assertFalse(bounding_volume.is_box())
        self.assertFalse(bounding_volume.is_region())
        self.assertFalse(bounding_volume.is_sphere())


if __name__ == "__main__":
    unittest.main()
