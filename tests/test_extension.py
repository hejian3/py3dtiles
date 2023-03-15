import unittest

from py3dtiles.tileset.extension import BaseExtension


class TestExtension(unittest.TestCase):
    def test_constructor(self) -> None:
        name = "name"
        extension = BaseExtension(name)
        self.assertEqual(extension.name, name)

    def test_to_dict(self) -> None:
        extension = BaseExtension("name")
        self.assertDictEqual(extension.to_dict(), {})


if __name__ == "__main__":
    unittest.main()
