import unittest

from py3dtiles import Extension


class TestExtension(unittest.TestCase):

    def test_constructor(self):
        name = "name"
        extension = Extension(name)
        self.assertEqual(extension.name, name)

    def test_to_dict(self):
        extension = Extension("name")
        self.assertDictEqual(extension.to_dict(), {})


if __name__ == "__main__":
    unittest.main()
