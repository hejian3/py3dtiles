import doctest
from pathlib import Path

if __name__ == "__main__":
    # With doctest.ELLIPSIS, an ellipsis marker (...) in the expected output can match any substring in the actual output.
    # Useful to match strings like "<py3dtiles.tileset.content.pnts.PntsHeader object at 0x7f73f5530d90>"
    test_result = doctest.testfile("../docs/api.rst", optionflags=doctest.ELLIPSIS)

    # Remove files created by the tested files.
    Path("mymodel.b3dm").unlink(missing_ok=True)
    Path("mypoints.pnts").unlink(missing_ok=True)

    exit(test_result.failed != 0)
