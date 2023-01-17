"""Test the point cloud readers.

The example that is run in the test (`b9_training.ply`) comes from the [CGAL repository](https://github.com/CGAL/cgal/blob/master/Data/data/points_3/b9_training.ply). Thanks to their maintainers (for more details, please refer to CGAL, Computational Geometry Algorithms Library, https://www.cgal.org):

"""

from pathlib import Path

import numpy as np
from pytest import fixture
import zmq

from py3dtiles.convert import URI
from py3dtiles.reader import ply_reader
from py3dtiles.utils import compute_spacing


DATA_DIRECTORY = Path(__file__).parent / 'fixtures'


@fixture
def filepath():
    yield DATA_DIRECTORY / "b9_training.ply"


def test_ply_get_metadata(filepath):
    ply_metadata = ply_reader.get_metadata(path=filepath)
    expected_point_count = 22300
    expected_aabb = (
        np.array([5.966480625e+05, 2.43620015625e+05, 7.350153350830078e+01]),
        np.array([5.967389375e+05, 2.43731984375e+05, 9.718580627441406e+01]),
    )
    assert list(ply_metadata.keys()) == [
        "portions", "aabb", "color_scale", "srs_in", "point_count", "avg_min"
    ]
    assert ply_metadata["portions"] == [
        (str(filepath), (0, expected_point_count, expected_point_count))
    ]
    assert np.all(ply_metadata["aabb"][0] == expected_aabb[0])
    assert np.all(ply_metadata["aabb"][1] == expected_aabb[1])
    assert ply_metadata["color_scale"] is None
    assert ply_metadata["srs_in"] is None
    assert ply_metadata["point_count"] == expected_point_count
    assert np.all(ply_metadata["avg_min"] == expected_aabb[0])


def test_ply_run(filepath):
    ply_metadata = ply_reader.get_metadata(path=filepath)
    ply_filepath, portion = ply_metadata["portions"][0]
    assert 1 < compute_spacing(ply_metadata["aabb"] - ply_metadata["avg_min"]) < 10
    offset_scale = (
        tuple([-metadata for metadata in ply_metadata["avg_min"]]),
        np.array([0.1, 0.1, 0.1]),
        None,
        None,
    )
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(URI)
    ply_reader.run(ply_filepath, offset_scale, portion, socket, transformer=None)
    # assert socket?
    socket.close()
    context.destroy()
