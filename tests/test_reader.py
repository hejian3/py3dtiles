"""Test the point cloud readers.

The example that is run in the test (`simple.ply`) comes from the [CGAL repository](https://github.com/CGAL/cgal/blob/master/Data/data/points_3/b9_training.ply). Thanks to their maintainers (for more details, please refer to CGAL, Computational Geometry Algorithms Library, https://www.cgal.org):

"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import plyfile
from pytest import fixture, raises
import zmq

from py3dtiles.convert import URI
from py3dtiles.reader import ply_reader
from py3dtiles.utils import compute_spacing


DATA_DIRECTORY = Path(__file__).parent / 'fixtures'


@fixture
def ply_filepath():
    yield DATA_DIRECTORY / "simple.ply"


@fixture
def buggy_ply_filepath():
    yield DATA_DIRECTORY / "buggy.ply"


@fixture(params=["wrongname", "vertex"])
def buggy_ply_data(request):
    """This ply data does not contain any 'vertex' element!"""
    types = [('x', np.float32, (5,)), ('y', np.float32, (5,)), ('z', np.float32, (5,))]
    data = [(np.random.sample(5), np.random.sample(5), np.random.sample(5))]
    arr = np.array(
        data if request.param == "wrongname" else [data[0][:2]],
        dtype=np.dtype(types) if request.param == "wrongname" else np.dtype(types[:2]),
    )
    ply_item = plyfile.PlyElement.describe(data=arr, name=request.param)
    ply_data = plyfile.PlyData(elements=[ply_item])
    yield {
        "data": ply_data,
        "msg": "vertex" if request.param == "wrongname" else "x, y, z"
    }


def test_ply_get_metadata(ply_filepath):
    ply_metadata = ply_reader.get_metadata(path=ply_filepath)
    expected_point_count = 22300
    expected_aabb = (
        np.array([5.966480625e+05, 2.43620015625e+05, 7.350153350830078e+01]),
        np.array([5.967389375e+05, 2.43731984375e+05, 9.718580627441406e+01]),
    )
    assert list(ply_metadata.keys()) == [
        "portions", "aabb", "color_scale", "srs_in", "point_count", "avg_min"
    ]
    assert ply_metadata["portions"] == [
        (str(ply_filepath), (0, expected_point_count, expected_point_count))
    ]
    assert np.all(ply_metadata["aabb"][0] == expected_aabb[0])
    assert np.all(ply_metadata["aabb"][1] == expected_aabb[1])
    assert ply_metadata["color_scale"] is None
    assert ply_metadata["srs_in"] is None
    assert ply_metadata["point_count"] == expected_point_count
    assert np.all(ply_metadata["avg_min"] == expected_aabb[0])


def test_ply_get_metadata_buggy(buggy_ply_data, buggy_ply_filepath):
    buggy_ply_data["data"].write(buggy_ply_filepath)
    with raises(KeyError, match=buggy_ply_data["msg"]):
        _ = ply_reader.get_metadata(path=buggy_ply_filepath)
    buggy_ply_filepath.unlink()
