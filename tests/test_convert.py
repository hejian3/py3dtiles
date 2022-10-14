import json
from pathlib import Path
import shutil
from unittest.mock import patch

import laspy
import numpy as np
from pyproj import CRS
from pytest import fixture, raises

from py3dtiles.convert import convert
from py3dtiles.exceptions import SrsInMissingException, SrsInMixinException
from py3dtiles.tileset.utils import number_of_points_in_tileset

DATA_DIRECTORY = Path(__file__).parent / 'fixtures'


@fixture()
def tmp_dir():
    yield Path('tmp/')
    shutil.rmtree('./tmp', ignore_errors=True)


def test_convert(tmp_dir):
    path = DATA_DIRECTORY / "ripple.las"
    convert(path, outfolder=tmp_dir)

    # basic asserts
    tileset_path = tmp_dir / 'tileset.json'
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [5.0, 5.0, 0.8593, 5.0, 0, 0, 0, 5.0, 0, 0, 0, 0.8593]
    box = [round(value, 4) for value in tileset['root']['boundingVolume']['box']]
    assert box == expecting_box

    assert Path(tmp_dir, 'r0.pnts').exists()

    with laspy.open(path) as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_without_srs(tmp_dir):
    with raises(SrsInMissingException, match="No input file has a srs. The srs_in should be provided as argument."):
        convert(DATA_DIRECTORY / 'without_srs.las',
                outfolder=tmp_dir,
                crs_out=CRS.from_epsg(4978),
                jobs=1)
    assert not tmp_dir.exists()

    convert(DATA_DIRECTORY / 'without_srs.las',
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3949),
            crs_out=CRS.from_epsg(4978),
            jobs=1)

    tileset_path = tmp_dir / 'tileset.json'
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [0.9662, 0.0008, 0.7066, 0.9662, 0, 0, 0, 0.0024, 0, 0, 0, 0.7066]
    box = [round(value, 4) for value in tileset['root']['boundingVolume']['box']]
    assert box == expecting_box

    assert Path(tmp_dir, 'r.pnts').exists()

    with laspy.open(DATA_DIRECTORY / 'without_srs.las') as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_with_srs(tmp_dir):
    convert(DATA_DIRECTORY / 'with_srs_3857.las',
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1)

    tileset_path = tmp_dir / 'tileset.json'
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [5.1633, 5.1834, 0.1731, 5.1631, 0, 0, 0, 5.1834, 0, 0, 0, 0.1867]
    box = [round(value, 4) for value in tileset['root']['boundingVolume']['box']]
    assert box == expecting_box

    assert Path(tmp_dir, 'r.pnts').exists()

    with laspy.open(DATA_DIRECTORY / 'with_srs_3857.las') as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_simple_xyz(tmp_dir):
    convert(DATA_DIRECTORY / 'simple.xyz',
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3857),
            crs_out=CRS.from_epsg(4978),
            jobs=1)
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r.pnts').exists()

    xyz_point_count = 0
    with open(DATA_DIRECTORY / 'simple.xyz') as f:
        line = True
        while line:
            line = f.readline()
            xyz_point_count += 1 if line else 0

    tileset_path = tmp_dir / 'tileset.json'
    assert xyz_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [0.3916, 0.3253, -0.0001, 0.39, 0, 0, 0, 0.3099, 0, 0, 0, 0.0001]
    box = [round(value, 4) for value in tileset['root']['boundingVolume']['box']]
    assert box == expecting_box


def test_convert_ply(tmp_dir):
    convert(DATA_DIRECTORY / 'simple.ply', outfolder=tmp_dir, jobs=1)
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r.pnts').exists()

    expected_point_count = 22300
    tileset_path = tmp_dir / 'tileset.json'
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [4.5437, 5.5984, 1.2002, 4.5437, 0, 0, 0, 5.5984, 0, 0, 0, 1.1681]
    box = [round(value, 4) for value in tileset['root']['boundingVolume']['box']]
    assert box == expecting_box


def test_convert_mix_las_xyz(tmp_dir):
    convert([DATA_DIRECTORY / 'simple.xyz', DATA_DIRECTORY / 'with_srs_3857.las'],
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1)
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r.pnts').exists()

    xyz_point_count = 0
    with open(DATA_DIRECTORY / 'simple.xyz') as f:
        line = True
        while line:
            line = f.readline()
            xyz_point_count += 1 if line else 0

    with laspy.open(DATA_DIRECTORY / 'with_srs_3857.las') as f:
        las_point_count = f.header.point_count

    tileset_path = tmp_dir / 'tileset.json'
    assert xyz_point_count + las_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [3416.2871, 5508.4194, -20921.0391, 44316.4531, 0, 0, 0, 14853.5332, 0, 0, 0, 1582.79]
    box = [round(value, 4) for value in tileset['root']['boundingVolume']['box']]
    assert box == expecting_box


def test_convert_mix_input_crs(tmp_dir):
    with raises(SrsInMixinException):
        convert([DATA_DIRECTORY / 'with_srs_3950.las', DATA_DIRECTORY / 'with_srs_3857.las'],
                outfolder=tmp_dir,
                crs_out=CRS.from_epsg(4978),
                jobs=1)
    assert not tmp_dir.exists()

    with raises(SrsInMixinException):
        convert([DATA_DIRECTORY / 'with_srs_3950.las', DATA_DIRECTORY / 'with_srs_3857.las'],
                outfolder=tmp_dir,
                crs_in=CRS.from_epsg(3432),
                crs_out=CRS.from_epsg(4978),
                jobs=1)
    assert not tmp_dir.exists()

    convert([DATA_DIRECTORY / 'with_srs_3950.las', DATA_DIRECTORY / 'with_srs_3857.las'],
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3432),
            crs_out=CRS.from_epsg(4978),
            force_crs_in=True,
            jobs=1)
    assert tmp_dir.exists()


def test_convert_xyz_missing_srs_in(tmp_dir):
    with raises(SrsInMissingException, match="No input file has a srs. The srs_in should be provided as argument."):
        convert(DATA_DIRECTORY / 'simple.xyz', outfolder=tmp_dir, crs_out='4978')


def test_convert_xyz_exception_in_run(tmp_dir):
    with patch('py3dtiles.reader.xyz_reader.run') as mock_run:
        with raises(Exception, match="An exception occurred in a worker: Exception in run"):
            mock_run.side_effect = Exception('Exception in run')
            convert(DATA_DIRECTORY / 'simple.xyz',
                    outfolder=tmp_dir,
                    crs_in=CRS.from_epsg(3857),
                    crs_out=CRS.from_epsg(4978))


def test_convert_las_exception_in_run(tmp_dir):
    with patch('py3dtiles.reader.las_reader.run') as mock_run:
        with raises(Exception, match="An exception occurred in a worker: Exception in run"):
            mock_run.side_effect = Exception('Exception in run')
            convert(DATA_DIRECTORY / 'with_srs_3857.las',
                    outfolder=tmp_dir,
                    crs_in=CRS.from_epsg(3857),
                    crs_out=CRS.from_epsg(4978))


def test_convert_export_folder_already_exists(tmp_dir):

    tmp_dir.mkdir()
    assert not (tmp_dir / 'tileset.json').exists()

    with raises(FileExistsError, match=f"Folder '{tmp_dir}' already exists"):
        convert(DATA_DIRECTORY / 'with_srs_3857.las',
                outfolder=tmp_dir,
                crs_out=CRS.from_epsg(4978),
                jobs=1)

    convert(DATA_DIRECTORY / 'with_srs_3857.las',
            outfolder=tmp_dir,
            overwrite=True,
            crs_out=CRS.from_epsg(4978),
            jobs=1)

    assert (tmp_dir / 'tileset.json').exists()


def test_convert_many_point_same_location(tmp_dir):
    tmp_dir.mkdir()

    # This is how the file has been generated.
    xyz_path = tmp_dir / 'pc_with_many_points_at_same_location.xyz'
    xyz_data = np.concatenate((np.random.random((10000, 3)), np.repeat([[0, 0, 0]], repeats=20000, axis=0)))
    with xyz_path.open('w') as f:
        np.savetxt(f, xyz_data, delimiter=" ", fmt='%.10f')

    convert(xyz_path, outfolder=tmp_dir / 'tiles')

    tileset_path = tmp_dir / 'tiles' / 'tileset.json'
    assert 30000 == number_of_points_in_tileset(tileset_path)
