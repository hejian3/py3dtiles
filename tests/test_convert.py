from pathlib import Path
import shutil
from unittest.mock import patch

from pytest import approx, fixture, raises

from py3dtiles import convert_to_ecef
from py3dtiles.convert import convert, SrsInMissingException

DATA_DIRECTORY = Path(__file__).parent / 'fixtures'


@fixture()
def tmp_dir():
    yield './tmp'
    shutil.rmtree('./tmp', ignore_errors=True)


def test_convert_to_ecef():
    # results tested with gdaltransform
    [x, y, z] = convert_to_ecef(-75.61200462622627,
                                40.03886513981721,
                                2848.448771114095,
                                4326)
    approx(x, 1215626.30684538)
    approx(y, -4738673.45914053)
    approx(z, 4083122.83975827)


def test_convert(tmp_dir):
    path = DATA_DIRECTORY / "ripple.las"
    convert(str(path), outfolder=tmp_dir)

    # basic asserts
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r0.pnts').exists()

    # test


def test_convert_without_srs(tmp_dir):
    with raises(SrsInMissingException):
        convert(str(DATA_DIRECTORY / 'without_srs.las'),
                outfolder=tmp_dir,
                srs_out='4978',
                jobs=1)
    assert not Path(tmp_dir).exists()

    convert(str(DATA_DIRECTORY / 'without_srs.las'),
            outfolder=tmp_dir,
            srs_in='3949',
            srs_out='4978',
            jobs=1)
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r.pnts').exists()


def test_convert_with_srs(tmp_dir):
    convert(str(DATA_DIRECTORY / 'with_srs.las'),
            outfolder=tmp_dir,
            srs_out='4978',
            jobs=1)
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r.pnts').exists()


def test_convert_simple_xyz(tmp_dir):
    convert(str(DATA_DIRECTORY / 'simple.xyz'),
            outfolder=tmp_dir,
            srs_in='3857',
            srs_out='4978',
            jobs=1)
    assert Path(tmp_dir, 'tileset.json').exists()
    assert Path(tmp_dir, 'r.pnts').exists()


def test_convert_xyz_exception_in_run(tmp_dir):
    with patch('py3dtiles.points.task.xyz_reader.run') as mock_run:
        with raises(Exception, match="An exception occurred in a worker: Exception in run"):
            mock_run.side_effect = Exception('Exception in run')
            convert(str(DATA_DIRECTORY / 'simple.xyz'),
                    outfolder=tmp_dir,
                    srs_in='3857',
                    srs_out='4978')


def test_convert_las_exception_in_run(tmp_dir):
    with patch('py3dtiles.points.task.las_reader.run') as mock_run:
        with raises(Exception, match="An exception occurred in a worker: Exception in run"):
            mock_run.side_effect = Exception('Exception in run')
            convert(str(DATA_DIRECTORY / 'with_srs.las'),
                    outfolder=tmp_dir,
                    srs_in='3857',
                    srs_out='4978')


def test_convert_export_folder_already_exists(tmp_dir):
    tmp_dir = Path(tmp_dir)

    tmp_dir.mkdir()
    assert not (tmp_dir / 'tileset.json').exists()

    with raises(FileExistsError, match=f"Folder '{tmp_dir}' already exists"):
        convert(str(DATA_DIRECTORY / 'with_srs.las'),
                outfolder=tmp_dir,
                srs_out='4978',
                jobs=1)

    convert(str(DATA_DIRECTORY / 'with_srs.las'),
            outfolder=tmp_dir,
            overwrite=True,
            srs_out='4978',
            jobs=1)

    assert (tmp_dir / 'tileset.json').exists()
