import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import laspy
import numpy as np
from numpy.testing import assert_array_equal
import plyfile
from pyproj import CRS
from pytest import fixture, mark, raises

from py3dtiles.convert import convert
from py3dtiles.exceptions import SrsInMissingException, SrsInMixinException
from py3dtiles.reader.ply_reader import create_plydata_with_renamed_property
from py3dtiles.tileset import tile_content_reader
from py3dtiles.tileset.utils import number_of_points_in_tileset

DATA_DIRECTORY = Path(__file__).parent / "fixtures"


@fixture()
def tmp_dir():
    yield Path("tmp/")
    shutil.rmtree("./tmp", ignore_errors=True)


def test_convert(tmp_dir):
    path = DATA_DIRECTORY / "ripple.las"
    convert(path, outfolder=tmp_dir)

    # basic asserts
    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [5.0, 5.0, 0.8593, 5.0, 0, 0, 0, 5.0, 0, 0, 0, 0.8593]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box

    assert Path(tmp_dir, "r0.pnts").exists()

    with laspy.open(path) as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_with_prune(tmp_dir):
    # This file has 1 point at (-2, -2, -2) and 20001 at (1, 1, 1)
    # like this, it triggers the prune mechanism
    laz_path = DATA_DIRECTORY / "stacked_points.las"

    convert(
        laz_path,
        outfolder=tmp_dir,
        jobs=1,
        rgb=False,  # search bound cases by disabling rgb export
    )

    # basic asserts
    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [1.5, 1.5, 1.5, 1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box

    assert Path(tmp_dir, "r0.pnts").exists()

    with laspy.open(laz_path) as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_without_srs(tmp_dir):
    with raises(SrsInMissingException):
        convert(
            DATA_DIRECTORY / "without_srs.las",
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )
    assert not tmp_dir.exists()

    convert(
        DATA_DIRECTORY / "without_srs.las",
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(3949),
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )

    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [0.9662, 0.0008, 0.7066, 0.9662, 0, 0, 0, 0.0024, 0, 0, 0, 0.7066]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box

    assert Path(tmp_dir, "r.pnts").exists()

    with laspy.open(DATA_DIRECTORY / "without_srs.las") as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)

    tile1 = tile_content_reader.read_file(tmp_dir / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((187, 187, 187), dtype=np.uint8))


def test_convert_las_color_scale(tmp_dir):
    convert(
        DATA_DIRECTORY / "without_srs.las",
        outfolder=tmp_dir,
        jobs=1,
    )

    tile1 = tile_content_reader.read_file(tmp_dir / "r2.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((187, 187, 187), dtype=np.uint8))
    convert(
        DATA_DIRECTORY / "without_srs.las",
        overwrite=True,
        color_scale=1.1,
        outfolder=tmp_dir,
        jobs=1,
    )
    tile1 = tile_content_reader.read_file(tmp_dir / "r2.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # it should clamp to 255
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((206, 206, 206), dtype=np.uint8))

    convert(
        DATA_DIRECTORY / "without_srs.las",
        overwrite=True,
        color_scale=1.5,
        outfolder=tmp_dir,
        jobs=1,
    )
    tile1 = tile_content_reader.read_file(tmp_dir / "r2.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # it should clamp to 255
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((255, 255, 255), dtype=np.uint8))


def test_convert_with_srs(tmp_dir):
    convert(
        DATA_DIRECTORY / "with_srs_3857.las",
        outfolder=tmp_dir,
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )

    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [5.1633, 5.1834, 0.1731, 5.1631, 0, 0, 0, 5.1834, 0, 0, 0, 0.1867]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box

    assert Path(tmp_dir, "r.pnts").exists()

    with laspy.open(DATA_DIRECTORY / "with_srs_3857.las") as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_simple_xyz(tmp_dir):
    convert(
        DATA_DIRECTORY / "simple.xyz",
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(3857),
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    xyz_point_count = 0
    with open(DATA_DIRECTORY / "simple.xyz") as f:
        line = True
        while line:
            line = f.readline()
            xyz_point_count += 1 if line else 0

    tileset_path = tmp_dir / "tileset.json"
    assert xyz_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [0.3916, 0.3253, -0.0001, 0.39, 0, 0, 0, 0.3099, 0, 0, 0, 0.0001]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box


def test_convert_xyz_with_rgb(tmp_dir):
    convert(DATA_DIRECTORY / "simple_with_rgb.xyz", outfolder=tmp_dir)

    tile1 = tile_content_reader.read_file(tmp_dir / "r1.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((10, 0, 0), dtype=np.uint8))

    tile2 = tile_content_reader.read_file(tmp_dir / "r5.pnts")
    assert tile2.body.feature_table.nb_points() == 1
    pt2_color = tile2.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt2_color is None:
        raise RuntimeError("pt2_color shouldn't be None.")
    assert_array_equal(pt2_color, np.array((0, 0, 200), dtype=np.uint8))

    tile3 = tile_content_reader.read_file(tmp_dir / "r7.pnts")
    assert tile3.body.feature_table.nb_points() == 1
    pt3_color = tile3.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt3_color is None:
        raise RuntimeError("pt3_color shouldn't be None.")
    assert_array_equal(pt3_color, np.array((0, 10, 0), dtype=np.uint8))


def test_convert_xyz_with_rgb_color_scale(tmp_dir):
    convert(DATA_DIRECTORY / "simple_with_rgb.xyz", outfolder=tmp_dir, color_scale=1.5)

    tile1 = tile_content_reader.read_file(tmp_dir / "r1.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((15, 0, 0), dtype=np.uint8))

    tile2 = tile_content_reader.read_file(tmp_dir / "r5.pnts")
    assert tile2.body.feature_table.nb_points() == 1
    pt2_color = tile2.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt2_color is None:
        raise RuntimeError("pt2_color shouldn't be None.")
    assert_array_equal(pt2_color, np.array((0, 0, 255), dtype=np.uint8))

    tile3 = tile_content_reader.read_file(tmp_dir / "r7.pnts")
    assert tile3.body.feature_table.nb_points() == 1
    pt3_color = tile3.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt3_color is None:
        raise RuntimeError("pt3_color shouldn't be None.")
    assert_array_equal(pt3_color, np.array((0, 15, 0), dtype=np.uint8))


def test_convert_ply(tmp_dir):
    convert(DATA_DIRECTORY / "simple.ply", outfolder=tmp_dir, jobs=1)
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    expected_point_count = 22300
    tileset_path = tmp_dir / "tileset.json"
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [4.5437, 5.5984, 1.2002, 4.5437, 0, 0, 0, 5.5984, 0, 0, 0, 1.1681]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box

    tile1 = tile_content_reader.read_file(tmp_dir / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 5293
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 0), dtype=np.uint8))


def test_convert_ply_with_color(tmp_dir):
    # 8 bits color
    convert(DATA_DIRECTORY / "simple_with_8_bits_colors.ply", outfolder=tmp_dir, jobs=1)
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    expected_point_count = 4
    tileset_path = tmp_dir / "tileset.json"
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    tile1 = tile_content_reader.read_file(tmp_dir / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 128, 0), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r3.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((10, 0, 0), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r5.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 20), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r6.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((40, 40, 40), dtype=np.uint8))

    # 16 bits colors
    # every value should be divided by 256
    convert(
        DATA_DIRECTORY / "simple_with_16_bits_colors.ply",
        outfolder=tmp_dir,
        jobs=1,
        overwrite=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    expected_point_count = 4
    tileset_path = tmp_dir / "tileset.json"
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    tile1 = tile_content_reader.read_file(tmp_dir / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 0), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r3.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((1, 0, 0), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r5.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 4), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r6.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((255, 255, 255), dtype=np.uint8))


def test_convert_ply_with_color_scale(tmp_dir):
    # 8 bits color
    convert(
        DATA_DIRECTORY / "simple_with_8_bits_colors.ply",
        outfolder=tmp_dir,
        jobs=1,
        color_scale=3,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    tile1 = tile_content_reader.read_file(tmp_dir / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 255, 0), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r3.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((30, 0, 0), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r5.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 60), dtype=np.uint8))

    tile1 = tile_content_reader.read_file(tmp_dir / "r6.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((120, 120, 120), dtype=np.uint8))


def test_convert_ply_with_wrong_classification(tmp_dir):
    # Buggy feature name, classification is lost.
    convert(
        DATA_DIRECTORY / "simple.ply",
        outfolder=tmp_dir,
        jobs=1,
        classification=True,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    for py3dt_file in tmp_dir.iterdir():
        if py3dt_file.suffix != ".pnts":
            continue
        tile_content = tile_content_reader.read_file(py3dt_file)
        assert "Classification" in tile_content.body.batch_table.header.data
        assert np.array_equal(
            np.unique(tile_content.body.batch_table.body.data[0]),
            np.array([0], dtype=np.uint8),  # classification is lost
        )


def test_convert_ply_with_good_classification(tmp_dir):
    EXPECTED_LABELS = np.array([0, 1, 2, -1], dtype=np.uint8)
    # Change the classification property name in the tested .ply file
    ply_data = plyfile.PlyData.read(DATA_DIRECTORY / "simple.ply")
    ply_data = create_plydata_with_renamed_property(ply_data, "label", "classification")
    modified_ply_filename = DATA_DIRECTORY / "modified.ply"
    ply_data.write(modified_ply_filename)
    # Valid feature name, classification is preserved.
    convert(
        modified_ply_filename,
        outfolder=tmp_dir,
        jobs=1,
        classification=True,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    tileset_labels = np.array((), dtype=np.uint8)
    for py3dt_file in tmp_dir.iterdir():
        if py3dt_file.suffix != ".pnts":
            continue
        tile_content = tile_content_reader.read_file(py3dt_file)
        assert "Classification" in tile_content.body.batch_table.header.data
        pnts_labels = np.unique(tile_content.body.batch_table.body.data[0])
        # classification is OK for each pnts
        assert np.all(labels in EXPECTED_LABELS for labels in pnts_labels)
        tileset_labels = np.unique(np.append(tileset_labels, pnts_labels))
    # Every label is encountered in the global tileset
    assert np.array_equal(tileset_labels, EXPECTED_LABELS)
    # Clean the test directory
    modified_ply_filename.unlink()


def test_convert_mix_las_xyz(tmp_dir):
    convert(
        [DATA_DIRECTORY / "simple.xyz", DATA_DIRECTORY / "with_srs_3857.las"],
        outfolder=tmp_dir,
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "r.pnts").exists()

    xyz_point_count = 0
    with open(DATA_DIRECTORY / "simple.xyz") as f:
        line = True
        while line:
            line = f.readline()
            xyz_point_count += 1 if line else 0

    with laspy.open(DATA_DIRECTORY / "with_srs_3857.las") as f:
        las_point_count = f.header.point_count

    tileset_path = tmp_dir / "tileset.json"
    assert xyz_point_count + las_point_count == number_of_points_in_tileset(
        tileset_path
    )

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [
        3416.2871,
        5508.4194,
        -20921.0391,
        44316.4531,
        0,
        0,
        0,
        14853.5332,
        0,
        0,
        0,
        1582.79,
    ]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box


def test_convert_mix_input_crs(tmp_dir):
    with raises(SrsInMixinException):
        convert(
            [
                DATA_DIRECTORY / "with_srs_3950.las",
                DATA_DIRECTORY / "with_srs_3857.las",
            ],
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )
    assert not tmp_dir.exists()

    with raises(SrsInMixinException):
        convert(
            [
                DATA_DIRECTORY / "with_srs_3950.las",
                DATA_DIRECTORY / "with_srs_3857.las",
            ],
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3432),
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )
    assert not tmp_dir.exists()

    convert(
        [DATA_DIRECTORY / "with_srs_3950.las", DATA_DIRECTORY / "with_srs_3857.las"],
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(3432),
        crs_out=CRS.from_epsg(4978),
        force_crs_in=True,
        jobs=1,
    )
    assert tmp_dir.exists()


@mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="'patch' function works only with the multiprocessing 'fork' method (not available on windows).",
)
def test_convert_xyz_exception_in_run(tmp_dir):
    with patch("py3dtiles.reader.xyz_reader.run") as mock_run, raises(
        Exception, match="An exception occurred in a worker: Exception in run"
    ):
        mock_run.side_effect = Exception("Exception in run")
        convert(
            DATA_DIRECTORY / "simple.xyz",
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3857),
            crs_out=CRS.from_epsg(4978),
        )


@mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="'patch' function works only with the multiprocessing 'fork' method (not available on windows).",
)
def test_convert_las_exception_in_run(tmp_dir):
    with patch("py3dtiles.reader.las_reader.run") as mock_run, raises(
        Exception, match="An exception occurred in a worker: Exception in run"
    ):
        mock_run.side_effect = Exception("Exception in run")
        convert(
            DATA_DIRECTORY / "with_srs_3857.las",
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3857),
            crs_out=CRS.from_epsg(4978),
        )


def test_convert_export_folder_already_exists(tmp_dir):

    tmp_dir.mkdir()
    assert not (tmp_dir / "tileset.json").exists()

    with raises(FileExistsError, match=f"Folder '{tmp_dir}' already exists"):
        convert(
            DATA_DIRECTORY / "with_srs_3857.las",
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )

    convert(
        DATA_DIRECTORY / "with_srs_3857.las",
        outfolder=tmp_dir,
        overwrite=True,
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )

    assert (tmp_dir / "tileset.json").exists()


def test_convert_many_point_same_location(tmp_dir):
    tmp_dir.mkdir()

    # This is how the file has been generated.
    xyz_path = tmp_dir / "pc_with_many_points_at_same_location.xyz"
    xyz_data = np.concatenate(
        (np.random.random((10000, 3)), np.repeat([[0, 0, 0]], repeats=20000, axis=0))
    )
    with xyz_path.open("w") as f:
        np.savetxt(f, xyz_data, delimiter=" ", fmt="%.10f")

    convert(xyz_path, outfolder=tmp_dir / "tiles")

    tileset_path = tmp_dir / "tiles" / "tileset.json"
    assert number_of_points_in_tileset(tileset_path) == 30000
