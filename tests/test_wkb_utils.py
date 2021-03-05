#!/usr/bin/env python
# -*- coding: utf-8 -*-
from py3dtiles import wkb_utils
import pytest
import numpy as np

def test_triangulate_winding_order_simple():
    # simple case: a square on xy plane, counter-clockwise
    polygon = [
        [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32),
            np.array([1, 1, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32)
        ]
    ]

    triangles = wkb_utils.triangulate(polygon)
    assert len(triangles[0]) == 2, 'Should generate 2 triangles'
    assert \
        all(np.cross(
            triangles[0][0][1] - triangles[0][0][0],
            triangles[0][0][2] - triangles[0][0][0]) == np.array([0, 0, 1], dtype=np.float32)), \
        'Check winding order is coherent with vertex order: counter-clockwise (triangle 1)'
    assert \
        all(np.cross(
            triangles[0][1][1] - triangles[0][1][0],
            triangles[0][1][2] - triangles[0][1][0]) == np.array([0, 0, 1], dtype=np.float32)), \
        'Check winding order is coherent with vertex order: counter-clockwise (triangle 2)'

    # simple case 2: a square on xy plane, clockwise
    polygon = [
        [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([1, 1, 0], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32)
        ]
    ]
    triangles = wkb_utils.triangulate(polygon)
    assert len(triangles[0]) == 2, 'Should generate 2 triangles'
    assert \
        all(np.cross(
            triangles[0][0][1] - triangles[0][0][0],
            triangles[0][0][2] - triangles[0][0][0]) == np.array([0, 0, -1], dtype=np.float32)), \
        'Check winding order is coherent with vertex order: clockwise (triangle1)'
    assert \
        all(np.cross(
            triangles[0][1][1] - triangles[0][1][0],
            triangles[0][1][2] - triangles[0][1][0]) == np.array([0, 0, -1], dtype=np.float32)), \
        'Check winding order is coherent with vertex order: clockwise (triangle2)'

@pytest.fixture
def complex_polygon():
    # tricky polygon 1:
    # 0x---------x 4
    #   \        |
    #    \       |
    #   1 x      |
    #    /       |
    #   /        |
    # 2x---------x 3
    # the first few vertices seems to indicate an inverse winding order
    return [ 
        [
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0.5, 0.5, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32),
            np.array([1, 1, 0], dtype=np.float32)
        ]
    ]


def test_triangulate_winding_order_complex(complex_polygon):
    triangles = wkb_utils.triangulate(complex_polygon)
    assert len(triangles[0]) == 3, 'Should generate 2 triangles'
    crossprod_triangle1 = np.cross(
                    triangles[0][0][1] - triangles[0][0][0],
                    triangles[0][0][2] - triangles[0][0][0])
    crossprod_triangle1 /= np.linalg.norm(crossprod_triangle1)
    assert \
        all(crossprod_triangle1 == np.array([0, 0, 1], dtype=np.float32)), \
        'Check winding order is coherent with vertex order: counter-clockwise'


def test_benchmark_triangulate(complex_polygon, benchmark):
    benchmark(wkb_utils.triangulate, complex_polygon)
