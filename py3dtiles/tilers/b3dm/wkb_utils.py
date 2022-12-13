import math
import struct

from earcut.earcut import earcut
import numpy as np

class TriangleSoup:
    def __init__(self):
        self.triangles = []

    @staticmethod
    def from_wkb_multipolygon(wkb, associated_data=None):
        """
        Parameters
        ----------
        wkb : string
            Well-Known Binary binary string describing a multipolygon

        associated_data : array
            array of multipolygons containing data attached to the wkb
            parameter multipolygon. Must be the same size as wkb.

        Returns
        -------
        ts : TriangleSoup
        """
        multipolygons = [parse(bytes(wkb))]

        if associated_data is None:
            associated_data = []

        for additional_wkb in associated_data:
            multipolygons.append(parse(bytes(additional_wkb)))

        triangles_array = [[] for _ in range(len(multipolygons))]
        for i in range(0, len(multipolygons[0])):
            polygon = multipolygons[0][i]
            additional_polygons = [mp[i] for mp in multipolygons[1:]]
            triangles = triangulate(polygon, additional_polygons)
            for array, tri in zip(triangles_array, triangles):
                array += tri

        ts = TriangleSoup()
        ts.triangles = triangles_array

        return ts

    def get_position_array(self):
        """
        Parameters
        ----------

        Returns
        -------
        Binary array of vertex positions
        """

        vertex_triangles = self.triangles[0]
        vertex_array = vertex_attribute_to_array(vertex_triangles)
        return b''.join(vertex_array)

    def get_data_array(self, index):
        """
        Parameters
        ----------
        index: int
            The index of the associated data

        Returns
        -------
        Binary array of vertex data
        """

        vertex_triangles = self.triangles[1 + index]
        vertex_array = vertex_attribute_to_array(vertex_triangles)
        return b''.join(vertex_array)

    def get_normal_array(self):
        """
        Parameters
        ----------

        Returns
        -------
        Binary array of vertice normals
        """
        normals = []
        for t in self.triangles[0]:
            U = t[1] - t[0]
            V = t[2] - t[0]
            N = np.cross(U, V)
            norm = np.linalg.norm(N)
            if norm == 0:
                normals.append(np.array([0, 0, 1], dtype=np.float32))
            else:
                normals.append(N / norm)

        vertex_array = face_attribute_to_array(normals)
        return b''.join(vertex_array)

    def get_bbox(self):
        """
        Parameters
        ----------

        Returns
        -------
        Array [[minX, minY, minZ],[maxX, maxY, maxZ]]
        """
        mins = np.array([np.min(t, 0) for t in self.triangles[0]])
        maxs = np.array([np.max(t, 0) for t in self.triangles[0]])
        return [np.min(mins, 0), np.max(maxs, 0)]


def face_attribute_to_array(triangles):
    array = []
    for face in triangles:
        array += [face, face, face]
    return array


def vertex_attribute_to_array(triangles):
    array = []
    for face in triangles:
        for vertex in face:
            array.append(vertex)
    return array


def parse(wkb):
    multipolygon = []
    byteorder = struct.unpack('b', wkb[0:1])
    bo = '<' if byteorder[0] else '>'
    geomtype = struct.unpack(bo + 'I', wkb[1:5])[0]
    has_z = (geomtype == 1006) or (geomtype == 1015)
    # MultipolygonZ or polyhedralSurface
    pnt_offset = 24 if has_z else 16
    pnt_unpack = 'ddd' if has_z else 'dd'
    geom_nb = struct.unpack(bo + 'I', wkb[5:9])[0]
    # print(struct.unpack('b', wkb[9:10])[0])
    # print(struct.unpack('I', wkb[10:14])[0])   # 1003 (Polygon)
    # print(struct.unpack('I', wkb[14:18])[0])   # num lines
    # print(struct.unpack('I', wkb[18:22])[0])   # num points
    offset = 9
    for _ in range(0, geom_nb):
        offset += 5  # struct.unpack('bI', wkb[offset:offset + 5])[0]
        # 1 (byteorder), 1003 (Polygon)
        line_nb = struct.unpack(bo + 'I', wkb[offset:offset + 4])[0]
        offset += 4
        polygon = []
        for _ in range(0, line_nb):
            point_nb = struct.unpack(bo + 'I', wkb[offset:offset + 4])[0]
            offset += 4
            line = []
            for _ in range(0, point_nb - 1):
                pt = np.array(
                    struct.unpack(bo + pnt_unpack, wkb[offset:offset + pnt_offset]),
                    dtype=np.float32
                )
                offset += pnt_offset
                line.append(pt)
            offset += pnt_offset  # skip redundant point
            polygon.append(line)
        multipolygon.append(polygon)
    return multipolygon


def triangulate(polygon, additional_polygons=None):
    """
    Triangulates 3D polygons
    """
    # let's find out if the polygon is *mostly* clockwise or counter-clockwise
    # and triangulate accordingly
    # for 2D explanations:
    # https://stackoverflow.com/a/1165943/1528985
    # and https://www.element84.com/blog/determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
    #
    # Quick explanation in case it goes down: for each edge we calculate the
    # area of the polygon formed by this edge, the x axis and the 2 vertical.
    # It's (x2-x1) / ((y2+y1 / 2) (draw it if you don't believe me). This
    # results will be positive for a edge that goes toward positive x.  Summing
    # all these areas will give plus or minus the total polygon area.  it would
    # be positive for a clockwise polygon (upper edges contributing positively)
    # and negative for counter-clockwise polygons (upper edges contributing
    # negatively)
    #
    # Adaptations here:
    # - we prefer to reason with counter-clockwise positive, hence the x1-x2 instead of x2-x1
    # - in 3D, we calcule this value for each axis planes (xy, yz, zx),
    # looking in the other axis negative direction.
    # - comparing these 3 results actually give us the most interesting plane
    # to triangulate in (the plane were the projected area is the biggest)
    # - we drop the 1/2 factor because we are only interesting in the sign and relative comparison
    vect_prod = np.array([0, 0, 0], dtype=np.float32)
    for i in range(len(polygon[0])):
        curr_edge = polygon[0][i]
        next_edge = polygon[0][(i + 1) % len(polygon[0])]
        vect_prod += np.array([
            # yz plane, seen from negative x
            (curr_edge[1] - next_edge[1]) * (next_edge[2] + curr_edge[2]),
            # zx plane, seen from negative y
            (curr_edge[2] - next_edge[2]) * (next_edge[0] + curr_edge[0]),
            # xy plane, seen from negative z
            (curr_edge[0] - next_edge[0]) * (next_edge[1] + curr_edge[1]),
        ], dtype=np.float32)

    if additional_polygons is None:
        additional_polygons = []

    polygon_2d = []
    holes = []
    delta = 0
    for p in polygon[:-1]:
        holes.append(delta + len(p))
        delta += len(p)
    # triangulation of the polygon projected on planes (xy) (zx) or (yz)
    if (math.fabs(vect_prod[0]) > math.fabs(vect_prod[1])
       and math.fabs(vect_prod[0]) > math.fabs(vect_prod[2])):
        # (yz) projection
        for linestring in polygon:
            for point in linestring:
                polygon_2d.extend([point[1], point[2]])
    elif math.fabs(vect_prod[1]) > math.fabs(vect_prod[2]):
        # (zx) projection
        for linestring in polygon:
            for point in linestring:
                polygon_2d.extend([point[0], point[2]])
    else:
        # (xy) projection
        for linestring in polygon:
            for point in linestring:
                polygon_2d.extend([point[0], point[1]])

    triangles_idx = earcut(polygon_2d, holes, 2)

    arrays = [[] for _ in range(len(additional_polygons) + 1)]
    for i in range(0, len(triangles_idx), 3):
        t = triangles_idx[i:i + 3]
        p0 = unflatten(polygon, holes, t[0])
        p1 = unflatten(polygon, holes, t[1])
        p2 = unflatten(polygon, holes, t[2])
        # triangulation may break triangle orientation, test it before
        # adding triangles
        # FIXME fix / change the triangulation code instead?
        cross_product = np.cross(p1 - p0, p2 - p0)
        invert = np.dot(vect_prod, cross_product) < 0
        if invert:
            arrays[0].append([p1, p0, p2])
        else:
            arrays[0].append([p0, p1, p2])
        for array, p in zip(arrays[1:], additional_polygons):
            pp0 = unflatten(p, holes, t[0])
            pp1 = unflatten(p, holes, t[1])
            pp2 = unflatten(p, holes, t[2])
            if invert:
                array.append([pp1, pp0, pp2])
            else:
                array.append([pp0, pp1, pp2])

    return arrays


def unflatten(array, lengths, index):
    for i in reversed(range(0, len(lengths))):
        lgth = lengths[i]
        if index >= lgth:
            return array[i + 1][index - lgth]
    return array[0][index]
