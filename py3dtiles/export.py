import errno
import getpass
import json
import math
import os

import numpy as np
import psycopg2

from py3dtiles.tilers.b3dm.wkb_utils import TriangleSoup
from py3dtiles.tileset.batch_table import BatchTable
from py3dtiles.tileset.content import B3dm, GlTF


class BoundingBox:
    def __init__(self, minimum, maximum):
        self.min = [float(i) for i in minimum]
        self.max = [float(i) for i in maximum]

    def inside(self, point):
        return ((self.min[0] <= point[0] < self.max[0])
                and (self.min[1] <= point[1] < self.max[1]))

    def center(self):
        return [(i + j) / 2 for (i, j) in zip(self.min, self.max)]

    def add(self, box):
        self.min = [min(i, j) for (i, j) in zip(self.min, box.min)]
        self.max = [max(i, j) for (i, j) in zip(self.max, box.max)]


class Feature():
    def __init__(self, index, box):
        self.index = index
        self.box = box


class Node:
    counter = 0

    def __init__(self, features=None):
        self.id = Node.counter
        Node.counter += 1
        self.features = features if features else []
        self.box = None
        self.children = []

    def add(self, node):
        self.children.append(node)

    def compute_bbox(self):
        self.box = BoundingBox(
            [float("inf"), float("inf"), float("inf")],
            [-float("inf"), -float("inf"), -float("inf")])
        for c in self.children:
            c.compute_bbox()
            self.box.add(c.box)
        for g in self.features:
            self.box.add(g.box)

    def to_tileset(self, transform):
        self.compute_bbox()
        tiles = {
            "asset": {"version": "1.0"},
            "geometricError": 500,  # TODO
            "root": self.to_tileset_r(500)
        }
        tiles["root"]["transform"] = [round(float(e), 3) for e in transform]
        return tiles

    def to_tileset_r(self, error):
        (c1, c2) = (self.box.min, self.box.max)
        center = [(c1[i] + c2[i]) / 2 for i in range(0, 3)]
        x_axis = [(c2[0] - c1[0]) / 2, 0, 0]
        y_axis = [0, (c2[1] - c1[1]) / 2, 0]
        z_axis = [0, 0, (c2[2] - c1[2]) / 2]
        box = [round(x, 3) for x in center + x_axis + y_axis + z_axis]
        tile = {
            "boundingVolume": {
                "box": box
            },
            "geometricError": error,  # TODO
            "children": [n.to_tileset_r(error / 2.) for n in self.children],
            "refine": "add"
        }
        if len(self.features) != 0:
            tile["content"] = {
                "uri": f"tiles/{self.id}.b3dm"
            }

        return tile

    def all_nodes(self):
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes


def tile_extent(extent, size, i, j):
    min_extent = [
        extent.min[0] + i * size,
        extent.min[1] + j * size]
    max_extent = [
        extent.min[0] + (i + 1) * size,
        extent.min[1] + (j + 1) * size]
    return BoundingBox(min_extent, max_extent)


# TODO: transform
def arrays2tileset(positions, normals, bboxes, transform, ids=None):
    print("Creating tileset...")
    max_tile_size = 2000
    features_per_tile = 20
    indices = [i for i in range(len(positions))]

    # glTF is Y-up, so to get the bounding boxes in the 3D tiles
    # coordinate system, we have to apply a Y-to-Z transform to the
    # glTF bounding boxes
    z_up_bboxes = []
    for bbox in bboxes:
        tmp = m = bbox[0]
        M = bbox[1]
        m = [m[0], -m[2], m[1]]
        M = [M[0], -tmp[2], M[1]]
        z_up_bboxes.append([m, M])

    # Compute extent
    x_min = y_min = float('inf')
    x_max = y_max = - float('inf')

    for bbox in z_up_bboxes:
        x_min = min(x_min, bbox[0][0])
        y_min = min(y_min, bbox[0][1])
        x_max = max(x_max, bbox[1][0])
        y_max = max(y_max, bbox[1][1])
    extent = BoundingBox([x_min, y_min], [x_max, y_max])
    extent_x = x_max - x_min
    extent_y = y_max - y_min

    # Create quadtree
    tree = Node()
    for i in range(0, int(math.ceil(extent_x / max_tile_size))):
        for j in range(0, int(math.ceil(extent_y / max_tile_size))):
            tile = tile_extent(extent, max_tile_size, i, j)

            geoms = []
            for idx, box in zip(indices, z_up_bboxes):
                bbox = BoundingBox(box[0], box[1])

                if tile.inside(bbox.center()):
                    geoms.append(Feature(idx, bbox))

            if len(geoms) == 0:
                continue

            if len(geoms) > features_per_tile:
                node = Node(geoms[0:features_per_tile])
                tree.add(node)
                divide(tile, geoms[features_per_tile:len(geoms)], i * 2,
                       j * 2, max_tile_size / 2., features_per_tile, node)
            else:
                node = Node(geoms)
                tree.add(node)

    # Export b3dm & tileset
    tileset = tree.to_tileset(transform)
    f = open("tileset.json", 'w')
    f.write(json.dumps(tileset))
    print("Creating tiles...")
    nodes = tree.all_nodes()
    identity = np.identity(4).flatten('F')
    try:
        os.makedirs("tiles")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for node in nodes:
        if len(node.features) != 0:
            bin_arrays = []
            gids = []
            for feature in node.features:
                pos = feature.index
                bin_arrays.append({
                    'position': positions[pos],
                    'normal': normals[pos],
                    'bbox': [[float(i) for i in j] for j in bboxes[pos]],
                })
                if ids is not None:
                    gids.append(ids[pos])
            gltf = GlTF.from_binary_arrays(bin_arrays, identity)
            bt = None
            if ids is not None:
                bt = BatchTable()
                bt.add_property_from_array("id", gids)
            b3dm = B3dm.from_glTF(gltf, bt).to_array()
            f = open(f"tiles/{node.id}.b3dm", 'wb')
            f.write(b3dm)


def divide(extent, geometries, x_offset, y_offset, tile_size,
           features_per_tile, parent):
    for i in range(0, 2):
        for j in range(0, 2):
            tile = tile_extent(extent, tile_size, i, j)

            geoms = []
            for g in geometries:
                if tile.inside(g.box.center()):
                    geoms.append(g)
            if len(geoms) == 0:
                continue

            if len(geoms) > features_per_tile:
                node = Node(geoms[0:features_per_tile])
                parent.add(node)
                divide(tile, geoms[features_per_tile:len(geoms)],
                       (x_offset + i) * 2, (y_offset + j) * 2,
                       tile_size / 2., features_per_tile, node)
            else:
                node = Node(geoms)
                parent.add(node)


def wkbs_to_tileset(wkbs, ids, transform):
    geoms = [TriangleSoup.from_wkb_multipolygon(wkb) for wkb in wkbs]
    positions = [ts.get_position_array() for ts in geoms]
    normals = [ts.get_normal_array() for ts in geoms]
    bboxes = [ts.get_bbox() for ts in geoms]
    arrays2tileset(positions, normals, bboxes, transform, ids)


def build_secure_conn(db_conn_info):
    """Get a psycopg2 connexion securely, e.g. without writing the password explicitely
    in the terminal

    Parameters
    ----------
    db_conn_info : str

    Returns
    -------
    psycopg2.extensions.connection
    """
    try:
        connection = psycopg2.connect(db_conn_info)
    except psycopg2.OperationalError:
        pw = getpass.getpass("Postgres password: ")
        connection = psycopg2.connect(db_conn_info + f" password={pw}")
    return connection


def from_db(db_conn_info, table_name, column_name, id_column_name):
    connection = build_secure_conn(db_conn_info)
    cur = connection.cursor()

    print("Loading data from database...")
    cur.execute(f"SELECT ST_3DExtent({column_name}) FROM {table_name}")
    extent = cur.fetchall()[0][0]
    extent = [m.split(" ") for m in extent[6:-1].split(",")]
    offset = [(float(extent[1][0]) + float(extent[0][0])) / 2,
              (float(extent[1][1]) + float(extent[0][1])) / 2,
              (float(extent[1][2]) + float(extent[0][2])) / 2]

    id_statement = ""
    if id_column_name is not None:
        id_statement = "," + id_column_name
    cur.execute("SELECT ST_AsBinary(ST_RotateX(ST_Translate({0}, {1}, {2}, {3}), -pi() / 2)),"
                "ST_Area(ST_Force2D({0})) AS weight{5} FROM {4} ORDER BY weight DESC"
                .format(column_name, -offset[0], -offset[1], -offset[2],
                        table_name, id_statement))
    res = cur.fetchall()
    wkbs = [t[0] for t in res]
    ids = None
    if id_column_name is not None:
        ids = [t[2] for t in res]
    transform = np.array([
        [1, 0, 0, offset[0]],
        [0, 1, 0, offset[1]],
        [0, 0, 1, offset[2]],
        [0, 0, 0, 1]], dtype=float)
    transform = transform.flatten('F')

    wkbs_to_tileset(wkbs, ids, transform)


def from_directory(directory, offset):
    # TODO: improvement -> order wkbs by geometry size, similarly to database mode
    offset = (0, 0, 0) if offset is None else offset
    # open all wkbs from directory
    files = os.listdir(directory)
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1] == '.wkb']
    wkbs = []
    for f in files:
        of = open(f, 'rb')
        wkbs.append(of.read())
        of.close()

    transform = np.array([
        [1, 0, 0, offset[0]],
        [0, 1, 0, offset[1]],
        [0, 0, 1, offset[2]],
        [0, 0, 0, 1]], dtype=float)
    transform = transform.flatten('F')
    wkbs_to_tileset(wkbs, None, transform)


def init_parser(subparser):
    descr = "Generate a tileset from a set of geometries"
    parser = subparser.add_parser('export', help=descr)

    group = parser.add_mutually_exclusive_group()

    d_help = "Name of the directory containing the geometries"
    group.add_argument('-d', metavar='DIRECTORY', type=str, help=d_help)

    o_help = "Offset of the geometries (only with '-d')"
    parser.add_argument('-o', nargs=3, metavar=('X', 'Y', 'Z'), type=float, help=o_help)

    D_help = """
    Database connexion info (e.g. 'service=py3dtiles' or \
    'dbname=py3dtiles host=localhost port=5432 user=yourname password=yourpwd')
    """
    group.add_argument('-D', metavar='DB_CONNINFO', type=str, help=D_help)

    t_help = "Table name (required if '-D' option is activated)"
    parser.add_argument('-t', metavar='TABLE', type=str, help=t_help)

    c_help = "Geometry column name (required if '-D' option is activated)"
    parser.add_argument('-c', metavar='COLUMN', type=str, help=c_help)

    i_help = "Id column name (only with '-D')"
    parser.add_argument('-i', metavar='IDCOLUMN', type=str, help=i_help)

    return parser


def main(args):
    if args.D is not None:
        if args.t is None or args.c is None:
            print('Error: please define a table (-t) and column (-c)')
            exit()

        from_db(args.D, args.t, args.c, args.i)
    elif args.d is not None:
        from_directory(args.d, args.o)
    else:
        raise NameError('Error: database or directory must be set')
