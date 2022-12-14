from py3dtiles.node import Node


class B3dmNode(Node):
    counter = 0

    def __init__(self, features=None):
        self.id = B3dmNode.counter
        B3dmNode.counter += 1
        self.features = features if features else []
        self.box = None
        self.children = []

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
            "geometricError": 5000,  # TODO
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
                "uri": f"b3dm_tiles/{self.id}.b3dm"
            }

        return tile

    def all_nodes(self):
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes


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
