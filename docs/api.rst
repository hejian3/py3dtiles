.. _api:

API usage
---------

Generic Tile
~~~~~~~~~~~~

The py3dtiles module provides some classes to fit into the
specification:

- *TileContent* with a header *TileContentHeader* and a body *TileContentBody*
- *TileContentHeader* represents the metadata of the tile (magic value, version, ...)
- *TileContentBody* contains varying semantic and geometric data depending on the the tile's type

Moreover, a utility module *tile_content_reader.py* provides a function *read_file* to read a tile
file as well as a simple command line tool to retrieve basic information about a tile:
**py3dtiles\_info**. We also provide a utility to generate a tileset from a list of 3D models in
WKB format or stored in a postGIS table.


Point Cloud
~~~~~~~~~~~

Points Tile Format:
https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/specification/TileFormats/PointCloud

In the current implementation, the *Pnts* class only contains a *FeatureTable*
(*FeatureTableHeader* and a *FeatureTableBody*, which contains features of type
*Feature*).

**How to read a .pnts file**

.. code-block:: python

    >>> from pathlib import Path
    >>>
    >>> from py3dtiles.tileset.content import Pnts
    >>> from py3dtiles.tileset import tile_content_reader
    >>>
    >>> filename = Path('tests/fixtures/pointCloudRGB.pnts')
    >>>
    >>> # read the file
    >>> pnts = tile_content_reader.read_file(filename)
    >>>
    >>> # pnts is an instance of the Pnts class
    >>> pnts
    <py3dtiles.tileset.content.pnts.Pnts object at 0x...>
    >>>
    >>> # extract information about the pnts header
    >>> pnts_header = pnts.header
    >>> pnts_header
    <py3dtiles.tileset.content.pnts.PntsHeader object at 0x...>
    >>> pnts_header.magic_value
    b'pnts'
    >>> pnts_header.tile_byte_length
    15176
    >>>
    >>> # extract the feature table
    >>> feature_table = pnts.body.feature_table
    >>> feature_table
    <py3dtiles.tileset.feature_table.FeatureTable object at 0x...>
    >>>
    >>> # display feature table header
    >>> feature_table.header.to_json()
    {'POINTS_LENGTH': 1000, 'RTC_CENTER': [1215012.8828876738, -4736313.051199594, 4081605.22126042], 'POSITION': {'byteOffset': 0}, 'RGB': {'byteOffset': 12000}}
    >>>
    >>> # extract positions and colors of the first point
    >>> feature = feature_table.feature(0)
    >>> feature
    <py3dtiles.tileset.feature_table.Feature object at 0x...>
    >>> feature.positions
    {'X': 2.19396, 'Y': 4.489685, 'Z': -0.17107764}
    >>> feature.colors
    {'Red': 44, 'Green': 243, 'Blue': 209}

**How to write a .pnts file**

To write a Point Cloud file, you have to build a numpy array with the
corresponding data type.

.. code-block:: python

    >>> from pathlib import Path
    >>>
    >>> import numpy as np
    >>>
    >>> from py3dtiles.tileset.content import Pnts
    >>> from py3dtiles.tileset.feature_table import Feature
    >>>
    >>> # create the numpy dtype for positions with 32-bit floating point numbers
    >>> dt = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
    >>>
    >>> # create a position array
    >>> position = np.array([(4.489, 2.19, -0.17)], dtype=dt)
    >>>
    >>> # create a new feature from a uint8 numpy array
    >>> feature = Feature.from_array(dt, position.view('uint8'))
    >>> feature
    <py3dtiles.tileset.feature_table.Feature object at 0x...>
    >>> feature.positions
    {'X': 4.489, 'Y': 2.19, 'Z': -0.17}
    >>>
    >>> # create a pnts directly from our feature. None is for "no colors".
    >>> pnts  = Pnts.from_features(dt, None, [feature])
    >>>
    >>> # the pnts is complete
    >>> pnts.body.feature_table.header.to_json()
    {'POINTS_LENGTH': 1, 'POSITION': {'byteOffset': 0}}
    >>>
    >>> # to save our tile as a .pnts file
    >>> pnts.save_as(Path("mypoints.pnts"))


Batched 3D Model
~~~~~~~~~~~~~~~~

Batched 3D Model Tile Format:
https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/TileFormats/Batched3DModel

**How to read a .b3dm file**

.. code-block:: python

    >>> from pathlib import Path
    >>>
    >>> from py3dtiles.tileset.content import B3dm
    >>> from py3dtiles.tileset import tile_content_reader
    >>>
    >>> filename = Path('tests/fixtures/dragon_low.b3dm')
    >>>
    >>> # read the file
    >>> b3dm = tile_content_reader.read_file(filename)
    >>>
    >>> # b3dm is an instance of the B3dm class
    >>> b3dm
    <py3dtiles.tileset.content.b3dm.B3dm object at 0x...>
    >>>
    >>> # extract information about the b3dm header
    >>> b3dm_header = b3dm.header
    >>> b3dm_header
    <py3dtiles.tileset.content.b3dm.B3dmHeader object at 0x...>
    >>> b3dm_header.magic_value
    b'b3dm'
    >>> b3dm_header.tile_byte_length
    47246
    >>>
    >>> # extract the glTF
    >>> gltf = b3dm.body.gltf
    >>> gltf
    <py3dtiles.tileset.content.gltf.GlTF object at 0x...>
    >>>
    >>> # display gltf header's asset field
    >>> gltf.header['asset']
    {'generator': 'OBJ2GLTF', 'premultipliedAlpha': True, 'profile': {'api': 'WebGL', 'version': '1.0'}, 'version': '1.0'}

**How to write a .b3dm file**

To write a Batched 3D Model file, you have to import the geometry from a wkb
file containing polyhedralsurfaces or multipolygons.

.. code-block:: python

    >>> from pathlib import Path
    >>>
    >>> import numpy as np
    >>>
    >>> from py3dtiles.tilers.b3dm.wkb_utils import TriangleSoup
    >>> from py3dtiles.tileset.content import B3dm, GlTF
    >>>
    >>> # load a wkb file
    >>> wkb = open('tests/fixtures/building.wkb', 'rb').read()
    >>>
    >>> # define the geometry's bounding box
    >>> box = [[-8.75, -7.36, -2.05], [8.80, 7.30, 2.05]]
    >>>
    >>> # define the geometry's world transformation
    >>> transform = np.array([
    ...             [1, 0, 0, 1842015.125],
    ...             [0, 1, 0, 5177109.25],
    ...             [0, 0, 1, 247.87364196777344],
    ...             [0, 0, 0, 1]], dtype=float)
    >>> transform = transform.flatten('F')
    >>>
    >>> # use the TriangleSoup helper class to transform the wkb into arrays
    >>> # of points and normals
    >>> ts = TriangleSoup.from_wkb_multipolygon(wkb)
    >>> positions = ts.get_position_array()
    >>> normals = ts.get_normal_array()
    >>> # generate the glTF part from the binary arrays.
    >>> # notice that from_binary_arrays accepts array of geometries
    >>> # for batching purposes.
    >>> geometry = { 'position': positions, 'normal': normals, 'bbox': box }
    >>> gltf = GlTF.from_binary_arrays([geometry], transform)
    >>>
    >>> # create a b3dm directly from the glTF.
    >>> b3dm = B3dm.from_gltf(gltf)
    >>>
    >>> # to save our tile content as a .b3dm file
    >>> b3dm.save_as(Path("mymodel.b3dm"))
