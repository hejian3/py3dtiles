class Py3dtilesException(Exception):
    """
    All exceptions thrown by py3dtiles code derives this class.

    Client code that wishes to catch all py3dtiles exception can use `except Py3dtilesException`.
    """


class TilerException(Py3dtilesException):
    """
    This exception will be thrown when there is an issue during a tiling task.
    """


class WorkerException(TilerException):
    """
    This exception will be thrown by the conversion code if one exception occurs inside a worker.
    """


class SrsInMissingException(Py3dtilesException):
    """
    This exception will be thrown when an input srs is required but not provided.
    """


class SrsInMixinException(Py3dtilesException):
    """
    This exception will be thrown when among all input files, there is a mix of input srs.
    """


class Invalid3dtilesError(Py3dtilesException):
    """
    This exception will be thrown if the 3d tile specification isn't respected.
    """


class InvalidPntsError(Invalid3dtilesError):
    """
    This exception will be thrown if the point cloud format isn't respected.
    """


class InvalidB3dmError(Invalid3dtilesError):
    """
    This exception will be thrown if the batched 3D model format isn't respected.
    """


class InvalidTilesetError(Invalid3dtilesError):
    """
    This exception will be thrown if the tileset format isn't respected.
    """
