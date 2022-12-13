class Py3dtilesException(Exception):
    """
    All exceptions thrown by py3dtiles code derives this class.

    Client code that wishes to catch all py3dtiles exception can use `except Py3dtilesException`.
    """


class WorkerException(Py3dtilesException):
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
