class Py3dtilesException(Exception):
    """
    All exceptions thrown by py3dtiles code derives this class.

    Client code that wishes to catch all py3dtiles exception can use `except Py3dtilesException`.
    """
    pass


class WorkerException(Py3dtilesException):
    """
    This exception will be thrown by the conversion code if one exception occurs inside a worker.
    """
    pass
