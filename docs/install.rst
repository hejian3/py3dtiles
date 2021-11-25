Install
-------

Dependencies:
- PDAL > 1.7
- llvm for numba

From pypi
~~~~~~~~~~~~

`py3dtiles` is published on pypi.org.

```
pip install py3dtiles
```



From sources
~~~~~~~~~~~~

To use py3dtiles from sources:

.. code-block:: shell

    $ apt install git python3 python3-pip virtualenv libopenblas-base liblas-c3
    $ git clone git@gitlab.com:Oslandia/py3dtiles.git
    $ cd py3dtiles
    $ virtualenv -p python3 venv
    $ . venv/bin/activate
    (venv)$ pip install -e .
    (venv)$ python setup.py install

If you wan to run unit tests:

`pdal` must be installed in order to run the tests locally:

.. code-block:: shell

    $ apt update
    $ apt install -y pdal


.. code-block:: shell

    (venv)$ pip install pytest pytest-benchmark
    (venv)$ pytest
    ...
