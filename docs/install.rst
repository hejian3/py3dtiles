Install
-------

Dependencies:
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

If you want to run unit tests:

.. code-block:: shell

    (venv)$ pip install pytest pytest-benchmark
    (venv)$ pytest


Supporting LAZ files
~~~~~~~~~~~~~~~~~~~~

To support laz files you need an external library and a laz backend for
laspy, see [this link]](https://laspy.readthedocs.io/en/latest/installation.html#pip). Short answer, for laszip, you need to follow these steps:

.. code-block:: shell

  # install liblaszip, for instance on ubuntu 22.04
  apt-get install -y liblaszip8

  # Install with LAZ support via laszip
  pip install laspy[laszip]
