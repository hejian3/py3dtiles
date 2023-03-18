import os
import re

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

requirements = (
    "cython",
    "earcut==1.1.5",
    "laspy>=2.0,<3.0",
    "lz4",
    "numba",
    "numpy>=1.20.0,<1.23",
    "plyfile",
    "psutil",
    "psycopg2-binary",
    "pyproj",
    "pyzmq",
)

dev_requirements = (
    "commitizen",
    "line_profiler",
    "pre-commit",
    "pytest",
    "pytest-benchmark",
    "pytest-cov",
    "mypy",
    "typing_extensions",
    "types_psutil",
    "types_psycopg2",
)

doc_requirements = (
    "sphinx",
    "sphinx-multiversion",
    "sphinx_rtd_theme",
)

packaging_requirements = sum(
    (
        dev_requirements,
        ("build", "twine", "wheel"),
    ),
    (),
)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version(*file_paths):
    """
    see https://github.com/pypa/sampleproject/blob/master/setup.py
    """

    with open(os.path.join(here, *file_paths)) as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError(
        "Unable to find version string. " "Should be at the first line of __init__.py."
    )


setup(
    name="py3dtiles",
    version=find_version("py3dtiles", "__init__.py"),
    description="Python module for 3D tiles format",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    url="https://gitlab.com/Oslandia/py3dtiles",
    author="Oslandia",
    author_email="contact@oslandia.com",
    license="Apache License Version 2.0",
    python_requires=">=3.8,<3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    install_requires=requirements,
    test_suite="tests",
    extras_require={
        "dev": dev_requirements,
        "doc": doc_requirements,
        "pack": packaging_requirements,
    },
    entry_points={
        "console_scripts": ["py3dtiles=py3dtiles.command_line:main"],
    },
    zip_safe=False,  # zip packaging conflicts with Numba cache (#25)
)
