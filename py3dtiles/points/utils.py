from enum import Enum
from io import StringIO
from pathlib import Path, PurePath
from typing import Callable

import numpy as np


class CommandType(Enum):
    READ_FILE = b'read_file'
    WRITE_PNTS = b'write_pnts'
    PROCESS_JOBS = b'process_jobs'
    SHUTDOWN = b'shutdown'


class ResponseType(Enum):
    IDLE = b'idle'
    HALTED = b'halted'
    READ = b'read'
    PROCESSED = b'processed'
    PNTS_WRITTEN = b'pnts_written'
    NEW_TASK = b'new_task'
    ERROR = b'error'


def profile(func: Callable) -> Callable:
    from line_profiler import LineProfiler

    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        deco = lp(func)
        res = deco(*args, **kwargs)
        s = StringIO()
        lp.print_stats(stream=s)
        print(s.getvalue())
        return res
    return wrapper


class SubdivisionType(Enum):
    OCTREE = 1
    QUADTREE = 2


def name_to_filename(working_dir: str, nameb: bytes, suffix: str = '', split_len: int = 8) -> str:
    """
    Get the filename of a tile from its name and the working directory.
    If the name is '222262175' with the suffix '.pnts', the result is 'working_dir/22226217/r5.pnts'
    """
    name = nameb.decode('ascii')
    folder_path = Path(working_dir)
    if len(name) <= split_len:
        filename = PurePath("r" + name + suffix)
    else:
        # the name is split on every 'split_len' char to avoid to have too many tiles on the same folder.
        sub_folders = [name[i:i + split_len] for i in range(0, len(name), split_len)]
        folder_path = folder_path.joinpath(*sub_folders[:-1])
        filename = PurePath("r" + sub_folders[-1] + suffix)

    full_path = folder_path / filename
    folder_path.mkdir(parents=True, exist_ok=True)

    return str(full_path)


def compute_spacing(aabb: np.ndarray) -> float:
    return float(np.linalg.norm(aabb[1] - aabb[0]) / 125)


def aabb_size_to_subdivision_type(size: np.ndarray) -> SubdivisionType:
    if size[2] / min(size[0], size[1]) < 0.5:
        return SubdivisionType.QUADTREE
    else:
        return SubdivisionType.OCTREE


def split_aabb(aabb: np.ndarray, index: int, force_quadtree: bool = False) -> np.ndarray:
    half = (aabb[1] - aabb[0]) * 0.5
    t = aabb_size_to_subdivision_type(half)

    new_aabb = np.array([np.copy(aabb[0]), aabb[0] + half])
    if index & 4:
        new_aabb[0][0] += half[0]
        new_aabb[1][0] += half[0]
    if index & 2:
        new_aabb[0][1] += half[1]
        new_aabb[1][1] += half[1]

    if force_quadtree or t == SubdivisionType.QUADTREE:
        new_aabb[1][2] += half[2]
    elif index & 1:
        new_aabb[0][2] += half[2]
        new_aabb[1][2] += half[2]

    return new_aabb


def make_aabb_cubic(aabb):
    s = max(aabb[1] - aabb[0])
    aabb[1][0] = aabb[0][0] + s
    aabb[1][1] = aabb[0][1] + s
    aabb[1][2] = aabb[0][2] + s
    return aabb

