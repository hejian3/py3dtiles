import gc
from pathlib import Path
from sys import getsizeof
import time
from typing import Dict, List, Tuple

import lz4.frame as gzip

from py3dtiles.utils import node_name_to_path


class SharedNodeStore:
    def __init__(self, folder: Path) -> None:
        self.metadata: Dict[bytes, Tuple[float, int] | None] = {}
        self.data: List[bytes | None] = []
        self.folder = folder
        self.stats = {
            'hit': 0,
            'miss': 0,
            'new': 0,
        }
        self.memory_size = {
            'content': 0,
            'container': getsizeof(self.data) + getsizeof(self.metadata),
        }

    def control_memory_usage(self, max_size_mb: int, verbose: int) -> None:
        bytes_to_mb = 1.0 / (1024 * 1024)
        max_size_mb = max(max_size_mb, 200)

        if verbose >= 3:
            self.print_statistics()

        # guess cache size
        cache_size = (self.memory_size['container'] + self.memory_size['content']) * bytes_to_mb

        before = cache_size
        if before < max_size_mb:
            return

        if verbose >= 2:
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CACHE CLEANING [{before}]')
        self.remove_oldest_nodes(1 - max_size_mb / before)
        gc.collect()

        if verbose >= 2:
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CACHE CLEANING')

    def get(self, name: bytes, stat_inc: int = 1) -> bytes:
        metadata = self.metadata.get(name, None)
        data = b''
        if metadata is not None:
            tmp_data = self.data[metadata[1]]
            if tmp_data is None:
                raise RuntimeError("tmp_data shouldn't be None if metadata is not None.")
            else:
                data = tmp_data
            self.stats['hit'] += stat_inc
        else:
            node_path = node_name_to_path(self.folder, name)
            if node_path.exists():
                self.stats['miss'] += stat_inc
                with node_path.open('rb') as f:
                    data = f.read()
            else:
                self.stats['new'] += stat_inc
            #  should we cache this node?

        return data

    def remove(self, name: bytes) -> None:
        meta = self.metadata.pop(name, None)

        node_path = node_name_to_path(self.folder, name)
        if meta is None:
            if not node_path.exists():
                raise FileNotFoundError(f"{node_path} should exist")
        else:
            self.memory_size['content'] -= getsizeof(meta)
            if self.data[meta[1]] is None:
                raise ValueError(f"{name!r} is present in self.metadata but not in self.data.")
            self.memory_size['content'] -= len(self.data[meta[1]]) # type: ignore
            self.memory_size['container'] = getsizeof(self.data) + getsizeof(self.metadata)
            self.data[meta[1]] = None

        if node_path.exists():
            node_path.unlink()

    def put(self, name: bytes, data: bytes) -> None:
        compressed_data = gzip.compress(data)

        metadata = self.metadata.get(name, None)
        if metadata is None:
            metadata = (time.time(), len(self.data))
            self.data.append(compressed_data)
        else:
            metadata = (time.time(), metadata[1])
            self.data[metadata[1]] = compressed_data
        self.metadata.update([(name, metadata)])

        self.memory_size['content'] += len(compressed_data) + getsizeof((name, metadata))
        self.memory_size['container'] = getsizeof(self.data) + getsizeof(self.metadata)

    def remove_oldest_nodes(self, percent: float = 100) -> Tuple[int, int]:
        count = _remove_all(self)

        self.memory_size['content'] = 0
        self.memory_size['container'] = getsizeof(self.data) + getsizeof(self.metadata)

        return count

    def print_statistics(self) -> None:
        print('Stats: Hits = {}, Miss = {}, New = {}'.format(
            self.stats['hit'],
            self.stats['miss'],
            self.stats['new']))


def _remove_all(store: SharedNodeStore) -> Tuple[int, int]:
    # delete the entries
    count = len(store.metadata)
    bytes_written = 0
    for name, meta in store.metadata.items():
        if meta is None:
            continue
        data = store.data[meta[1]]
        if data is None:
            raise ValueError(f"{name!r} is present in self.metadata but not in self.data.")
        node_path = node_name_to_path(store.folder, name)
        with node_path.open('wb') as f:
            bytes_written += f.write(data)

    store.metadata = {}
    store.data = []

    return count, bytes_written
