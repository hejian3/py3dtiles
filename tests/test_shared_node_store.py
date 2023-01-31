from pathlib import Path
import shutil
import unittest

from py3dtiles.tilers.node import SharedNodeStore


class TestSharedNodeStore(unittest.TestCase):
    TMP_DIR = Path('tmp/')
    def test_remove_oldest_nodes(self):
        shared_node_store = SharedNodeStore(TestSharedNodeStore.TMP_DIR)

        self.assertEqual(len(shared_node_store.data), 0)
        self.assertEqual(len(shared_node_store.metadata), 0)

        shared_node_store.put(b'0', b"11111111")

        self.assertEqual(len(shared_node_store.data), 1)
        self.assertEqual(len(shared_node_store.metadata), 1)

        shared_node_store.remove_oldest_nodes()

        self.assertEqual(len(shared_node_store.data), 0)
        self.assertEqual(len(shared_node_store.metadata), 0)

        shutil.rmtree(self.TMP_DIR)
