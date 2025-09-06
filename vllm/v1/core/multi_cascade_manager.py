from kv_cache_utils import KVPrefixTrie
class MultiCascadeManager:
    def __init__(self, kv_prefix_trie: KVPrefixTrie):
        self.kv_prefix_trie = kv_prefix_trie

    def alloc_groups(self):
        pass

    def alloc_leaf_pass(self):
        pass

    def alloc_full_pass(self):
        pass