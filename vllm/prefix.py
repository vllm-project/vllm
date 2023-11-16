from typing import Dict, List, Optional, Union

# Define the prefix class, which is a collection of prefix (a sequence of tokens).
# The class contains the following main methods:
# 1. A match method that checks if a prefix matches a given sequence of tokens.
# 2. A swapping method that can load or offload the prefix to or from GPU
# 3. An update_frequency method that updates the frequency of the prefix.
# 4. A get_status method that tells if the prefix is on GPU or not.


class Prefix:
    def __init__(self, prefix_id, token_ids, block_size):
        self.prefix_id = prefix_id
        self.token_ids = token_ids
        self.length = len(token_ids)
        print("prefix length: ", self.length)
        print("block size: ", block_size)
        assert self.length % block_size == 0
        self.on_gpu = False
        self.on_cpu = False
        self.block_table = None
        # a lock to prevent multiple sequence from calculating the same prefix
        self.swap_to_gpu = False

        # freq-related
        self.freq = 1
        self.alpha = 0.8
        self.beta = 0.5
    
    def get_block_table_num(self) -> List[int]:
        return [block.block_number for block in self.block_table]
    
    def match(self, tokens):
        return tokens[:self.length] == self.token_ids
    
    # should be called if the prefix is hit for this iteration
    def update_freq(self, new_hit_rate):
        self.freq = self.alpha * self.freq + (1 - self.alpha) * new_hit_rate
        self.alpha = 0.8
    
    # should be called if the prefix is not hit for this iteration
    def punish_freq(self):
        self.alpha = self.beta * self.alpha if self.alpha > 0.1 else 0.1
   
    # whether the prefix is on GPU or not
    def get_status(self):
        return self.on_gpu
    
    def get_length(self):
        return self.length


# Define the prefix pool class, which is a collection of prefixes.
# The class contains the following main methods:
# 1. add a prefix to the pool, with a computed hash
# 2. TODO: create subprefix, if one is a prefix of the other: they can share some memory blocks
# 3. efficient_search: given a sequence of tokens, find the longest prefix in the pool that matches the sequence
# 4. fixed_search: given the prefix's hash, find the prefix in the pool
# 5. TODO: approximate_search: given a sequence of tokens, find the similar prefixes in the pool


class PrefixPool:
    def __init__(self, block_size):
        self.prefixes = []
        self.prefixes_hash = {}
        self.block_size = block_size
    
    def add_prefix(self, token_ids: List[int]):
        # generate prefix_id
        prefix_id = len(self.prefixes)
        # create a new prefix
        prefix = Prefix(prefix_id, token_ids, self.block_size)
        self.prefixes.append(prefix)
        # @TODO: compute the hash of the prefix
        prefix_hash = hash(tuple(prefix.token_ids))
        self.prefixes_hash[prefix.prefix_id] = prefix_hash
        return prefix
        
    # @TODO: this one should also come with a method to identify the prefix
    def efficient_search(self, token_ids: List[int]):
        # improve this search
        for prefix in self.prefixes:
            if prefix.match(token_ids):
                return prefix
        return None
    
    # use this first, if we already know from the application which part of the tokens are prefix.
    def fixed_search(self, prefix_hash):
        if prefix_hash not in self.prefixes_hash:
            return None
        prefix_id = self.prefixes_hash[prefix_hash]
        return self.prefixes[prefix_id]

