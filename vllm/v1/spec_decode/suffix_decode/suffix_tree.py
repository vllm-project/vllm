# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Python implementation of the C++ suffix tree for suffix decoding.
This is a temporary drop-in replacement for the C++ SuffixTree and Candidate classes.
"""

from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import heapq


class Node:
    """Represents a node in the suffix tree."""
    
    def __init__(self):
        # Token referenced by this node (first token if sequence)
        self.token = 0
        
        # Number of suffixes from root that end at or pass through this node
        self.count = 0
        
        # Parent node
        self.parent = None
        
        # Children nodes, key is the first token of the child
        self.children = {}  # Dict[int, Node]
        
        # Maps sequence ID -> index of the end of the suffix in that sequence
        self.endpoints = {}  # Dict[int, int]
        
        # Reference sequence ID and starting index for tokens in this node
        self.ref_seq = 0
        self.ref_idx = -1
        
        # Number of tokens in this node
        self.length = 0
    
    def memory_usage(self):
        """Estimate memory usage of this node."""
        # Simplified estimation
        total = 56  # Base object size
        total += len(self.children) * 280  # Dict overhead + pointers
        total += len(self.endpoints) * 72  # Dict items
        return total


class Candidate:
    """Represents a speculation candidate."""
    
    def __init__(self):
        # The token ids of the speculation candidate
        self.token_ids = []
        
        # For each token, the index of its parent token (-1 if no parent)
        self.parents = []
        
        # For each token, the estimated probability of the token
        self.probs = []
        
        # Floating point score of the candidate (sum of all probs)
        self.score = 0.0
        
        # Length of the prefix match for the speculated tokens
        self.match_len = 0


class SuffixTree:
    """
    Suffix tree implementation for token sequence speculation.
    """
    
    def __init__(self, max_depth: int):
        """
        Initialize the suffix tree with a maximum depth.
        
        Args:
            max_depth: Maximum depth of the suffix tree
        """
        self._max_depth = max_depth
        self._root = Node()
        self._seqs: dict[int, list[int]] = {}  # mapping from seq_id to token sequence
        self._active_nodes: dict[int, deque[Node]] = {}  # active nodes per sequence
    
    def num_seqs(self) -> int:
        """Return the number of sequences in the tree."""
        return len(self._seqs)
    
    def append(self, seq_id: int, token: int):
        """
        Append a new element to a new or existing sequence.
        
        Args:
            seq_id: Sequence identifier
            token: Token to append
        """
        # Initialize the sequence if it doesn't exist
        if seq_id not in self._seqs:
            self._seqs[seq_id] = []
            self._active_nodes[seq_id] = deque()
        
        # Insert a new active node at the root
        self._active_nodes[seq_id].append(self._root)
        self._root.endpoints[seq_id] = len(self._seqs[seq_id])
        self._root.count += 1
        
        # Ensure the number of active nodes doesn't exceed max_depth
        if len(self._active_nodes[seq_id]) > self._max_depth:
            self._active_nodes[seq_id].popleft()
        
        self._seqs[seq_id].append(token)
        
        # Iterate over all active nodes for this sequence
        for i in range(len(self._active_nodes[seq_id])):
            node = self._active_nodes[seq_id][i]
            child = node.children.get(token)
            
            assert seq_id in node.endpoints
            assert node.endpoints[seq_id] == len(self._seqs[seq_id]) - 1
            
            if child is None:
                # No existing child node for the new token
                if node.count == 1 and node != self._root:
                    # The active node has count = 1, extend the length of this node
                    assert len(node.children) == 0
                    assert node.ref_seq == seq_id
                    node.length += 1
                    node.endpoints[seq_id] += 1
                else:
                    # Need to extend the current suffix into a new child
                    new_child = Node()
                    new_child.token = token
                    new_child.parent = node
                    new_child.count = 1
                    new_child.endpoints[seq_id] = len(self._seqs[seq_id])
                    new_child.ref_seq = seq_id
                    new_child.ref_idx = len(self._seqs[seq_id]) - 1
                    new_child.length = 1
                    node.children[token] = new_child
                    del node.endpoints[seq_id]
                    self._active_nodes[seq_id][i] = new_child
            
            elif node.count == child.count + 1 and node != self._root:
                # All other suffixes that pass through this node must go to that child
                assert len(node.children) == 1
                assert len(node.endpoints) == 1
                
                if child.length == 1:
                    # Fuse the current suffix into the child
                    parent = node.parent
                    child.token = node.token
                    child.count += 1
                    child.length = node.length + 1
                    child.endpoints[seq_id] = len(self._seqs[seq_id])
                    child.ref_seq = seq_id
                    child.ref_idx = len(self._seqs[seq_id]) - child.length
                    child.parent = parent
                    
                    # Update parent's child pointer
                    assert isinstance(parent, Node)
                    assert child.token in parent.children
                    assert parent.children[child.token] == node
                    parent.children[child.token] = child
                    
                    # Replace active node with child node
                    self._active_nodes[seq_id][i] = child
                else:
                    # Extend the length of the current node by 1
                    node.length += 1
                    node.endpoints[seq_id] += 1
                    node.ref_seq = seq_id
                    node.ref_idx = len(self._seqs[seq_id]) - node.length
                    child.length -= 1
                    child.ref_idx += 1
                    
                    # Update child's first token
                    child.token = self._seqs[child.ref_seq][child.ref_idx]
                    if child.token != token:
                        node.children[child.token] = node.children.pop(token)
            
            else:
                # Move the active node into the child
                if child.length == 1:
                    # Update the active node pointer to child
                    if seq_id in node.endpoints:
                        del node.endpoints[seq_id]
                    child.count += 1
                    child.endpoints[seq_id] = len(self._seqs[seq_id])
                    child.ref_seq = seq_id
                    child.ref_idx = len(self._seqs[seq_id]) - 1
                    self._active_nodes[seq_id][i] = child
                else:
                    # Split the child node
                    new_node = Node()
                    new_node.token = token
                    new_node.count = child.count + 1
                    new_node.parent = node
                    new_node.length = 1
                    new_node.endpoints[seq_id] = len(self._seqs[seq_id])
                    new_node.ref_seq = seq_id
                    new_node.ref_idx = len(self._seqs[seq_id]) - new_node.length
                    
                    # Update child's first token
                    child.token = self._seqs[child.ref_seq][child.ref_idx + 1]
                    new_node.children[child.token] = child
                    node.children[token] = new_node
                    if seq_id in node.endpoints:
                        del node.endpoints[seq_id]
                    child.parent = new_node
                    child.length -= 1
                    child.ref_idx += 1
                    self._active_nodes[seq_id][i] = new_node
    
    def extend(self, seq_id: int, tokens: List[int]):
        """
        Extend a new or existing sequence with multiple tokens.
        
        Args:
            seq_id: Sequence identifier
            tokens: List of tokens to append
        """
        for token in tokens:
            self.append(seq_id, token)
    
    def remove(self, seq_id: int):
        """
        Remove an existing sequence from the tree.
        
        Args:
            seq_id: Sequence identifier to remove
        """
        if seq_id not in self._seqs:
            return
        
        seq = self._seqs[seq_id]
        path: list[Node] = []
        
        # Loop through all suffix starting indices
        for start in range(len(seq)):
            node = self._root
            node.count -= 1
            idx = start
            path.clear()
            
            # Loop through the nodes for this suffix
            while idx < len(seq):
                token = seq[idx]
                if token not in node.children:
                    break
                
                child = node.children[token]
                assert child.count > 0
                child.count -= 1
                
                if child.count == 0:
                    del node.children[token]
                    break
                
                if seq_id in child.endpoints:
                    del child.endpoints[seq_id]
                
                idx += child.length
                node = child
                path.append(node)
            
            # The last visited node may be mergeable with its child
            if node != self._root and len(node.children) == 1:
                token, child = next(iter(node.children.items()))
                if node.count == child.count:
                    # Merge node into child
                    child.token = node.token
                    child.length += node.length
                    child.ref_idx -= node.length
                    child.parent = node.parent
                    path[-1] = node = child
                    assert node.parent is not None
                    node.parent.children[node.token] = node
            
            # Update ref_seq and ref_idx of all nodes in the path
            # 1. Go to an arbitrary leaf
            leaf = node
            distance = 0
            while len(leaf.children) > 0:
                leaf = next(iter(leaf.children.values()))
                distance += leaf.length
            
            # 2. Pick an arbitrary endpoint for reference
            if len(leaf.endpoints) == 0 or seq_id in leaf.endpoints:
                continue
            
            ref_seq, ref_idx = next(iter(leaf.endpoints.items()))
            ref_idx = ref_idx - distance
            
            # 3. Update all nodes' refs
            while path:
                n = path.pop()
                ref_idx -= n.length
                if n.ref_seq == seq_id:
                    n.ref_seq = ref_seq
                    n.ref_idx = ref_idx
        
        del self._seqs[seq_id]
        del self._active_nodes[seq_id]
    
    def speculate(self, pattern: List[int], max_spec_tokens: int,
                  max_spec_factor: float = 1.0, max_spec_offset: float = 0.0,
                  min_token_prob: float = 0.1, use_tree_spec: bool = False) -> Candidate:
        """
        Given a pattern, speculate the next tokens using the suffix tree.
        
        Args:
            pattern: Input token pattern
            max_spec_tokens: Maximum number of tokens to speculate
            max_spec_factor: Factor for computing dynamic max tokens
            max_spec_offset: Offset for computing dynamic max tokens
            min_token_prob: Minimum probability threshold for tokens
            use_tree_spec: Whether to use tree speculation (vs path speculation)
        
        Returns:
            Candidate object with speculation results
        """
        result = Candidate()
        start_idx = max(len(pattern) - self._max_depth, 0)
        
        for idx in range(start_idx, len(pattern)):
            node, offset = self._match_pattern(pattern, idx)
            if node is None:
                continue
            
            match_len = len(pattern) - idx
            max_tokens = min(max_spec_tokens,
                           int(match_len * max_spec_factor + max_spec_offset + 1e-6))
            max_tokens = max(max_tokens, 0)
            
            if use_tree_spec:
                candidate = self._speculate_tree(node, offset, max_tokens, min_token_prob)
            else:
                candidate = self._speculate_path(node, offset, max_tokens, min_token_prob)
            
            if candidate.score > result.score:
                result = candidate
                result.match_len = match_len
        
        return result
    
    def check_integrity(self) -> str:
        """
        Check the integrity of the suffix tree.
        
        Returns:
            Empty string if ok, otherwise an error message
        """
        # 1. Check structural integrity of all nodes
        queue = deque([self._root])
        while queue:
            node = queue.popleft()
            ret = self._check_node_integrity(node)
            if ret:
                return ret
            for child in node.children.values():
                queue.append(child)
        
        # 2. Check all sequences are represented in the tree
        visit_count: dict[int, int] = {}
        for seq_id, seq in self._seqs.items():
            # Loop through all suffix starting indices
            for start in range(len(seq)):
                idx = start
                # Traverse the tree along this suffix
                node = self._root
                visit_count[id(node)] = visit_count.get(id(node), 0) + 1
                
                while idx < len(seq) and idx - start < self._max_depth:
                    if seq[idx] not in node.children:
                        return "missing child node for sequence"
                    
                    node = node.children[seq[idx]]
                    visit_count[id(node)] = visit_count.get(id(node), 0) + 1
                    
                    if idx + node.length > len(seq):
                        return "path exceeds sequence length"
                    
                    for i in range(node.length):
                        ref_seq = node.ref_seq
                        ref_idx = node.ref_idx + i
                        if seq[idx + i] != self._seqs[ref_seq][ref_idx]:
                            return "path does not match sequence tokens"
                    
                    idx += node.length
                
                # The last node should have an endpoint
                if seq_id not in node.endpoints:
                    return "missing endpoint for sequence"
        
        # 3. Check all nodes were visited the correct number of times
        queue = deque([self._root])
        while queue:
            node = queue.popleft()
            if node.count != visit_count.get(id(node), 0):
                return "node count does not match visit count"
            for child in node.children.values():
                queue.append(child)
        
        return ""
    
    def estimate_memory(self) -> int:
        """
        Estimate memory usage of the suffix tree.
        
        Returns:
            Estimated memory in bytes
        """
        total = 56  # Base object size
        
        # Traverse all nodes
        stack = [self._root]
        while stack:
            node = stack.pop()
            total += node.memory_usage()
            for child in node.children.values():
                stack.append(child)
        
        # Add sequence memory
        for seq in self._seqs.values():
            total += len(seq) * 8  # Approximate int size in list
        
        # Add active nodes memory
        for active_nodes in self._active_nodes.values():
            total += len(active_nodes) * 8  # Pointer size
        
        return total
    
    def _check_node_integrity(self, node: Node) -> str:
        """Check integrity of a single node."""
        children_count = len(node.children)
        
        # Check children have correct parent
        for child in node.children.values():
            if child.parent != node:
                return "child node has incorrect parent pointer"
        
        # Check counter
        if children_count > node.count:
            return "node count is less than sum children counts"
        
        if node == self._root:
            # Root node checks
            if node.count < 0:
                return "root node has negative count"
            if node.parent is not None:
                return "root node has non-null parent pointer"
            if node.length != 0:
                return "root node has non-zero length"
            if len(node.endpoints) != 0:
                return "root node has non-empty endpoints"
            if node.ref_idx != -1:
                return "root node has invalid ref_idx"
            return ""
        
        # Internal node checks
        if node.length <= 0:
            return "internal node has non-positive length"
        if node.count <= 0:
            return "internal node has non-positive count"
        
        # Check children counts
        for child in node.children.values():
            if child.count >= node.count:
                return "internal node count is not greater than child count"
        
        # Check reference sequence
        if node.ref_seq not in self._seqs:
            return "internal node has invalid ref_seq"
        if node.ref_idx < 0:
            return "internal node has invalid ref_idx"
        if node.ref_idx + node.length > len(self._seqs[node.ref_seq]):
            return "internal node has invalid token range"
        
        # Check first token
        if node.token != self._seqs[node.ref_seq][node.ref_idx]:
            return "internal node has incorrect first token"
        
        # Check parent relationship
        assert node.parent is not None
        if node.token not in node.parent.children:
            return "internal node is not a child of parent node"
        if node.parent.children[node.token] != node:
            return "parent node has incorrect child pointer"
        
        # Check endpoints
        for seq_id, end_idx in node.endpoints.items():
            if seq_id not in self._seqs:
                return "node endpoint refers to nonexistent sequence"
            if end_idx <= 0 or end_idx > len(self._seqs[seq_id]):
                return "invalid endpoint index"
            
            # Check tokens from start of suffix to endpoint
            n = node
            idx = end_idx
            while n is not None:
                assert n.parent is not None
                if n.length > idx:
                    return "invalid endpoint length"
                idx -= n.length
                for i in range(n.length):
                    tok = self._seqs[n.ref_seq][n.ref_idx + i]
                    if self._seqs[seq_id][idx + i] != tok:
                        return "invalid endpoint token"
                n = n.parent
        
        return ""
    
    def _match_pattern(self, pattern: List[int], start_idx: int = 0) -> Tuple[Optional[Node], int]:
        """
        Match a pattern in the tree starting from start_idx.
        
        Returns:
            (node, offset) where offset is the position within the node, or (None, -1) if no match
        """
        node = self._root
        idx = 0
        
        for i in range(start_idx, len(pattern)):
            c = pattern[i]
            if idx >= node.length:
                if c not in node.children:
                    return None, -1
                node = node.children[c]
                idx = 0
            
            assert idx < node.length
            if self._seqs[node.ref_seq][node.ref_idx + idx] != c:
                return None, -1
            idx += 1
        
        return node, idx
    
    def _speculate_path(self, node: Node, idx: int, max_spec_tokens: int,
                       min_token_prob: float) -> Candidate:
        """Speculate along the most probable path."""
        ret = Candidate()
        prob = 1.0
        
        while len(ret.token_ids) < max_spec_tokens and prob >= min_token_prob:
            if idx < node.length:
                # Use previous token index as parent; if none, mark as -1
                ret.parents.append(len(ret.token_ids) - 1)
                token = self._seqs[node.ref_seq][node.ref_idx + idx]
                ret.token_ids.append(token)
                ret.probs.append(prob)
                ret.score += prob
                idx += 1
            else:
                # Choose the child with maximum count
                best_child = None
                best_count = 0
                for child in node.children.values():
                    if child.count > best_count:
                        best_child = child
                        best_count = child.count
                
                if best_child is None:
                    break
                
                prob *= best_count / node.count
                node = best_child
                idx = 0
        
        return ret
    
    def _speculate_tree(self, node: Node, idx: int, max_spec_tokens: int,
                       min_token_prob: float) -> Candidate:
        """Speculate using a priority queue to build a token tree."""
        ret = Candidate()
        
        # Priority queue: (-prob, node, idx, parent_idx)
        # Using negative prob for max heap behavior
        heap = [(-1.0, node, idx, -1)]
        
        while len(ret.token_ids) < max_spec_tokens and heap:
            neg_prob, node, idx, parent_idx = heapq.heappop(heap)
            prob = -neg_prob
            
            if idx < node.length:
                token = self._seqs[node.ref_seq][node.ref_idx + idx]
                ret.token_ids.append(token)
                ret.parents.append(parent_idx)
                ret.probs.append(prob)
                ret.score += prob
                
                # Add continuation within same node
                new_parent_idx = len(ret.token_ids) - 1
                heapq.heappush(heap, (-prob, node, idx + 1, new_parent_idx))
            else:
                # Add all children to the queue
                for child in node.children.values():
                    child_prob = prob * child.count / node.count
                    if child_prob >= min_token_prob:
                        heapq.heappush(heap, (-child_prob, child, 0, parent_idx))
        
        return ret