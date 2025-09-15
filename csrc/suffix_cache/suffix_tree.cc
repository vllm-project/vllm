// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "suffix_tree.h"

#define CHECK_OR_RETURN(cond, msg) if (!(cond)) return msg;

SuffixTree::SuffixTree(int max_depth)
    : _max_depth(max_depth), _root(new Node()) {
}

// Append a new element to a new or existing sequence.
void SuffixTree::append(int seq_id, int token) {
    // Initialize the sequence if it doesn't exist.
    _seqs.try_emplace(seq_id);
    _active_nodes.try_emplace(seq_id);

    // Insert a new active node at the root.
    _active_nodes[seq_id].push_back(_root.get());
    _root->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
    _root->count += 1;

    // Ensure the number of active nodes doesn't exceed max_depth.
    if (_active_nodes[seq_id].size() > static_cast<size_t>(_max_depth)) {
        _active_nodes[seq_id].pop_front();
    }
    _seqs[seq_id].push_back(token);
    
    // Iterate over all active nodes for this sequence.
    for (size_t i = 0; i < _active_nodes[seq_id].size(); ++i) {
        Node* node = _active_nodes[seq_id][i];
        Node* child = nullptr;
        if (node->children.contains(token)) {
            child = node->children[token].get();
        }

        assert(node->endpoints.contains(seq_id));
        assert(node->endpoints[seq_id] == _seqs[seq_id].size() - 1);

        if (child == nullptr) {
            // No existing child node for the new token.
            if (node->count == 1 && node != _root.get()) {
                // The active node has count = 1, which means the only suffix that ends here is the
                // one that's being extended right now. Then this node should be a leaf node, and
                // we can simply extend the length of this node.
                assert(node->children.empty());
                assert(node->ref_seq == seq_id);
                node->length += 1;
                node->endpoints[seq_id] += 1;
            } else {
                // Either this is the root node, or the current suffix is not the only one that
                // ends here. Either case, we need to extend the current suffix into a new child.
                Node* new_child = new Node();
                new_child->token = token;
                new_child->parent = node;
                new_child->count = 1;
                new_child->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                new_child->ref_seq = seq_id;
                new_child->ref_idx = static_cast<int>(_seqs[seq_id].size()) - 1;
                new_child->length = 1;
                node->children.emplace(token, new_child);
                node->endpoints.erase(seq_id);
                _active_nodes[seq_id][i] = new_child;
            }
        }
        else if (node->count == child->count + 1 && node != _root.get()) {
            // The active node has a child for the new token, and the child's count is exactly one
            // fewer than the active node's count. Since the suffix for the active node ends here,
            // that means all other suffixes that pass through this node must go to that child.
            assert(node->children.size() == 1);  // The active node should have only one child.
            assert(node->endpoints.size() == 1);  // Only the current suffix should end here.
            if (child->length == 1) {
                // The child only has length 1. If we append the new token to the current suffix,
                // then it will perfectly overlap with the child. In this case, we should just fuse
                // the current suffix into the child and eliminate the current node.
                Node* parent = node->parent;
                // Update child to take the place of the current node.
                child->token = node->token;
                child->count += 1;  // Current suffix extends into the child
                child->length = node->length + 1;
                child->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                child->ref_seq = seq_id;
                child->ref_idx = static_cast<int>(_seqs[seq_id].size()) - child->length;
                child->parent = parent;
                // Give ownership of child pointer to parent and should also free the current node.
                assert(parent->children.contains(child->token));
                assert(parent->children[child->token].get() == node);
                Node* tmp = node->children[token].release();
                parent->children[child->token].reset(tmp);
                // Replace active node with child node.
                _active_nodes[seq_id][i] = child;
            } else {
                // The child has length > 1. If we append the new token to the current suffix, then
                // it still does not reach the child node. In this case, we keep both nodes but
                // extend the length of the current node by 1 into the child node.
                node->length += 1;
                node->endpoints[seq_id] += 1;
                node->ref_seq = seq_id;
                node->ref_idx = static_cast<int>(_seqs[seq_id].size()) - node->length;
                child->length -= 1;
                child->ref_idx += 1;
                // The child node's first token should be updated to its second token.
                child->token = _seqs[child->ref_seq][child->ref_idx];
                if (child->token != token) {
                    Node* tmp = node->children[token].release();
                    node->children.emplace(child->token, tmp);
                    node->children.erase(token);
                }
            }
        }
        else {
            // There is a child for the new token, and should move the active node into that child.
            if (child->length == 1) {
                // The child node has length 1, just update the active node pointer to it.
                node->endpoints.erase(seq_id);
                child->count += 1;
                child->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                child->ref_seq = seq_id;
                child->ref_idx = static_cast<int>(_seqs[seq_id].size()) - 1;
                _active_nodes[seq_id][i] = child;
            } else {
                // The child node has length > 1. If we extend the current suffix into it, then it
                // must be split into a segment of length 1 and another segment with the remainder.
                Node* new_node = new Node();
                new_node->token = token;
                new_node->count = child->count + 1;
                new_node->parent = node;
                new_node->length = 1;
                new_node->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                new_node->ref_seq = seq_id;
                new_node->ref_idx = static_cast<int>(_seqs[seq_id].size()) - new_node->length;
                // The child node's first token should be updated to its second token.
                child->token = _seqs[child->ref_seq][child->ref_idx + 1];
                Node* tmp = node->children[token].release();
                new_node->children.emplace(child->token, tmp);
                node->children[token].reset(new_node);
                node->endpoints.erase(seq_id);
                child->parent = new_node;
                child->length -= 1;
                child->ref_idx += 1;
                _active_nodes[seq_id][i] = new_node;
            }
        }
    }
}

// Extend a new or existing sequence.
void SuffixTree::extend(int seq_id, const std::vector<int>& tokens) {
    for (int token : tokens) {
        append(seq_id, token);
    }
}

// Remove an existing sequence.
void SuffixTree::remove(int seq_id) {
    const std::vector<int>& seq = _seqs[seq_id];
    std::vector<Node*> path;  // Declare here to avoid repeated allocations.
    // Loop through all suffix starting indices.
    for (int start = 0; start < seq.size(); start++) {
        Node *node = _root.get();
        node->count--;
        int idx = start;
        path.clear();
        // Loop through the nodes for this suffix.
        while (idx < seq.size()) {
            int token = seq[idx];
            if (!node->children.contains(token)) {
                break;
            }
            Node* child = node->children[token].get();
            assert(child->count > 0);
            child->count--;
            if (child->count == 0) {
                node->children.erase(token);
                break;
            }
            if (child->endpoints.contains(seq_id)) {
                child->endpoints.erase(seq_id);
            }
            idx += child->length;
            node = child;
            path.push_back(node);
        }
        // The last visited node may be mergeable with its child.
        if (node != _root.get() && node->children.size() == 1) {
            const auto& it = *node->children.begin();
            std::unique_ptr<Node>& child_uptr = node->children[it.first];
            if (node->count == child_uptr->count) {
                // Merge node into child.
                child_uptr->token = node->token;
                child_uptr->length += node->length;
                child_uptr->ref_idx -= node->length;
                child_uptr->parent = node->parent;
                path.back() = node = child_uptr.release();
                node->parent->children[node->token].reset(node);
            }
        }
        // ref_seq and ref_idx of all nodes in the path may need to be updated.
        // 1. Go to an arbitrary leaf to get its endpoints.
        Node* leaf = node;
        int distance = 0;  // Distance from node to leaf.
        while (!leaf->children.empty()) {
            leaf = (*leaf->children.begin()).second.get();
            distance += leaf->length;
        }
        // 2. Pick an arbitrary endpoint for the reference sequence and index.
        if (leaf->endpoints.empty() || leaf->endpoints.contains(seq_id)) {
            // Still need to visit this leaf later when removing this sequence.
            // We can skip updating the refs until the next time it's visited.
            continue;
        }
        const auto& ref = *leaf->endpoints.begin();
        // 3. Go back up the path to update all nodes' refs.
        int32_t ref_seq = ref.first;
        int32_t ref_idx = ref.second - distance;
        while (!path.empty()) {
            Node* n = path.back();
            path.pop_back();
            ref_idx -= n->length;
            if (n->ref_seq == seq_id) {
                n->ref_seq = ref_seq;
                n->ref_idx = ref_idx;
            }
        }
    }
    _seqs.erase(seq_id);
    _active_nodes.erase(seq_id);
}

Candidate SuffixTree::speculate(const std::vector<int>& pattern,
                                int max_spec_tokens,
                                float max_spec_factor,
                                float max_spec_offset,
                                float min_token_prob,
                                bool use_tree_spec) {
    Candidate result;
    int start_idx = std::max(static_cast<int>(pattern.size()) - _max_depth, 0);
    for ( ; start_idx < pattern.size(); start_idx++) {
        auto[node, idx] = _match_pattern(pattern, start_idx);
        if (node == nullptr) {
            continue;
        }
        int match_len = static_cast<int>(pattern.size()) - start_idx;
        int max_tokens = std::min(max_spec_tokens,
                                  static_cast<int>(match_len * max_spec_factor
                                                   + max_spec_offset + 1e-6));
        max_tokens = std::max(max_tokens, 0);
        Candidate candidate;
        if (use_tree_spec) {
            candidate = _speculate_tree(node, idx, max_tokens, min_token_prob);
        } else {
            candidate = _speculate_path(node, idx, max_tokens, min_token_prob);
        }
        if (candidate.score > result.score) {
            result = std::move(candidate);
            result.match_len = match_len;
        }
    }
    return result;
}

std::string SuffixTree::check_integrity() {
    // 1. Check structural integrity of all nodes.
    std::queue<Node*> queue;
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        std::string ret = _check_node_integrity(node);
        if (!ret.empty()) {
            return ret;
        }
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    // 2. Check all sequences are represented in the tree.
    std::unordered_map<Node*, int64_t> visit_count;
    for (int seq_id = 0; seq_id < _seqs.size(); seq_id++) {
        const std::vector<int>& seq = _seqs[seq_id];
        // Loop through all suffix starting indices.
        for (int start = 0; start < seq.size(); start++) {
            int idx = start;
            // Traverse the tree along this suffix.
            Node* node = _root.get();
            visit_count[node]++;
            while (idx < seq.size() && idx - start < _max_depth) {
                CHECK_OR_RETURN(node->children.contains(seq[idx]),
                                "missing child node for sequence");
                node = node->children[seq[idx]].get();
                visit_count[node]++;
                CHECK_OR_RETURN(idx + node->length <= seq.size(),
                                "path exceeds sequence length");
                for (int i = 0; i < node->length; ++i) {
                    int ref_seq = node->ref_seq;
                    int ref_idx = node->ref_idx + i;
                    CHECK_OR_RETURN(seq[idx + i] == _seqs[ref_seq][ref_idx],
                                    "path does not match sequence tokens");
                }
                idx += node->length;
            }
            // The last node on this path should have an endpoint.
            CHECK_OR_RETURN(node->endpoints.contains(seq_id),
                            "missing endpoint for sequence");
        }
    }
    // 3. Check all nodes were visited the correct number of times.
    assert(queue.empty());
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        CHECK_OR_RETURN(node->count == visit_count[node],
                        "node count does not match visit count");
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    return "";
}

std::string SuffixTree::_check_node_integrity(Node* node) {
    int64_t children_count = 0;
    for (const auto& [token, child] : node->children) {
        // Do all my children have me as their parent?
        CHECK_OR_RETURN(child->parent == node, "child node has incorrect parent pointer");
        children_count++;
    }
    // Is my counter at least the sum of my childrens' counters?
    CHECK_OR_RETURN(children_count <= node->count, "node count is less than sum children counts");
    if (node == _root.get()) {
        // Root node can stop here after some simple checks.
        CHECK_OR_RETURN(node->count >= 0, "root node has negative count");
        CHECK_OR_RETURN(node->parent == nullptr, "root node has non-null parent pointer");
        CHECK_OR_RETURN(node->length == 0, "root node has non-zero length");
        CHECK_OR_RETURN(node->endpoints.empty(), "root node has non-empty endpoints");
        CHECK_OR_RETURN(node->ref_idx == -1, "root node has invalid ref_idx");
        return "";
    }
    // Is my length positive? Otherwise, I shouldn't exist.
    CHECK_OR_RETURN(node->length > 0, "internal node has non-positive length");
    // Is my count positive? Otherwise, I shouldn't exist.
    CHECK_OR_RETURN(node->count > 0, "internal node has non-positive count");
    // Are all my children's counts less than mine? If equal, then we should have been merged.
    for (const auto& [token, child] : node->children) {
        CHECK_OR_RETURN(
            child->count < node->count, "internal node count is not greater than child count");
    }
    // Check my reference sequence and index.
    CHECK_OR_RETURN(_seqs.count(node->ref_seq), "internal node has invalid ref_seq");
    CHECK_OR_RETURN(node->ref_idx >= 0, "internal node has invalid ref_idx");
    CHECK_OR_RETURN(node->ref_idx + node->length <= _seqs[node->ref_seq].size(),
                    "internal node has invalid token range");
    // Check my first token is correct.
    CHECK_OR_RETURN(node->token == _seqs[node->ref_seq][node->ref_idx],
                    "internal node has incorrect first token");
    // Check I am my parent's child.
    CHECK_OR_RETURN(node->parent->children.contains(node->token),
                    "internal node is not a child of parent node");
    CHECK_OR_RETURN(node->parent->children[node->token].get() == node,
                    "parent node has incorrect child pointer");
    // Check all my endpoint references are correct.
    for (auto [seq_id, end_idx] : node->endpoints) {
        CHECK_OR_RETURN(_seqs.count(seq_id), "node endpoint refers to nonexistent sequence");
        CHECK_OR_RETURN(end_idx > 0 && end_idx <= _seqs[seq_id].size(), "invalid endpoint index");
        // Check all tokens from the start of the suffix to the endpoint.
        Node* n = node;
        int idx = end_idx;
        do {
            CHECK_OR_RETURN(n->length <= idx, "invalid endpoint length");
            idx -= n->length;
            for (int i = 0; i < n->length; ++i) {
                int tok = _seqs[n->ref_seq][n->ref_idx + i];
                CHECK_OR_RETURN(_seqs[seq_id][idx + i] == tok, "invalid endpoint token");
            }
            n = n->parent;
        } while (n != nullptr);
    }
    return "";
}

std::pair<Node*, int> SuffixTree::_match_pattern(
        const std::vector<int>& pattern, int start_idx) {
    Node* node = _root.get();
    int idx = 0;
    for (int i = start_idx; i < pattern.size(); i++) {
        int c = pattern[i];
        if (idx >= node->length) {
            if (!node->children.contains(c)) {
                return {nullptr, -1};
            }
            node = node->children[c].get();
            idx = 0;
        }
        assert(idx < node->length);
        if (_seqs[node->ref_seq][node->ref_idx + idx] != c) {
            return {nullptr, -1};
        }
        idx++;
    }
    return {node, idx};
}

Candidate SuffixTree::_speculate_path(Node* node, int idx,
                                      int max_spec_tokens,
                                      float min_token_prob) {
    Candidate ret;
    float prob = 1.0f;
    while (ret.token_ids.size() < max_spec_tokens && prob >= min_token_prob) {
        if (idx < node->length) {
            // Use previous token index as parent; if none, mark as -1.
            ret.parents.push_back(static_cast<int>(ret.token_ids.size()) - 1);
            int token = _seqs[node->ref_seq][node->ref_idx + idx];
            ret.token_ids.push_back(token);
            ret.probs.push_back(prob);
            ret.score += prob;
            idx++;
        } else {
            Node* child = nullptr;
            int64_t count = 0;
            // Choose the child with the maximum count.
            for (const auto& kv : node->children) {
                Node* ch = kv.second.get();
                if (ch->count > count) {
                    child = ch;
                    count = ch->count;
                }
            }
            if (child == nullptr) {
                break;
            }
            prob *= static_cast<float>(count) / node->count;
            node = child;
            idx = 0;
        }
    }
    return ret;
}

struct HeapItem {
    float prob;
    Node* node;
    int idx;
    int parent;   // index in the candidate token list; -1 if none.

    HeapItem(float p, Node* n, int i, int par)
        : prob(p), node(n), idx(i), parent(par) {}
};

struct HeapItemCompare {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        // In C++ priority_queue by default returns the largest element.
        // Thus, we compare probabilities so that the highest prob is returned.
        return a.prob < b.prob;
    }
};

// Get a candidate token tree using a priority queue.
Candidate SuffixTree::_speculate_tree(Node* node, int idx,
                                      int max_spec_tokens,
                                      float min_token_prob) {
    Candidate ret;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapItemCompare> queue;
    queue.emplace(1.0, node, idx, -1);
    while (ret.token_ids.size() < max_spec_tokens && !queue.empty()) {
        HeapItem item = queue.top();
        queue.pop();
        if (item.idx < item.node->length) {
            int token = _seqs[item.node->ref_seq][item.node->ref_idx + item.idx];
            ret.token_ids.push_back(token);
            ret.parents.push_back(item.parent);
            ret.probs.push_back(item.prob);
            ret.score += item.prob;
            queue.emplace(item.prob, item.node, item.idx + 1,
                          static_cast<int>(ret.token_ids.size()) - 1);
        } else {
            for (const auto& kv : item.node->children) {
                Node* child = kv.second.get();
                float prob = item.prob * child->count / 
                    static_cast<float>(item.node->count);
                if (prob >= min_token_prob) {
                    queue.emplace(prob, child, 0, item.parent);
                }
            }
        }
    }
    return ret;
}

size_t SuffixTree::estimate_memory() const {
    size_t total = sizeof(*this);
    std::vector<Node*> stack;
    stack.push_back(_root.get());
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        total += node->memory_usage();
        for (const auto& [token, child] : node->children) {
            stack.push_back(child.get());
        }
    }
    for (const auto& [seq_id, seq] : _seqs) {
        total += sizeof(decltype(seq)::value_type) * seq.capacity();
    }
    for (const auto& [seq_id, active_nodes] : _active_nodes) {
        total += sizeof(decltype(active_nodes)::value_type) * active_nodes.size();
    }
    return total;
}
