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

#pragma once

#include <cassert>
#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "int32_map.h"

struct Node {
    // Token referenced by this node. Node can refer to a sequence of tokens,
    // this is just the ID of the first token.
    int token = 0;

    // Number of suffixes from the root that end at or pass through this node.
    int64_t count = 0;

    // Parent node.
    Node* parent = nullptr;

    // Children nodes, the key should always be the first token of the child.
    Int32Map<std::unique_ptr<Node>> children;

    // Maps sequence ID -> index of the end of the suffix in that sequence.
    Int32Map<int> endpoints;

    // Reference sequence ID and starting index for the tokens in this node.
    int ref_seq = 0;
    int ref_idx = -1;

    // Number of tokens in this node.
    int length = 0;

    // Memory usage of this node.
    size_t memory_usage() const {
        size_t total = sizeof(*this);
        total += children.memory_usage();
        total += endpoints.memory_usage();
        return total;
    }
};

struct Candidate {
    // The token ids of the speculation candidate.
    std::vector<int> token_ids;

    // For each token, the index of its parent token (-1 if no parent).
    std::vector<int> parents;

    // For each token, the estimated probability of the token.
    std::vector<float> probs;

    // Floating point score of the candidate (sum of all probs).
    float score = 0.0;

    // Length of the prefix match for the speculated tokens.
    int match_len = 0;
};

class SuffixTree {
public:

    SuffixTree(int max_depth);

    int num_seqs() const {
        return static_cast<int>(_seqs.size());
    }

    // Append a new element to the sequence with id seq_id.
    void append(int seq_id, int token);

    // Append multiple new elements to the sequence with id seq_id.
    void extend(int seq_id, const std::vector<int>& tokens);

    // Remove the sequence with id seq_id.
    void remove(int seq_id);

    // Given a pattern, speculate the next tokens using the suffix tree.
    Candidate speculate(const std::vector<int>& pattern,
                        int max_spec_tokens,
                        float max_spec_factor = 1.0f,
                        float max_spec_offset = 0.0f,
                        float min_token_prob = 0.1f,
                        bool use_tree_spec = false);

    // Check the integrity of the suffix tree, return empty string if ok,
    // otherwise return an error message.
    std::string check_integrity();

    // Estimate memory usage of the suffix tree, for debugging only. It
    // walks the entire tree so can be slow.
    size_t estimate_memory() const;

private:

    // Maximum depth of the suffix tree.
    int _max_depth;

    // The root node of the suffix tree.
    std::unique_ptr<Node> _root;

    // Mapping from seq id to its sequence (vector of ints).
    std::unordered_map<int, std::vector<int>> _seqs;

    // For each sequence, a sliding window of active nodes. Maintains at most
    // _max_depth active nodes for each sequence. Queue is shifted when a new
    // token is added to the sequence. Each active node is in the queue for at
    // most _max_depth iterations before being removed.
    std::unordered_map<int, std::deque<Node*>> _active_nodes;

    std::pair<Node*, int> _match_pattern(const std::vector<int>& pattern,
                                         int start_idx = 0);

    Candidate _speculate_path(Node* node, int idx, int max_spec_tokens,
                              float min_token_prob);

    Candidate _speculate_tree(Node* node, int idx, int max_spec_tokens,
                              float min_token_prob);

    std::string _check_node_integrity(Node* node);
};
