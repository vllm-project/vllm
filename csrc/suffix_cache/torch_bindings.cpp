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

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/core/jit_type.h>

#include "suffix_tree.h"
#include "core/registration.h"

// Register custom types with PyTorch
namespace {
c10::intrusive_ptr<c10::ivalue::Object> make_candidate(
    const std::vector<int64_t>& token_ids,
    const std::vector<int64_t>& parents,
    const std::vector<double>& probs,
    double score,
    int64_t match_len) {
    
    auto obj = c10::ivalue::Object::create(
        c10::StrongTypePtr(nullptr, c10::ClassType::create(
            "_suffix_cache.Candidate", c10::nullopt)));
    
    obj->setAttr(0, token_ids);
    obj->setAttr(1, parents);
    obj->setAttr(2, probs);
    obj->setAttr(3, score);
    obj->setAttr(4, match_len);
    
    return obj;
}

// Wrapper functions for SuffixTree operations
class SuffixTreeWrapper {
    std::unique_ptr<SuffixTree> tree_;
public:
    explicit SuffixTreeWrapper(int64_t max_depth) 
        : tree_(std::make_unique<SuffixTree>(static_cast<int>(max_depth))) {}
    
    int64_t num_seqs() const {
        return static_cast<int64_t>(tree_->num_seqs());
    }
    
    void append(int64_t seq_id, int64_t token) {
        tree_->append(static_cast<int>(seq_id), static_cast<int>(token));
    }
    
    void extend(int64_t seq_id, const std::vector<int64_t>& tokens) {
        std::vector<int> int_tokens;
        int_tokens.reserve(tokens.size());
        for (int64_t token : tokens) {
            int_tokens.push_back(static_cast<int>(token));
        }
        tree_->extend(static_cast<int>(seq_id), int_tokens);
    }
    
    void remove(int64_t seq_id) {
        tree_->remove(static_cast<int>(seq_id));
    }
    
    c10::intrusive_ptr<c10::ivalue::Object> speculate(
        const std::vector<int64_t>& pattern,
        int64_t max_spec_tokens,
        double max_spec_factor,
        double max_spec_offset,
        double min_token_prob,
        bool use_tree_spec) {
        
        std::vector<int> int_pattern;
        int_pattern.reserve(pattern.size());
        for (int64_t token : pattern) {
            int_pattern.push_back(static_cast<int>(token));
        }
        
        Candidate result = tree_->speculate(
            int_pattern,
            static_cast<int>(max_spec_tokens),
            static_cast<float>(max_spec_factor),
            static_cast<float>(max_spec_offset),
            static_cast<float>(min_token_prob),
            use_tree_spec);
        
        // Convert Candidate to PyTorch custom type
        std::vector<int64_t> token_ids(result.token_ids.begin(), result.token_ids.end());
        std::vector<int64_t> parents(result.parents.begin(), result.parents.end());
        std::vector<double> probs(result.probs.begin(), result.probs.end());
        
        return make_candidate(token_ids, parents, probs, 
                            static_cast<double>(result.score), 
                            static_cast<int64_t>(result.match_len));
    }
    
    std::string check_integrity() {
        return tree_->check_integrity();
    }
    
    int64_t estimate_memory() const {
        return static_cast<int64_t>(tree_->estimate_memory());
    }
};

// Shim functions for TORCH_LIBRARY registration
torch::Tensor suffix_tree_create(int64_t max_depth) {
    auto wrapper = std::make_unique<SuffixTreeWrapper>(max_depth);
    void* ptr = wrapper.release();
    
    // Store the pointer in a tensor (this is a common pattern in vLLM)
    // We use a CPU int64 tensor to store the pointer
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto tensor = torch::empty({1}, options);
    tensor.data_ptr<int64_t>()[0] = reinterpret_cast<int64_t>(ptr);
    return tensor;
}

void suffix_tree_destroy(torch::Tensor handle) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    delete wrapper;
}

int64_t suffix_tree_num_seqs(torch::Tensor handle) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    return wrapper->num_seqs();
}

void suffix_tree_append(torch::Tensor handle, int64_t seq_id, int64_t token) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    wrapper->append(seq_id, token);
}

void suffix_tree_extend(torch::Tensor handle, int64_t seq_id, torch::Tensor tokens) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    
    auto tokens_accessor = tokens.accessor<int64_t, 1>();
    std::vector<int64_t> token_vec;
    token_vec.reserve(tokens_accessor.size(0));
    for (int64_t i = 0; i < tokens_accessor.size(0); ++i) {
        token_vec.push_back(tokens_accessor[i]);
    }
    
    wrapper->extend(seq_id, token_vec);
}

void suffix_tree_remove(torch::Tensor handle, int64_t seq_id) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    wrapper->remove(seq_id);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t> 
suffix_tree_speculate(torch::Tensor handle, 
                     torch::Tensor pattern,
                     int64_t max_spec_tokens,
                     double max_spec_factor,
                     double max_spec_offset,
                     double min_token_prob,
                     bool use_tree_spec) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    
    auto pattern_accessor = pattern.accessor<int64_t, 1>();
    std::vector<int64_t> pattern_vec;
    pattern_vec.reserve(pattern_accessor.size(0));
    for (int64_t i = 0; i < pattern_accessor.size(0); ++i) {
        pattern_vec.push_back(pattern_accessor[i]);
    }
    
    auto result = wrapper->speculate(pattern_vec, max_spec_tokens, 
                                   max_spec_factor, max_spec_offset,
                                   min_token_prob, use_tree_spec);
    
    // Extract attributes from the custom object
    auto token_ids_list = result->getAttr(0).toIntList();
    auto parents_list = result->getAttr(1).toIntList();
    auto probs_list = result->getAttr(2).toDoubleList();
    double score = result->getAttr(3).toDouble();
    int64_t match_len = result->getAttr(4).toInt();
    
    // Convert to tensors
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    
    auto token_ids_tensor = torch::tensor(token_ids_list, options);
    auto parents_tensor = torch::tensor(parents_list, options);
    auto probs_tensor = torch::tensor(probs_list, float_options);
    
    return std::make_tuple(token_ids_tensor, parents_tensor, probs_tensor, score, match_len);
}

std::string suffix_tree_check_integrity(torch::Tensor handle) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    return wrapper->check_integrity();
}

int64_t suffix_tree_estimate_memory(torch::Tensor handle) {
    int64_t ptr_value = handle.data_ptr<int64_t>()[0];
    auto* wrapper = reinterpret_cast<SuffixTreeWrapper*>(ptr_value);
    return wrapper->estimate_memory();
}

} // anonymous namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, suffix_cache) {
    // SuffixTree operations
    suffix_cache.def("suffix_tree_create(int max_depth) -> Tensor");
    suffix_cache.impl("suffix_tree_create", torch::kCPU, &suffix_tree_create);
    
    suffix_cache.def("suffix_tree_destroy(Tensor handle) -> ()");
    suffix_cache.impl("suffix_tree_destroy", torch::kCPU, &suffix_tree_destroy);
    
    suffix_cache.def("suffix_tree_num_seqs(Tensor handle) -> int");
    suffix_cache.impl("suffix_tree_num_seqs", torch::kCPU, &suffix_tree_num_seqs);
    
    suffix_cache.def("suffix_tree_append(Tensor handle, int seq_id, int token) -> ()");
    suffix_cache.impl("suffix_tree_append", torch::kCPU, &suffix_tree_append);
    
    suffix_cache.def("suffix_tree_extend(Tensor handle, int seq_id, Tensor tokens) -> ()");
    suffix_cache.impl("suffix_tree_extend", torch::kCPU, &suffix_tree_extend);
    
    suffix_cache.def("suffix_tree_remove(Tensor handle, int seq_id) -> ()");
    suffix_cache.impl("suffix_tree_remove", torch::kCPU, &suffix_tree_remove);
    
    suffix_cache.def("suffix_tree_speculate(Tensor handle, Tensor pattern, int max_spec_tokens, float max_spec_factor, float max_spec_offset, float min_token_prob, bool use_tree_spec) -> (Tensor, Tensor, Tensor, float, int)");
    suffix_cache.impl("suffix_tree_speculate", torch::kCPU, &suffix_tree_speculate);
    
    suffix_cache.def("suffix_tree_check_integrity(Tensor handle) -> str");
    suffix_cache.impl("suffix_tree_check_integrity", torch::kCPU, &suffix_tree_check_integrity);
    
    suffix_cache.def("suffix_tree_estimate_memory(Tensor handle) -> int");
    suffix_cache.impl("suffix_tree_estimate_memory", torch::kCPU, &suffix_tree_estimate_memory);
}

REGISTER_EXTENSION(_suffix_cache_C)
