#pragma once

#include <torch/all.h>

#include <map>
#include <vector>
#include<string>
#include <ATen/cuda/CUDAContext.h>

class FileSwapperParam {
public:
    FileSwapperParam(char* cache_ptr, const std::string& file_name, int64_t file_offset,
                      int64_t size, cudaStream_t stream) :
        cache_ptr(cache_ptr), file_name(file_name), file_offset(file_offset),
        size(size), stream(stream) {}

    char* cache_ptr;
    std::string file_name;
    int64_t file_offset;
    int64_t size;
    cudaStream_t stream;
};

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

void swap_out_to_local_file(torch::Tensor& src, std::string file_name,
                            const torch::Tensor& block_mapping);

void swap_in_from_local_file(std::string src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype, const double k_scale,
                       const double v_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             const double k_scale, const double v_scale);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);
