#pragma once

template <int feat_in, int feat_out, typename in_T, typename out_T,
          typename W_T>
void bgmv_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                 const W_T *__restrict__ W,
                 const int64_t *__restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size, int64_t num_layers,
                 int64_t layer_idx, float scale);

// clang-format off

#define FOR_BGMV_WIDE(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, narrow, 128) \
    f(in_T, out_T, W_T, narrow, 256) \
    f(in_T, out_T, W_T, narrow, 512) \
    f(in_T, out_T, W_T, narrow, 768) \
    f(in_T, out_T, W_T, narrow, 1024) \
    f(in_T, out_T, W_T, narrow, 1152) \
    f(in_T, out_T, W_T, narrow, 1280) \
    f(in_T, out_T, W_T, narrow, 1536) \
    f(in_T, out_T, W_T, narrow, 1728) \
    f(in_T, out_T, W_T, narrow, 1792) \
    f(in_T, out_T, W_T, narrow, 2048) \
    f(in_T, out_T, W_T, narrow, 2304) \
    f(in_T, out_T, W_T, narrow, 2560) \
    f(in_T, out_T, W_T, narrow, 2752) \
    f(in_T, out_T, W_T, narrow, 2816) \
    f(in_T, out_T, W_T, narrow, 3072) \
    f(in_T, out_T, W_T, narrow, 3456) \
    f(in_T, out_T, W_T, narrow, 3584) \
    f(in_T, out_T, W_T, narrow, 4096) \
    f(in_T, out_T, W_T, narrow, 4608) \
    f(in_T, out_T, W_T, narrow, 5120) \
    f(in_T, out_T, W_T, narrow, 5504) \
    f(in_T, out_T, W_T, narrow, 5632) \
    f(in_T, out_T, W_T, narrow, 6144) \
    f(in_T, out_T, W_T, narrow, 6848) \
    f(in_T, out_T, W_T, narrow, 6912) \
    f(in_T, out_T, W_T, narrow, 7168) \
    f(in_T, out_T, W_T, narrow, 8192) \
    f(in_T, out_T, W_T, narrow, 9216) \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 11008) \
    f(in_T, out_T, W_T, narrow, 12288) \
    f(in_T, out_T, W_T, narrow, 13696) \
    f(in_T, out_T, W_T, narrow, 13824) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 16384) \
    f(in_T, out_T, W_T, narrow, 20480) \
    f(in_T, out_T, W_T, narrow, 22016) \
    f(in_T, out_T, W_T, narrow, 24576) \
    f(in_T, out_T, W_T, narrow, 27392) \
    f(in_T, out_T, W_T, narrow, 28672) \
    f(in_T, out_T, W_T, narrow, 32000) \
    f(in_T, out_T, W_T, narrow, 32256) \
    f(in_T, out_T, W_T, narrow, 32512) \
    f(in_T, out_T, W_T, narrow, 32768) \
    f(in_T, out_T, W_T, narrow, 33024) \
    f(in_T, out_T, W_T, narrow, 36864) \
    f(in_T, out_T, W_T, narrow, 49152) \
// Keep above in sync with vllm/lora/layers::SamplerWithLoRA

// Keep this in sync with vllm/config::LoRAConfig
#define FOR_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 64)

// clang-format on
