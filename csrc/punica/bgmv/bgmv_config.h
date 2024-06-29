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
    f(in_T, out_T, W_T, narrow, 640) \
    f(in_T, out_T, W_T, narrow, 768) \
    f(in_T, out_T, W_T, narrow, 896) \
    f(in_T, out_T, W_T, narrow, 1024) \
    f(in_T, out_T, W_T, narrow, 1152) \
    f(in_T, out_T, W_T, narrow, 1216) \
    f(in_T, out_T, W_T, narrow, 1280) \
    f(in_T, out_T, W_T, narrow, 1536) \
    f(in_T, out_T, W_T, narrow, 1664) \
    f(in_T, out_T, W_T, narrow, 1728) \
    f(in_T, out_T, W_T, narrow, 1792) \
    f(in_T, out_T, W_T, narrow, 2048) \
    f(in_T, out_T, W_T, narrow, 2240) \
    f(in_T, out_T, W_T, narrow, 2304) \
    f(in_T, out_T, W_T, narrow, 2368) \
    f(in_T, out_T, W_T, narrow, 2432) \
    f(in_T, out_T, W_T, narrow, 2560) \
    f(in_T, out_T, W_T, narrow, 2752) \
    f(in_T, out_T, W_T, narrow, 2816) \
    f(in_T, out_T, W_T, narrow, 3072) \
    f(in_T, out_T, W_T, narrow, 3328) \
    f(in_T, out_T, W_T, narrow, 3456) \
    f(in_T, out_T, W_T, narrow, 3584) \
    f(in_T, out_T, W_T, narrow, 3712) \
    f(in_T, out_T, W_T, narrow, 4096) \
    f(in_T, out_T, W_T, narrow, 4480) \
    f(in_T, out_T, W_T, narrow, 4608) \
    f(in_T, out_T, W_T, narrow, 4736) \
    f(in_T, out_T, W_T, narrow, 4864) \
    f(in_T, out_T, W_T, narrow, 5120) \
    f(in_T, out_T, W_T, narrow, 5504) \
    f(in_T, out_T, W_T, narrow, 5632) \
    f(in_T, out_T, W_T, narrow, 5888) \
    f(in_T, out_T, W_T, narrow, 6144) \
    f(in_T, out_T, W_T, narrow, 6400) \
    f(in_T, out_T, W_T, narrow, 6848) \
    f(in_T, out_T, W_T, narrow, 6912) \
    f(in_T, out_T, W_T, narrow, 7168) \
    f(in_T, out_T, W_T, narrow, 7424) \
    f(in_T, out_T, W_T, narrow, 8192) \
    f(in_T, out_T, W_T, narrow, 8960) \
    f(in_T, out_T, W_T, narrow, 9216) \
    f(in_T, out_T, W_T, narrow, 9472) \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 11008) \
    f(in_T, out_T, W_T, narrow, 11264) \
    f(in_T, out_T, W_T, narrow, 12288) \
    f(in_T, out_T, W_T, narrow, 13696) \
    f(in_T, out_T, W_T, narrow, 13824) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 14784) \
    f(in_T, out_T, W_T, narrow, 14848) \
    f(in_T, out_T, W_T, narrow, 15360) \
    f(in_T, out_T, W_T, narrow, 16384) \
    f(in_T, out_T, W_T, narrow, 18944) \
    f(in_T, out_T, W_T, narrow, 20480) \
    f(in_T, out_T, W_T, narrow, 22016) \
    f(in_T, out_T, W_T, narrow, 22528) \
    f(in_T, out_T, W_T, narrow, 24576) \
    f(in_T, out_T, W_T, narrow, 27392) \
    f(in_T, out_T, W_T, narrow, 27648) \
    f(in_T, out_T, W_T, narrow, 28672) \
    f(in_T, out_T, W_T, narrow, 29568) \
    f(in_T, out_T, W_T, narrow, 29696) \
    f(in_T, out_T, W_T, narrow, 32000) \
    f(in_T, out_T, W_T, narrow, 32256) \
    f(in_T, out_T, W_T, narrow, 32512) \
    f(in_T, out_T, W_T, narrow, 32768) \
    f(in_T, out_T, W_T, narrow, 33024) \
    f(in_T, out_T, W_T, narrow, 36864) \
    f(in_T, out_T, W_T, narrow, 43264) \
    f(in_T, out_T, W_T, narrow, 49152) \
    f(in_T, out_T, W_T, narrow, 60544) \
    f(in_T, out_T, W_T, narrow, 60672) \
    f(in_T, out_T, W_T, narrow, 64000) \
    f(in_T, out_T, W_T, narrow, 64256) \
    f(in_T, out_T, W_T, narrow, 64512) \
    f(in_T, out_T, W_T, narrow, 102400) \
    f(in_T, out_T, W_T, narrow, 102656) \
    f(in_T, out_T, W_T, narrow, 102912) \
    f(in_T, out_T, W_T, narrow, 128000) \
    f(in_T, out_T, W_T, narrow, 128256) \
    f(in_T, out_T, W_T, narrow, 128512) \
    
    
// Keep above in sync with vllm/lora/layers::LogitsProcessorWithLoRA
// and vllm/tests/lora/test_punica.py

// Used for defining kernels going from the variety of
// dim in to the narrow dim out
    // Using it for the fully sharded column
    // parallel LoRA A which splits the rank dim
#define FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, 128, narrow) \
    f(in_T, out_T, W_T, 256, narrow) \
    f(in_T, out_T, W_T, 512, narrow) \
    f(in_T, out_T, W_T, 640, narrow) \
    f(in_T, out_T, W_T, 768, narrow) \
    f(in_T, out_T, W_T, 896, narrow) \
    f(in_T, out_T, W_T, 1024, narrow) \
    f(in_T, out_T, W_T, 1152, narrow) \
    f(in_T, out_T, W_T, 1216, narrow) \
    f(in_T, out_T, W_T, 1280, narrow) \
    f(in_T, out_T, W_T, 1536, narrow) \
    f(in_T, out_T, W_T, 1664, narrow) \
    f(in_T, out_T, W_T, 1728, narrow) \
    f(in_T, out_T, W_T, 1792, narrow) \
    f(in_T, out_T, W_T, 2048, narrow) \
    f(in_T, out_T, W_T, 2240, narrow) \
    f(in_T, out_T, W_T, 2304, narrow) \
    f(in_T, out_T, W_T, 2368, narrow) \
    f(in_T, out_T, W_T, 2432, narrow) \
    f(in_T, out_T, W_T, 2560, narrow) \
    f(in_T, out_T, W_T, 2752, narrow) \
    f(in_T, out_T, W_T, 2816, narrow) \
    f(in_T, out_T, W_T, 3072, narrow) \
    f(in_T, out_T, W_T, 3328, narrow) \
    f(in_T, out_T, W_T, 3456, narrow) \
    f(in_T, out_T, W_T, 3584, narrow) \
    f(in_T, out_T, W_T, 3712, narrow) \
    f(in_T, out_T, W_T, 4096, narrow) \
    f(in_T, out_T, W_T, 4480, narrow) \
    f(in_T, out_T, W_T, 4608, narrow) \
    f(in_T, out_T, W_T, 4736, narrow) \
    f(in_T, out_T, W_T, 4864, narrow) \
    f(in_T, out_T, W_T, 5120, narrow) \
    f(in_T, out_T, W_T, 5504, narrow) \
    f(in_T, out_T, W_T, 5632, narrow) \
    f(in_T, out_T, W_T, 5888, narrow) \
    f(in_T, out_T, W_T, 6144, narrow) \
    f(in_T, out_T, W_T, 6400, narrow) \
    f(in_T, out_T, W_T, 6848, narrow) \
    f(in_T, out_T, W_T, 6912, narrow) \
    f(in_T, out_T, W_T, 7168, narrow) \
    f(in_T, out_T, W_T, 7424, narrow) \
    f(in_T, out_T, W_T, 8192, narrow) \
    f(in_T, out_T, W_T, 8960, narrow) \
    f(in_T, out_T, W_T, 9216, narrow) \
    f(in_T, out_T, W_T, 9472, narrow) \
    f(in_T, out_T, W_T, 10240, narrow) \
    f(in_T, out_T, W_T, 11008, narrow) \
    f(in_T, out_T, W_T, 11264, narrow) \
    f(in_T, out_T, W_T, 12288, narrow) \
    f(in_T, out_T, W_T, 13696, narrow) \
    f(in_T, out_T, W_T, 13824, narrow) \
    f(in_T, out_T, W_T, 14336, narrow) \
    f(in_T, out_T, W_T, 14784, narrow) \
    f(in_T, out_T, W_T, 14848, narrow) \
    f(in_T, out_T, W_T, 15360, narrow) \
    f(in_T, out_T, W_T, 16384, narrow) \
    f(in_T, out_T, W_T, 18944, narrow) \
    f(in_T, out_T, W_T, 20480, narrow) \
    f(in_T, out_T, W_T, 22016, narrow) \
    f(in_T, out_T, W_T, 22528, narrow) \
    f(in_T, out_T, W_T, 24576, narrow) \
    f(in_T, out_T, W_T, 27392, narrow) \
    f(in_T, out_T, W_T, 27648, narrow) \
    f(in_T, out_T, W_T, 28672, narrow) \
    f(in_T, out_T, W_T, 29568, narrow) \
    f(in_T, out_T, W_T, 29696, narrow) \
    f(in_T, out_T, W_T, 32000, narrow) \
    f(in_T, out_T, W_T, 32256, narrow) \
    f(in_T, out_T, W_T, 32512, narrow) \
    f(in_T, out_T, W_T, 32768, narrow) \
    f(in_T, out_T, W_T, 33024, narrow) \
    f(in_T, out_T, W_T, 36864, narrow) \
    f(in_T, out_T, W_T, 43264, narrow) \
    f(in_T, out_T, W_T, 49152, narrow) \
    f(in_T, out_T, W_T, 60544, narrow) \
    f(in_T, out_T, W_T, 60672, narrow) \
    f(in_T, out_T, W_T, 64000, narrow) \
    f(in_T, out_T, W_T, 64256, narrow) \
    f(in_T, out_T, W_T, 64512, narrow) \
    f(in_T, out_T, W_T, 102400, narrow) \
    f(in_T, out_T, W_T, 102656, narrow) \
    f(in_T, out_T, W_T, 102912, narrow) \
    f(in_T, out_T, W_T, 128000, narrow) \
    f(in_T, out_T, W_T, 128256, narrow) \
    f(in_T, out_T, W_T, 128512, narrow) \
// Keep above in sync with vllm/lora/layers::SamplerWithLoRA


// Keep this in sync with vllm/config::LoRAConfig
#define FOR_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 64)


#define FOR_INST_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 1) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 2) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 4) \
    f(in_T, out_T, W_T, 8, 64) \
    f(in_T, out_T, W_T, 16, 64) \
    f(in_T, out_T, W_T, 32, 64) \
    f(in_T, out_T, W_T, 64, 64)

// clang-format on
