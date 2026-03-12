#pragma once
#include "tensorrt_llm/common/assert.h"
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/runtime/ipcUtils.h"


namespace kernels::minimax_ar
{
template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half>
{
    static constexpr int value = 8;
    using norm_weight_type = common::__nv_bfloat168;
};

template <>
struct ElemsPerAccess<nv_bfloat16>
{
    static constexpr int value = 8;
    using norm_weight_type = common::__nv_bfloat168;
};

template <>
struct ElemsPerAccess<float>
{
    static constexpr int value = 4;
    using norm_weight_type = common::__nv_bfloat164;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

struct MiniMaxReduceRMSParams
{
    int nranks{};
    int rank{};
    nvinfer1::DataType dtype;
    int size_q{};           // numel of Q (num_token * head_dim_q)
    int hidden_dim{};       // head_dim_q
    int size_k{};           // numel of K (num_token * head_dim_k)
    int hidden_dim_k{};     // head_dim_k; must have head_dim_q >= head_dim_k
    void** workspace{};
    void* allreduce_in{};   // Q input
    void* rms_norm_out{};   // Q output
    void* rms_gamma{};      // Q norm weight
    void* allreduce_in_k{}; // K input (nullptr for single-matrix path)
    void* rms_norm_out_k{}; // K output
    void* rms_gamma_k{};    // K norm weight
    float rms_eps{};
    cudaStream_t stream{};
    bool trigger_completion_at_end = true;
};

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params);

} // namespace kernels::minimax_ar


