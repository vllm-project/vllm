#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "format.h"
#include "gemm_s4_f16.h"


// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now


void autoquant_convert_s4_k_m8(
  torch::Tensor _weight_dest,
  torch::Tensor _quant_scales_zeros_dest,
  torch::Tensor _workspace,
  torch::Tensor _quant_weight_src,
  torch::Tensor _quant_scales,
  torch::Tensor _quant_zeros,
  int m,
  int k,
  int group_size){
      auto st_ = _quant_scales.scalar_type();
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      if(st_ == at::ScalarType::Half){
              auto weight_dest = reinterpret_cast<uint32_t*>(_weight_dest.data_ptr<int32_t>());
              auto quant_scales_zeros_dest = reinterpret_cast<half2*>(_quant_scales_zeros_dest.data_ptr<int32_t>());
              auto workspace = reinterpret_cast<half*>(_workspace.data_ptr<at::Half>());
              auto quant_weight_src = reinterpret_cast<uint32_t*>(_quant_weight_src.data_ptr<int32_t>());
              auto quant_scales = reinterpret_cast<half*>(_quant_scales.data_ptr<at::Half>());
              auto quant_zeros = reinterpret_cast<uint32_t*>(_quant_zeros.data_ptr<int32_t>());
              vllm::autoquant::convert_s4_k_m8(weight_dest, quant_scales_zeros_dest, workspace, quant_weight_src, quant_scales, quant_zeros,
                            m, k, group_size, stream);
      }
      else{
              auto weight_dest = reinterpret_cast<uint32_t*>(_weight_dest.data_ptr<int32_t>());
              auto quant_scales_zeros_dest = reinterpret_cast<__nv_bfloat162*>(_quant_scales_zeros_dest.data_ptr<int32_t>());
              auto workspace = reinterpret_cast<__nv_bfloat16*>(_workspace.data_ptr<at::BFloat16>());
              auto quant_weight_src = reinterpret_cast<uint32_t*>(_quant_weight_src.data_ptr<int32_t>());
              auto quant_scales = reinterpret_cast<__nv_bfloat16*>(_quant_scales.data_ptr<at::BFloat16>());
              auto quant_zeros = reinterpret_cast<uint32_t*>(_quant_zeros.data_ptr<int32_t>());
              vllm::autoquant::convert_s4_k_m8(weight_dest, quant_scales_zeros_dest, workspace, quant_weight_src, quant_scales, quant_zeros,
                            m, k, group_size, stream);
      }
}


torch::Tensor autoquant_s4_f16_gemm(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales_zeros)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(1) * 8}, options);

    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto st_ = _in_feats.scalar_type();
    if(st_ == at::ScalarType::Half){
        static vllm::autoquant::GemmS4F16<half, half2> gemm_s4_f16_;
        auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
        auto kernel = reinterpret_cast<const uint*>(_kernel.data_ptr<int32_t>());
        auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
        auto scales_zeros = reinterpret_cast<half2*>(_scales_zeros.data_ptr<int32_t>());
        int group_size = num_in_channels / _scales_zeros.size(0);
    
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gemm_s4_f16_.Run(out_feats,
                         kernel,
                         in_feats,
                         scales_zeros,
                         num_out_channels,
                         num_in_feats,
                         num_in_channels,
                         group_size,
                         vllm::autoquant::kGemm,
                         -1,
                         stream);
        return _out_feats;
    }
    else{
        static vllm::autoquant::GemmS4F16<__nv_bfloat16, __nv_bfloat162> gemm_s4_bf16_;
        auto in_feats = reinterpret_cast<__nv_bfloat16*>(_in_feats.data_ptr<at::BFloat16>());
        auto kernel = reinterpret_cast<const uint*>(_kernel.data_ptr<int32_t>());
        auto out_feats = reinterpret_cast<__nv_bfloat16*>(_out_feats.data_ptr<at::BFloat16>());
        auto scales_zeros = reinterpret_cast<__nv_bfloat162*>(_scales_zeros.data_ptr<int32_t>());
        int group_size = num_in_channels / _scales_zeros.size(0);
    
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gemm_s4_bf16_.Run(out_feats,
                         kernel,
                         in_feats,
                         scales_zeros,
                         num_out_channels,
                         num_in_feats,
                         num_in_channels,
                         group_size,
                         vllm::autoquant::kGemm,
                         -1,
                         stream);
        return _out_feats;   
    }
}
