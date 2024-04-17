#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

void init_shm_manager(const std::string &ip_port,
                      const int group_size, const int rank,
                      const size_t rank_buffer_size);

std::string join_shm_manager(const std::string &ip_port,
                      const int group_size, const int rank,
                      const size_t rank_buffer_size);

void shm_allreduce(torch::Tensor &data, int rank);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def("paged_attention_v1", &paged_attention_v1,
          "Compute the attention between an input query and the cached "
          "keys/values using PagedAttention.");
  ops.def("paged_attention_v2", &paged_attention_v2, "PagedAttention V2.");

  // Activation ops
  ops.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
  ops.def("gelu_and_mul", &gelu_and_mul,
          "Activation function used in GeGLU with `none` approximation.");
  ops.def("gelu_tanh_and_mul", &gelu_tanh_and_mul,
          "Activation function used in GeGLU with `tanh` approximation.");
  ops.def("gelu_new", &gelu_new, "GELU implementation used in GPT-2.");
  ops.def("gelu_fast", &gelu_fast, "Approximate GELU implementation.");

  // Layernorm
  ops.def("rms_norm", &rms_norm,
          "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def("fused_add_rms_norm", &fused_add_rms_norm,
          "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def("rotary_embedding", &rotary_embedding,
          "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");

  // SHM CCL
  ops.def(
    "init_shm_manager",
    &init_shm_manager,
    "Initialize shared memory collective communication manager."); 
  ops.def(
    "join_shm_manager",
    &join_shm_manager,
    "Exchange shared memory collective communication managers between ranks."); 
  ops.def(
    "shm_allreduce",
    &shm_allreduce,
    "SHM based sum AllReduce operation."); 
}
