#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <tuple>
#include <cstdint>
#include <iostream>
#include <memory>
#include <chrono>
#include "moe.h"
#include "primitives.h"


torch::Tensor
cpu_moe_sync(torch::Tensor output, int64_t moe_engine_ptr) {
    auto* moe_engine = reinterpret_cast<MoeOffloadEngine*>(moe_engine_ptr);
    moe_engine->sync();
    moe_engine->get_output(output);
    return output;
}

torch::Tensor
cpu_moe_sync_meta(torch::Tensor output, int64_t moe_engine_ptr){
    return output;
}

void cpu_moe_submit(torch::Tensor hidden_states, torch::Tensor topk_ids, torch::Tensor topk_weights,
                int64_t moe_engine_ptr, int64_t layer_id, int64_t batch_idx) {
    auto* moe_engine = reinterpret_cast<MoeOffloadEngine*>(moe_engine_ptr);
    moe_engine->set_input(hidden_states, topk_ids, topk_weights);
    moe_engine->submit(layer_id, batch_idx, hidden_states.size(0));
}

void cpu_moe_submit_meta(torch::Tensor hidden_states, torch::Tensor topk_ids, torch::Tensor topk_weights,
                int64_t moe_engine_ptr, int64_t layer_id, int64_t batch_idx) {

}


std::tuple<torch::Tensor, torch::Tensor>
expert_cache_policy(torch::Tensor& topk_ids, torch::Tensor& cache_map,
                    torch::Tensor& miss_map, torch::Tensor& policy_sort,
                    int64_t moe_engine_ptr){
    auto* moe_engine = reinterpret_cast<MoeOffloadEngine*>(moe_engine_ptr);
    auto copy_map = torch::zeros_like(cache_map);
    auto cpu_topk = torch::zeros_like(topk_ids);
    moe_engine->expert_cache_policy(cache_map, miss_map, policy_sort, topk_ids,
                             cpu_topk, copy_map);
    return std::make_tuple(cpu_topk, copy_map);
}

std::tuple<torch::Tensor, torch::Tensor>
expert_cache_policy_meta(torch::Tensor& topk_ids, torch::Tensor& cache_map,
                    torch::Tensor& miss_map, torch::Tensor& policy_sort,
                    int64_t moe_engine_ptr){
    auto copy_map = torch::zeros_like(cache_map);
    auto cpu_topk = torch::zeros_like(topk_ids);
    return std::make_tuple(cpu_topk, copy_map);
}

void update_expert_cache(
    torch::Tensor w13_cache, torch::Tensor w2_cache,
    torch::Tensor w13_scale_cache, torch::Tensor w2_scale_cache,
    torch::Tensor map, int64_t num_experts, int64_t layer_id, int64_t moe_engine_ptr) {

    auto* moe_engine = reinterpret_cast<MoeOffloadEngine*>(moe_engine_ptr);
    moe_engine->update_expert_cache(w13_cache, w2_cache, w13_scale_cache,
                            w2_scale_cache, map, layer_id, num_experts);
}

void update_expert_cache_meta(
    torch::Tensor w13_cache, torch::Tensor w2_cache,
    torch::Tensor w13_scale_cache, torch::Tensor w2_scale_cache,
    torch::Tensor map, int64_t num_experts, int64_t layer_id, int64_t moe_engine_ptr) {
}

namespace py = pybind11;

// ========== 算子注册 ==========
TORCH_LIBRARY(moe_offload_ops, m) {
    m.def("expert_cache_policy(Tensor topk_ids, Tensor cache_map, Tensor miss_map, "
          "Tensor policy_sort, int moe_engine_ptr) -> (Tensor, Tensor)");

    m.def("update_expert_cache(Tensor w13_cache, Tensor w2_cache, "
          "Tensor w13_scale_cache, Tensor w2_scale_cache, Tensor map, "
          "int num_experts, int layer_id, int moe_engine_ptr) -> ()");

    m.def("cpu_moe_submit(Tensor hidden_states, Tensor topk_ids, Tensor topk_weights, "
          "int moe_engine_ptr, int layer_id, int batch_idx) -> ()");

    m.def("cpu_moe_sync(Tensor(a!) output, int moe_engine_ptr) -> Tensor");
}

// ========== Python绑定 ==========
PYBIND11_MODULE(_offload_C, m) {
    m.doc() = "MoE Offload Engine (Minimal C++ Interface)";

    py::class_<MOEConfig>(m, "MOEConfig")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int, int>(),
             py::arg("tp_rank"), py::arg("tp_size"), py::arg("expert_num"),
             py::arg("num_experts_per_tok"), py::arg("hidden_size"),
             py::arg("intermediate_size"), py::arg("max_batch_token"),
             py::arg("cache_expert_num"), py::arg("block_size"),
             py::arg("cache_topk"), py::arg("update_expert_num"),
             py::arg("forward_context_num_threads") = 14);

    py::class_<Moe>(m, "Moe")
        .def(py::init([](uint64_t w13_weights_ptr, uint64_t w2_weights_ptr,
                         uint64_t w13_scales_ptr, uint64_t w2_scales_ptr,
                         int layer_id,
                         MOEConfig config) {
            return new Moe(
                reinterpret_cast<float8_e4m3_t*>(w13_weights_ptr),
                reinterpret_cast<float8_e4m3_t*>(w2_weights_ptr),
                reinterpret_cast<float*>(w13_scales_ptr),
                reinterpret_cast<float*>(w2_scales_ptr),
                layer_id,
                config);
        }), py::arg("w13_weights_ptr"), py::arg("w2_weights_ptr"),
           py::arg("w13_scales_ptr"), py::arg("w2_scales_ptr"),
           py::arg("layer_id"),
           py::arg("config"))

        .def("forward", [](Moe& self, uint64_t input_ptr, uint64_t topk_ids_ptr,
                          uint64_t topk_weights_ptr, uint64_t output_ptr,
                          int num_tokens) {

            self.forward(
                reinterpret_cast<bfloat16_t*>(input_ptr),
                reinterpret_cast<int*>(topk_ids_ptr),
                reinterpret_cast<float*>(topk_weights_ptr),
                reinterpret_cast<bfloat16_t*>(output_ptr),
                num_tokens);
            
        }, py::arg("input_ptr"), py::arg("topk_ids_ptr"),
           py::arg("topk_weights_ptr"), py::arg("output_ptr"),
           py::arg("num_tokens"));


    py::class_<MoeOffloadEngine>(m, "MoeOffloadEngine")
           .def(py::init<MOEConfig>())
           .def("create_layer", &MoeOffloadEngine::create_cpu_moe_layer)
           .def("ptr", &MoeOffloadEngine::ptr);

    m.def("set_tiledata_use", &set_tiledata_use, "Enable AMX-Tile feature");
}

TORCH_LIBRARY_IMPL(moe_offload_ops, CUDA, m) {
    m.impl("update_expert_cache",   &update_expert_cache);
    m.impl("expert_cache_policy",   &expert_cache_policy);
    m.impl("cpu_moe_submit",        &cpu_moe_submit);
    m.impl("cpu_moe_sync",          &cpu_moe_sync);
}

// ==========  Meta / FakeTensor / Dynamo  ==========
TORCH_LIBRARY_IMPL(moe_offload_ops, Meta, m) {
    m.impl("update_expert_cache",   &update_expert_cache_meta);
    m.impl("expert_cache_policy",   &expert_cache_policy_meta);
    m.impl("cpu_moe_submit",        &cpu_moe_submit_meta);
    m.impl("cpu_moe_sync",          &cpu_moe_sync_meta);
}

// CompositeExplicitAutogradNonFunctional
TORCH_LIBRARY_IMPL(moe_offload_ops, CompositeExplicitAutogradNonFunctional, m) {
    m.impl("update_expert_cache",   &update_expert_cache_meta);
    m.impl("expert_cache_policy",   &expert_cache_policy_meta);
    m.impl("cpu_moe_submit",        &cpu_moe_submit_meta);
    m.impl("cpu_moe_sync",          &cpu_moe_sync_meta);
}
