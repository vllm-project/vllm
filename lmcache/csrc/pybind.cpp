// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mem_kernels.cuh"
#include "mp_mem_kernels.cuh"
#include "blend_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "utils.h"
#include "event_recorder.h"
#include "completion_recorder.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<EngineKVFormat>(m, "EngineKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", EngineKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", EngineKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", EngineKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", EngineKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_NH_BS_TWO_HS", EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)
      .value("NL_X_NB_BS_NH_TWO_HS", EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)
      .value("NL_X_NB_NH_BS_CS", EngineKVFormat::NL_X_NB_NH_BS_CS)
      .value("NL_X_NB_BS_NH_CS", EngineKVFormat::NL_X_NB_BS_NH_CS)
      .value("NL_X_NB_BSV_BSS", EngineKVFormat::NL_X_NB_BSV_BSS)
      .export_values();
  // Format classification, shared with the device kernels (engine_kv_format.h).
  m.def(
      "is_cross_layer", [](EngineKVFormat f) { return is_cross_layer(f); },
      py::arg("engine_kv_format"));
  m.def(
      "is_kv_list", [](EngineKVFormat f) { return is_kv_list(f); },
      py::arg("engine_kv_format"));
  m.def(
      "is_layer_list", [](EngineKVFormat f) { return is_layer_list(f); },
      py::arg("engine_kv_format"));
  m.def(
      "is_mla", [](EngineKVFormat f) { return is_mla(f); },
      py::arg("engine_kv_format"));
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("engine_kv_format"), py::arg("block_size") = 0,
        py::arg("head_size") = 0, py::arg("skip_prefix_n_tokens") = 0,
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"),
        py::arg("engine_kv_format"), py::arg("token_major") = false);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl,
        py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
        py::arg("sgl_value_cache"), py::arg("slot_mapping"),
        py::arg("direction"), py::arg("token_major") = false);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("rotary_embedding_k_fused_strided", &rotary_embedding_k_fused_strided);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_hugepage_pinned_ptr", &alloc_hugepage_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_hugepage_pinned_ptr", &free_hugepage_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("alloc_hugepage_pinned_numa_ptr", &alloc_hugepage_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_hugepage_pinned_numa_ptr", &free_hugepage_pinned_numa_ptr);
  m.def("alloc_numa_ptr", &alloc_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_numa_ptr", &free_numa_ptr);
  m.def("alloc_shm_pinned_ptr", &alloc_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_shm_pinned_ptr", &free_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("batched_memcpy", &batched_memcpy, py::arg("src_ptrs"),
        py::arg("dst_ptrs"), py::arg("sizes"),
        py::call_guard<py::gil_scoped_release>());
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
  m.def("multi_layer_block_kv_transfer", &multi_layer_block_kv_transfer,
        py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
        py::arg("block_ids"), py::arg("device"), py::arg("direction"),
        py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
        py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"),
        py::call_guard<py::gil_scoped_release>());
  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc")
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems);
  // Object-group transfer plan types (see mp_mem_kernels.cuh). Built on the
  // Python side and consumed by execute_object_group_transfer.
  py::class_<StagingCopy>(m, "StagingCopy")
      .def(py::init([](uintptr_t dest, uintptr_t src, size_t nbytes,
                       size_t host_offset) {
             return StagingCopy{dest, src, nbytes, host_offset};
           }),
           py::arg("dest"), py::arg("src"), py::arg("nbytes"),
           py::arg("host_offset"));
  py::class_<LaunchVar>(m, "LaunchVar")
      .def(
          py::init([](int group_idx, int64_t block_ids_offset, int total_blocks,
                      int num_objects, int skip_prefix_n_blocks) {
            return LaunchVar{group_idx, block_ids_offset, total_blocks,
                             num_objects, skip_prefix_n_blocks};
          }),
          py::arg("group_idx"), py::arg("block_ids_offset"),
          py::arg("total_blocks"), py::arg("num_objects"),
          py::arg("skip_prefix_n_blocks"));
  py::class_<BatchStep>(m, "BatchStep")
      .def(py::init([](std::vector<StagingCopy> staging,
                       std::vector<LaunchVar> launches) {
             return BatchStep{std::move(staging), std::move(launches)};
           }),
           py::arg("staging"), py::arg("launches"));
  py::class_<KernelGroupSpec>(m, "KernelGroupSpec")
      .def(py::init([](uintptr_t paged_buffer_ptrs,
                       std::vector<int64_t> lmcache_objects_ptrs,
                       PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
                       EngineKVFormat engine_kv_format,
                       uintptr_t block_ids_base, int64_t block_ids_capacity) {
             return KernelGroupSpec{
                 paged_buffer_ptrs, std::move(lmcache_objects_ptrs),
                 shape_desc,        lmcache_chunk_size,
                 engine_kv_format,  block_ids_base,
                 block_ids_capacity};
           }),
           py::arg("paged_buffer_ptrs"), py::arg("lmcache_objects_ptrs"),
           py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
           py::arg("engine_kv_format"), py::arg("block_ids_base"),
           py::arg("block_ids_capacity"));
  m.def("execute_object_group_transfer", &execute_object_group_transfer,
        py::arg("direction"), py::arg("device"),
        py::arg("host_buffer_alignment"), py::arg("kernel_group_specs"),
        py::arg("batch_steps"), py::call_guard<py::gil_scoped_release>());
  // CB retrieve plan spec (see blend_kernels.cuh). Built on the Python side
  // (blend_v3.cb_retrieve_pre_computed) and consumed by
  // execute_cb_retrieve_plan_flat.
  py::class_<CBGroupSpec>(m, "CBGroupSpec")
      .def(py::init(
               [](uintptr_t paged_kv_ptrs,
                  std::vector<int64_t> temp_buffer_ptrs, int num_layers,
                  int slot_tokens, int hidden_elems, int element_size,
                  EngineKVFormat engine_kv_format, int page_buffer_size,
                  int block_size, int head_size, uintptr_t slot_mapping_base,
                  int64_t slot_mapping_capacity, uintptr_t cos_sin_cache,
                  int rot_dim, int rope_num_kv_heads, int64_t rope_head_stride,
                  int key_scalar_type, bool is_neox, int64_t rope_base_offset) {
                 return CBGroupSpec{
                     paged_kv_ptrs,     std::move(temp_buffer_ptrs),
                     num_layers,        slot_tokens,
                     hidden_elems,      element_size,
                     engine_kv_format,  page_buffer_size,
                     block_size,        head_size,
                     slot_mapping_base, slot_mapping_capacity,
                     cos_sin_cache,     rot_dim,
                     rope_num_kv_heads, rope_head_stride,
                     key_scalar_type,   is_neox,
                     rope_base_offset};
               }),
           py::arg("paged_kv_ptrs"), py::arg("temp_buffer_ptrs"),
           py::arg("num_layers"), py::arg("slot_tokens"),
           py::arg("hidden_elems"), py::arg("element_size"),
           py::arg("engine_kv_format"), py::arg("page_buffer_size"),
           py::arg("block_size"), py::arg("head_size"),
           py::arg("slot_mapping_base"), py::arg("slot_mapping_capacity"),
           py::arg("cos_sin_cache"), py::arg("rot_dim"),
           py::arg("rope_num_kv_heads"), py::arg("rope_head_stride"),
           py::arg("key_scalar_type"), py::arg("is_neox"),
           py::arg("rope_base_offset") = 0)
      // Mutable so the Python planner can cache one spec per (context, group)
      // and re-stamp only the per-request slot-mapping tensor between calls;
      // every other field is invariant for the life of the paged registration.
      .def_readwrite("slot_mapping_base", &CBGroupSpec::slot_mapping_base)
      .def_readwrite("slot_mapping_capacity",
                     &CBGroupSpec::slot_mapping_capacity);
  // The whole plan arrives as four int64 numpy tables and is unpacked into
  // CBRetrieveStep vectors in C++ (microseconds), replacing ~5 pybind object
  // constructions per chunk on the Python side. Tables:
  //   staging  [n_staging, 4]: dest, src, nbytes, host_offset
  //   ropes    [n_ropes,  4]: group_idx, slot_idx, old_st, cur_st
  //   scatters [n_scatter,4]: group_idx, slot_idx, slot_mapping_offset, n_tok
  //   step_offsets [n_steps, 3]: end index into each table (exclusive,
  //   cumulative) after this step -- i.e. a per-kind CSR without the leading 0.
  m.def(
      "execute_cb_retrieve_plan_flat",
      [](const torch::Device& device, size_t host_buffer_alignment,
         const std::vector<CBGroupSpec>& group_specs,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast>
             staging,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast> ropes,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast>
             scatters,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast>
             step_offsets) {
        TORCH_CHECK(staging.ndim() == 2 && staging.shape(1) == 4,
                    "staging table must be [n, 4]");
        TORCH_CHECK(ropes.ndim() == 2 && ropes.shape(1) == 4,
                    "ropes table must be [n, 4]");
        TORCH_CHECK(scatters.ndim() == 2 && scatters.shape(1) == 4,
                    "scatters table must be [n, 4]");
        TORCH_CHECK(step_offsets.ndim() == 2 && step_offsets.shape(1) == 3,
                    "step_offsets table must be [n_steps, 3]");
        const int64_t* st = staging.data();
        const int64_t* rp = ropes.data();
        const int64_t* sc = scatters.data();
        const int64_t* so = step_offsets.data();
        const int64_t n_steps = step_offsets.shape(0);

        std::vector<CBRetrieveStep> steps;
        steps.reserve(n_steps);
        int64_t s0 = 0, r0 = 0, c0 = 0;
        for (int64_t i = 0; i < n_steps; ++i) {
          const int64_t s1 = so[i * 3 + 0];
          const int64_t r1 = so[i * 3 + 1];
          const int64_t c1 = so[i * 3 + 2];
          TORCH_CHECK(s1 >= s0 && s1 <= staging.shape(0) && r1 >= r0 &&
                          r1 <= ropes.shape(0) && c1 >= c0 &&
                          c1 <= scatters.shape(0),
                      "step_offsets not monotone or out of range at step ", i);
          CBRetrieveStep step;
          step.staging.reserve(s1 - s0);
          for (int64_t j = s0; j < s1; ++j) {
            step.staging.push_back(
                StagingCopy{static_cast<uintptr_t>(st[j * 4 + 0]),
                            static_cast<uintptr_t>(st[j * 4 + 1]),
                            static_cast<size_t>(st[j * 4 + 2]),
                            static_cast<size_t>(st[j * 4 + 3])});
          }
          step.ropes.reserve(r1 - r0);
          for (int64_t j = r0; j < r1; ++j) {
            step.ropes.push_back(CBRopeVar{static_cast<int>(rp[j * 4 + 0]),
                                           static_cast<int>(rp[j * 4 + 1]),
                                           rp[j * 4 + 2], rp[j * 4 + 3]});
          }
          step.scatters.reserve(c1 - c0);
          for (int64_t j = c0; j < c1; ++j) {
            step.scatters.push_back(
                CBScatterVar{static_cast<int>(sc[j * 4 + 0]),
                             static_cast<int>(sc[j * 4 + 1]), sc[j * 4 + 2],
                             static_cast<int>(sc[j * 4 + 3])});
          }
          steps.push_back(std::move(step));
          s0 = s1;
          r0 = r1;
          c0 = c1;
        }
        {
          py::gil_scoped_release release;
          execute_cb_retrieve_plan(device, host_buffer_alignment, group_specs,
                                   steps);
        }
      },
      py::arg("device"), py::arg("host_buffer_alignment"),
      py::arg("group_specs"), py::arg("staging"), py::arg("ropes"),
      py::arg("scatters"), py::arg("step_offsets"));
  m.def("record_event_on_stream", &record_event_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("event_type_name"),
        py::arg("session_id"), py::arg("str_metadata"), py::arg("int_metadata"),
        py::call_guard<py::gil_scoped_release>());
  m.def("drain_recorded_events", &drain_recorded_events);
  m.def("record_completion_on_stream", &record_completion_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("kind"), py::arg("payload"),
        py::call_guard<py::gil_scoped_release>());
  // Return each payload as py::bytes; pybind11 utf-8-decodes std::string
  // by default, corrupting binary payloads (e.g. msgpack).
  m.def("drain_recorded_completions", []() {
    auto items = drain_recorded_completions();
    py::list out;
    for (auto& kv : items) {
      out.append(py::make_tuple(py::str(kv.first), py::bytes(kv.second)));
    }
    return out;
  });
  // Backward-compat alias: GPUKVFormat -> EngineKVFormat
  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");
}
