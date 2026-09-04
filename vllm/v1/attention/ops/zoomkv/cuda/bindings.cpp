#include <torch/python.h>

#include <optional>

at::Tensor float_topk_cuda(at::Tensor input, int64_t k);
at::Tensor float_topk_3d_cuda(at::Tensor input, int64_t k);
at::Tensor float_topk_values_3d_cuda(at::Tensor input, at::Tensor values,
                                    int64_t k,
                                    std::optional<at::Tensor> output);
at::Tensor float_topk_3d_varlen_cuda(at::Tensor input, at::Tensor lengths,
                                    at::Tensor ks, int64_t max_k);
at::Tensor float_topk_values_3d_varlen_cuda(
    at::Tensor input, at::Tensor values, at::Tensor lengths, at::Tensor ks,
    int64_t max_k, std::optional<at::Tensor> output);

void partial_chunk_density_scores_interface(at::Tensor chunk_ids,
                                            at::Tensor chunk_centroids,
                                            at::Tensor raw_q,
                                            at::Tensor out_scores);

void mask_from_topk_interface(at::Tensor positions, at::Tensor mask);

void density_score_physical_cuda(
    at::Tensor chunk_ids, at::Tensor physical_ids, at::Tensor global_centroid,
    at::Tensor global_valid, at::Tensor q, at::Tensor scores,
    int64_t n_chunks, std::optional<at::Tensor> actual_num_chunks);
void centroid_score_physical_cuda(
    at::Tensor physical_ids, at::Tensor global_centroid, at::Tensor global_valid,
    at::Tensor q, at::Tensor scores, int64_t n_chunks,
    std::optional<at::Tensor> actual_num_chunks);
void kivi_physical_cuda(
    at::Tensor chunk_ids, at::Tensor dense_mask, at::Tensor physical_ids,
    at::Tensor global_packed, at::Tensor global_min, at::Tensor global_max,
    at::Tensor global_valid, at::Tensor q, int64_t dense_topk,
    int64_t sparse_topk, int64_t token_offset, at::Tensor out_scores,
    at::Tensor out_indices, std::optional<at::Tensor> actual_num_chunks,
    bool compact = false, int64_t n_dense = 0);

void partial_chunk_kivi_qk_dense_sparse_interface(
    at::Tensor chunk_ids, at::Tensor dense_mask, at::Tensor packed_k,
    at::Tensor chunk_min, at::Tensor chunk_max, at::Tensor raw_q,
    int dense_topk, int sparse_topk, int group_size, at::Tensor out_scores,
    at::Tensor out_indices);

void h2d_gather_keys(const at::Tensor& src_k, const at::Tensor& slots,
                     const at::Tensor& offsets, at::Tensor& out_k);

void h2d_gather_keys_hybrid(const at::Tensor& src_k,
                            const at::Tensor& logical_ids,
                            const at::Tensor& block_table,
                            const at::Tensor& cpu_slots,
                            const at::Tensor& offloaded_mask,
                            int64_t start_block, at::Tensor& out_k);

void h2d_gather_kv_hybrid(const at::Tensor& src_k, const at::Tensor& src_v,
                          const at::Tensor& gpu_k, const at::Tensor& gpu_v,
                          const at::Tensor& logical_ids,
                          const at::Tensor& block_table,
                          const at::Tensor& physical_to_slot,
                          const at::Tensor& offloaded_mask, at::Tensor& out_k,
                          at::Tensor& out_v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("float_topk", &float_topk_cuda);
  m.def("float_topk_3d", &float_topk_3d_cuda);
  m.def("float_topk_values_3d", &float_topk_values_3d_cuda,
        py::arg("input"), py::arg("values"), py::arg("k"),
        py::arg("output") = std::nullopt);
  m.def("float_topk_3d_varlen", &float_topk_3d_varlen_cuda,
        py::arg("input"), py::arg("lengths"), py::arg("ks"),
        py::arg("max_k"));
  m.def("float_topk_values_3d_varlen",
        &float_topk_values_3d_varlen_cuda, py::arg("input"),
        py::arg("values"), py::arg("lengths"), py::arg("ks"),
        py::arg("max_k"), py::arg("output") = std::nullopt);
  m.def("partial_chunk_density_scores",
        &partial_chunk_density_scores_interface);
  m.def("mask_from_topk", &mask_from_topk_interface);
  m.def("density_score_physical", &density_score_physical_cuda,
        py::arg("chunk_ids"), py::arg("physical_ids"),
        py::arg("global_centroid"), py::arg("global_valid"), py::arg("q"),
        py::arg("scores"), py::arg("n_chunks"),
        py::arg("actual_num_chunks") = std::nullopt);
  m.def("centroid_score_physical", &centroid_score_physical_cuda,
        py::arg("physical_ids"), py::arg("global_centroid"),
        py::arg("global_valid"), py::arg("q"), py::arg("scores"),
        py::arg("n_chunks"), py::arg("actual_num_chunks") = std::nullopt);
  m.def("kivi_physical", &kivi_physical_cuda, py::arg("chunk_ids"),
        py::arg("dense_mask"), py::arg("physical_ids"),
        py::arg("global_packed"), py::arg("global_min"),
        py::arg("global_max"), py::arg("global_valid"), py::arg("q"),
        py::arg("dense_topk"), py::arg("sparse_topk"),
        py::arg("token_offset"), py::arg("out_scores"),
        py::arg("out_indices"),
        py::arg("actual_num_chunks") = std::nullopt,
        py::arg("compact") = false, py::arg("n_dense") = 0);
  m.def("partial_chunk_kivi_qk_dense_sparse",
        &partial_chunk_kivi_qk_dense_sparse_interface);
  m.def("h2d_gather_keys", &h2d_gather_keys);
  m.def("h2d_gather_keys_hybrid", &h2d_gather_keys_hybrid);
  m.def("h2d_gather_kv_hybrid", &h2d_gather_kv_hybrid);
}
