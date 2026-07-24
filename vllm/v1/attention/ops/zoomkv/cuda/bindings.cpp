#include <torch/extension.h>

#include <optional>

torch::Tensor float_topk_cuda(torch::Tensor input, int64_t k);
torch::Tensor float_topk_3d_cuda(torch::Tensor input, int64_t k);

void quest_chunk_score_cuda(at::Tensor q, at::Tensor chunk_min,
                            at::Tensor chunk_max, at::Tensor scores,
                            int64_t n_chunks, std::optional<at::Tensor> valid);

void quest_sub_chunk_score_cuda(at::Tensor q, at::Tensor chunk_min,
                                at::Tensor chunk_max, at::Tensor large_ids,
                                at::Tensor scores, int64_t n_selected,
                                int64_t factor);

void quest_map_back_cuda(at::Tensor large_ids, at::Tensor sub_topk_pos,
                         at::Tensor chunk_idx, int64_t factor,
                         int64_t n_chunks);

void partial_chunk_density_scores_interface(at::Tensor chunk_ids,
                                            at::Tensor chunk_centroids,
                                            at::Tensor raw_q,
                                            at::Tensor out_scores);

void mask_from_topk_interface(at::Tensor positions, at::Tensor mask);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("float_topk", &float_topk_cuda);
  m.def("float_topk_3d", &float_topk_3d_cuda);
  m.def("quest_chunk_score", &quest_chunk_score_cuda, py::arg("q"),
        py::arg("chunk_min"), py::arg("chunk_max"), py::arg("scores"),
        py::arg("n_chunks"), py::arg("chunk_valid") = std::nullopt);
  m.def("quest_sub_chunk_score", &quest_sub_chunk_score_cuda);
  m.def("quest_map_back", &quest_map_back_cuda);
  m.def("partial_chunk_density_scores",
        &partial_chunk_density_scores_interface);
  m.def("mask_from_topk", &mask_from_topk_interface);
  m.def("partial_chunk_kivi_qk_dense_sparse",
        &partial_chunk_kivi_qk_dense_sparse_interface);
  m.def("h2d_gather_keys", &h2d_gather_keys);
  m.def("h2d_gather_keys_hybrid", &h2d_gather_keys_hybrid);
}
