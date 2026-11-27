#include "common.cuh"
#include "native_fp32_boundary_map.cuh"
#include "native_fp32_while.cuh"

namespace DOSA_NAMESPACE::native_fp32_host {
using Scalar = torch::headeronly::ScalarType;

void same_device(const Tensor& anchor, std::initializer_list<Tensor> tensors) {
  STD_TORCH_CHECK(anchor.is_cuda(),
                  "native FP32 operators require CUDA tensors");
  for (const auto& x : tensors)
    STD_TORCH_CHECK(x.is_cuda() && x.device() == anchor.device(),
                    "native FP32 operands must share a CUDA device");
}
void contiguous(const Tensor& x, Scalar dtype) {
  STD_TORCH_CHECK(
      x.scalar_type() == dtype && x.is_contiguous(),
      "native FP32 operand has wrong dtype or noncontiguous storage");
}
void boundary_tasks(Tensor partition, Tensor error, Tensor tasks, Tensor state,
                    int64_t id_stride) {
  same_device(partition, {error, tasks, state});
  for (const auto& x : {partition, error, tasks, state})
    contiguous(x, Scalar::Int);
  STD_TORCH_CHECK(error.numel() == partition.numel() && tasks.dim() == 2 &&
                  tasks.size(1) == 3 && tasks.size(0) >= partition.numel());
  STD_TORCH_CHECK(state.numel() == 2 && id_stride >= 4096 &&
                  id_stride <= INT_MAX &&
                  partition.numel() <= INT_MAX / id_stride);
  DeviceGuard guard(partition.get_device_index());
  DOSA_CUDA_CHECK(cudaMemsetAsync(state.mutable_data_ptr<int>(), 0,
                                  2 * sizeof(int), get_current_cuda_stream()));
  if (partition.numel())
    launch<native_fp32_boundary_map::tasks, 256>(
        (partition.numel() + 255) / 256, partition.const_data_ptr<int>(),
        error.mutable_data_ptr<int>(), tasks.mutable_data_ptr<int>(),
        state.mutable_data_ptr<int>(), int(partition.numel()), int(id_stride));
}
void attach_while(Tensor cursor, Tensor count, int64_t batch, int64_t body) {
  same_device(cursor, {count});
  contiguous(cursor, Scalar::Int);
  contiguous(count, Scalar::Int);
  STD_TORCH_CHECK(cursor.numel() == 1 && count.numel() == 1 &&
                  cursor.data_ptr() != count.data_ptr());
  STD_TORCH_CHECK(batch > 0 && batch <= INT_MAX && body != 0);
  DeviceGuard guard(cursor.get_device_index());
  native_fp32_while::attach(get_current_cuda_stream(),
                            cursor.mutable_data_ptr<int>(),
                            count.const_data_ptr<int>(), int(batch),
                            reinterpret_cast<cudaGraph_t>(body));
}
}  // namespace DOSA_NAMESPACE::native_fp32_host

DOSA_REGISTER("boundary_tasks",
              "(Tensor partition, Tensor! error, Tensor! tasks, Tensor! state, "
              "int id_stride) -> ()",
              CUDA, native_fp32_host::boundary_tasks)
DOSA_REGISTER("attach_while",
              "(Tensor! cursor, Tensor count, int batch, int body) -> ()", CUDA,
              native_fp32_host::attach_while)
