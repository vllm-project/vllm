// Variant 3: apache-tvm-ffi. Direct C ABI, no torch dispatcher involved. The
// .so does not link against libtorch at all; it only links against libtvm_ffi.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>

#include "scale_kernel.cuh"

namespace bench_tvmffi {

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
}

void scale(tvm::ffi::TensorView out, tvm::ffi::TensorView in, double factor) {
  cudaStream_t stream = get_stream(in.device());
  launch_scale_f32(static_cast<float*>(out.data_ptr()),
                   static_cast<const float*>(in.data_ptr()),
                   static_cast<float>(factor), in.numel(), stream);
}

}  // namespace bench_tvmffi

TVM_FFI_DLL_EXPORT_TYPED_FUNC(scale, bench_tvmffi::scale);
