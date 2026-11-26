// tvm-ffi entry points of the KDA ReplaySSM step: shape, dtype and layout
// checks, then kernel_run. The rings take their own slots and a slot stride
// (vLLM's Mamba page).
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include "kda_replayssm.cuh"

#include <initializer_list>

using tvm::ffi::TensorView;

namespace {

void check_ndim(const TensorView& t, const char* name, int ndim) {
  TVM_FFI_CHECK(t.ndim() == ndim, ValueError)
      << name << ": expected " << ndim << " dims, got " << t.ndim();
}

void check_tensor(const TensorView& t, const char* name,
                  std::initializer_list<int64_t> shape, int code, int bits,
                  int device_id, bool view = false) {
  check_ndim(t, name, static_cast<int>(shape.size()));
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    int64_t want = shape.begin()[i];
    TVM_FFI_CHECK(t.size(i) == want, ValueError)
        << name << ": size(" << i << ") is " << t.size(i) << ", expected "
        << want;
    TVM_FFI_CHECK(
        want == 1 || t.stride(i) == stride || (view && i + 1 < t.ndim()),
        ValueError)
        << name << ": not contiguous";
    stride *= want;
  }
  TVM_FFI_CHECK(
      t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1,
      ValueError)
      << name << ": wrong dtype";
  TVM_FFI_CHECK(
      t.device().device_type == kDLCUDA && t.device().device_id == device_id,
      ValueError)
      << name << ": must be on the same CUDA device as the first tensor";
}

void step_impl(TensorView x, TensorView beta, TensorView f_a, TensorView g_a,
               TensorView conv_weight, TensorView A_log, TensorView dt_bias,
               TensorView f_b_weight, TensorView g_b_weight,
               TensorView norm_weight, TensorView query_start_loc,
               TensorView slot, TensorView ring_slot, TensorView num_accepted,
               TensorView ring_start, TensorView num_committed,
               TensorView conv_state, TensorView checkpoint, TensorView u_ring,
               TensorView k_ring, TensorView g_ring, TensorView out,
               double lower_bound, double scale, double norm_eps,
               TensorView scratch) {
  check_ndim(x, "x", 4);
  check_ndim(beta, "beta", 2);
  check_ndim(f_a, "f_a", 2);
  check_ndim(g_a, "g_a", 2);
  check_ndim(conv_weight, "conv_weight", 4);
  check_ndim(A_log, "A_log", 1);
  check_ndim(dt_bias, "dt_bias", 2);
  check_ndim(f_b_weight, "f_b_weight", 3);
  check_ndim(g_b_weight, "g_b_weight", 3);
  check_ndim(norm_weight, "norm_weight", 1);
  check_ndim(query_start_loc, "query_start_loc", 1);
  check_ndim(slot, "slot", 1);
  check_ndim(ring_slot, "ring_slot", 1);
  check_ndim(num_accepted, "num_accepted", 1);
  check_ndim(ring_start, "ring_start", 1);
  check_ndim(num_committed, "num_committed", 1);
  check_ndim(conv_state, "conv_state", 5);
  check_ndim(checkpoint, "checkpoint", 4);
  check_ndim(u_ring, "u_ring", 4);
  check_ndim(k_ring, "k_ring", 4);
  check_ndim(g_ring, "g_ring", 4);
  check_ndim(out, "out", 3);
  const int64_t M = x.size(0);
  const int64_t P = x.size(1);
  const int64_t H = x.size(2);
  const int64_t D = x.size(3);
  const int64_t N = slot.size(0);
  const int64_t T = conv_state.size(1) - 2;
  const int64_t W = conv_weight.size(1);
  const int64_t S = conv_state.size(0);
  const int64_t R = u_ring.size(2);
  const int64_t MO = out.size(0);
  const int device = x.device().device_id;
  check_tensor(x, "x", {M, P, H, D}, kDLBfloat, 16, device, true);
  check_tensor(beta, "beta", {M, H}, kDLBfloat, 16, device, true);
  check_tensor(f_a, "f_a", {M, D}, kDLBfloat, 16, device, true);
  check_tensor(g_a, "g_a", {M, D}, kDLBfloat, 16, device, true);
  check_tensor(conv_weight, "conv_weight", {P, W, H, D}, kDLFloat, 32, device);
  check_tensor(A_log, "A_log", {H}, kDLFloat, 32, device);
  check_tensor(dt_bias, "dt_bias", {H, D}, kDLFloat, 32, device);
  check_tensor(f_b_weight, "f_b_weight", {H, D, D}, kDLBfloat, 16, device);
  check_tensor(g_b_weight, "g_b_weight", {H, D, D}, kDLBfloat, 16, device);
  check_tensor(norm_weight, "norm_weight", {D}, kDLFloat, 32, device);
  check_tensor(query_start_loc, "query_start_loc", {N + 1}, kDLInt, 32, device);
  check_tensor(slot, "slot", {N}, kDLInt, 32, device);
  check_tensor(ring_slot, "ring_slot", {N}, kDLInt, 32, device);
  check_tensor(num_accepted, "num_accepted", {N}, kDLInt, 32, device);
  check_tensor(ring_start, "ring_start", {N}, kDLInt, 32, device);
  check_tensor(num_committed, "num_committed", {N}, kDLInt, 32, device);
  check_tensor(conv_state, "conv_state", {S, T + 2, P, H, D}, kDLBfloat, 16,
               device, true);
  check_tensor(checkpoint, "checkpoint", {S, H, D, D}, kDLFloat, 32, device,
               true);
  check_tensor(u_ring, "u_ring", {S, H, R, D}, kDLFloat, 32, device, true);
  TVM_FFI_CHECK(u_ring.stride(1) == R * D && u_ring.stride(2) == D, ValueError)
      << "u_ring: inner dims not contiguous";
  check_tensor(k_ring, "k_ring", {S, H, R, D}, kDLBfloat, 16, device, true);
  TVM_FFI_CHECK(k_ring.stride(1) == R * D && k_ring.stride(2) == D, ValueError)
      << "k_ring: inner dims not contiguous";
  check_tensor(g_ring, "g_ring", {S, H, R, D}, kDLFloat, 32, device, true);
  TVM_FFI_CHECK(g_ring.stride(1) == R * D && g_ring.stride(2) == D, ValueError)
      << "g_ring: inner dims not contiguous";
  auto aligned16 = [](const TensorView& t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) + t.byte_offset()) % 16 ==
               0 &&
           t.stride(0) * t.dtype().bits / 8 % 16 == 0;
  };
  TVM_FFI_CHECK(aligned16(u_ring) && aligned16(k_ring) && aligned16(g_ring),
                ValueError)
      << "rings: base and slot stride must be 16-byte aligned";
  check_tensor(out, "out", {MO, H, D}, kDLBfloat, 16, device);
  const kda_replayssm::Dims dims{N, T, H, D, P, W, S, R, M, MO};
  size_t scratch_need = 0;
  kda_replayssm::kernel_workspace(dims, &scratch_need);
  check_tensor(scratch, "scratch", {scratch.ndim() == 1 ? scratch.size(0) : -1},
               kDLUInt, 8, device);
  TVM_FFI_CHECK(static_cast<size_t>(scratch.size(0)) >= scratch_need,
                ValueError)
      << "scratch too small";
  kda_replayssm::Args args;
  args.dims = dims;
  args.x = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<char*>(x.data_ptr()) + x.byte_offset());
  args.x_stride[0] = x.stride(0);
  args.x_stride[1] = x.stride(1);
  args.x_stride[2] = x.stride(2);
  args.x_stride[3] = x.stride(3);
  args.beta = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<char*>(beta.data_ptr()) + beta.byte_offset());
  args.beta_stride[0] = beta.stride(0);
  args.beta_stride[1] = beta.stride(1);
  args.f_a = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<char*>(f_a.data_ptr()) + f_a.byte_offset());
  args.f_a_stride[0] = f_a.stride(0);
  args.f_a_stride[1] = f_a.stride(1);
  args.g_a = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<char*>(g_a.data_ptr()) + g_a.byte_offset());
  args.g_a_stride[0] = g_a.stride(0);
  args.g_a_stride[1] = g_a.stride(1);
  args.conv_weight = static_cast<const float*>(conv_weight.data_ptr());
  args.A_log = static_cast<const float*>(A_log.data_ptr());
  args.dt_bias = static_cast<const float*>(dt_bias.data_ptr());
  args.f_b_weight = static_cast<const __nv_bfloat16*>(f_b_weight.data_ptr());
  args.g_b_weight = static_cast<const __nv_bfloat16*>(g_b_weight.data_ptr());
  args.norm_weight = static_cast<const float*>(norm_weight.data_ptr());
  args.query_start_loc =
      static_cast<const int32_t*>(query_start_loc.data_ptr());
  args.slot = static_cast<const int32_t*>(slot.data_ptr());
  args.ring_slot = static_cast<const int32_t*>(ring_slot.data_ptr());
  args.num_accepted = static_cast<const int32_t*>(num_accepted.data_ptr());
  args.ring_start = static_cast<const int32_t*>(ring_start.data_ptr());
  args.num_committed = static_cast<const int32_t*>(num_committed.data_ptr());
  args.conv_state = reinterpret_cast<__nv_bfloat16*>(
      static_cast<char*>(conv_state.data_ptr()) + conv_state.byte_offset());
  args.conv_state_stride[0] = conv_state.stride(0);
  args.conv_state_stride[1] = conv_state.stride(1);
  args.conv_state_stride[2] = conv_state.stride(2);
  args.conv_state_stride[3] = conv_state.stride(3);
  args.conv_state_stride[4] = conv_state.stride(4);
  args.checkpoint = reinterpret_cast<float*>(
      static_cast<char*>(checkpoint.data_ptr()) + checkpoint.byte_offset());
  args.checkpoint_stride[0] = checkpoint.stride(0);
  args.checkpoint_stride[1] = checkpoint.stride(1);
  args.checkpoint_stride[2] = checkpoint.stride(2);
  args.checkpoint_stride[3] = checkpoint.stride(3);
  args.u_ring = reinterpret_cast<float*>(static_cast<char*>(u_ring.data_ptr()) +
                                         u_ring.byte_offset());
  args.u_ring_stride0 = u_ring.stride(0);
  args.k_ring = reinterpret_cast<__nv_bfloat16*>(
      static_cast<char*>(k_ring.data_ptr()) + k_ring.byte_offset());
  args.k_ring_stride0 = k_ring.stride(0);
  args.g_ring = reinterpret_cast<float*>(static_cast<char*>(g_ring.data_ptr()) +
                                         g_ring.byte_offset());
  args.g_ring_stride0 = g_ring.stride(0);
  args.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  args.lower_bound = static_cast<float>(lower_bound);
  args.scale = static_cast<float>(scale);
  args.norm_eps = static_cast<float>(norm_eps);
  kda_replayssm::Ctx ctx;
  ctx.stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, device));
  ctx.device = device;
  ctx.sm_count = 0;
  TVM_FFI_CHECK(
      cudaDeviceGetAttribute(&ctx.sm_count, cudaDevAttrMultiProcessorCount,
                             device) == cudaSuccess,
      RuntimeError)
      << "cudaDeviceGetAttribute failed";
  ctx.scratch = scratch.data_ptr();
  ctx.scratch_bytes = static_cast<size_t>(scratch.size(0));
  kda_replayssm::kernel_run(args, ctx);
  cudaError_t err = cudaGetLastError();
  TVM_FFI_CHECK(err == cudaSuccess, RuntimeError)
      << "kernel_run: " << cudaGetErrorString(err);
}

// Scratch of a step with M token rows of H heads of D channels.
int64_t scratch_bytes_impl(int64_t M, int64_t H, int64_t D) {
  kda_replayssm::Dims dims{};
  dims.M = M, dims.H = H, dims.D = D;
  size_t bytes = 0;
  kda_replayssm::kernel_workspace(dims, &bytes);
  return static_cast<int64_t>(bytes);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(step, step_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scratch_bytes, scratch_bytes_impl);
