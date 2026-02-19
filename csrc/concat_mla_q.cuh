 #ifndef CONCAT_MLA_Q_CUH_
 #define CONCAT_MLA_Q_CUH_
 
 #include <cuda_bf16.h>
 #include <cuda_fp16.h>
 #include <cuda_runtime.h>
 
 namespace vllm {
 
 struct __align__(32) vec8 {
   unsigned int d[8];
 };

 __forceinline__ __device__ vec8 ld_cs_v8(const vec8* addr) {
   vec8 val;
   asm volatile(
       "ld.global.cs.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
       : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
         "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
       : "l"(addr));
   return val;
 }
 
 __forceinline__ __device__ void st_cs_v8(vec8* addr, vec8 val) {
   asm volatile(
       "st.global.cs.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
       ::"l"(addr),
       "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]), "r"(val.d[3]),
       "r"(val.d[4]), "r"(val.d[5]), "r"(val.d[6]), "r"(val.d[7]));
 }
 
 __forceinline__ __device__ int ld_cs_v1(const int* addr) {
   int val;
   asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(addr));
   return val;
 }
 
 __forceinline__ __device__ void st_cs_v1(int* addr, int val) {
   asm volatile("st.global.cs.b32 [%0], %1;" ::"l"(addr), "r"(val));
 }
 
 template <typename DType, int NOPE_V8_LOADS>
 __global__ void ConcatMLAQKernel(
     DType* __restrict__ q_out,
     const DType* __restrict__ ql_nope,
     const DType* __restrict__ q_pe,
     const int num_tokens,
     const int num_heads,
     const int nope_dim,
     const int64_t out_stride_0,
     const int64_t out_stride_1,
     const int64_t nope_stride_0,
     const int64_t nope_stride_1,
     const int64_t pe_stride_0,
     const int64_t pe_stride_1) {
   const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
   if (flat_warp_id >= num_tokens * num_heads) return;
 
   const int token_id = flat_warp_id / num_heads;
   const int head_id = flat_warp_id % num_heads;
   const int lane_id = threadIdx.x & 31;
 
   const vec8* nope_src = reinterpret_cast<const vec8*>(
       ql_nope + token_id * nope_stride_0 + head_id * nope_stride_1);
   vec8* nope_dst = reinterpret_cast<vec8*>(
       q_out + token_id * out_stride_0 + head_id * out_stride_1);
 
 #pragma unroll
   for (int i = 0; i < NOPE_V8_LOADS; i++) {
     const int offset = i * 32 + lane_id;
     st_cs_v8(nope_dst + offset, ld_cs_v8(nope_src + offset));
   }
 
   const int* rope_src = reinterpret_cast<const int*>(
       q_pe + token_id * pe_stride_0 + head_id * pe_stride_1);
   int* rope_dst = reinterpret_cast<int*>(
       q_out + token_id * out_stride_0 + head_id * out_stride_1 + nope_dim);
 
   st_cs_v1(rope_dst + lane_id, ld_cs_v1(rope_src + lane_id));
 }
 
 }  // namespace vllm
 
 #endif  // CONCAT_MLA_Q_CUH_
 