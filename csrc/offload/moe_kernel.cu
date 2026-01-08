#include <cuda_runtime.h>
#ifndef cudaErrorNotInitialized
#define cudaErrorNotInitialized ((cudaError_t)1000)  // Arbitrary unused error code
#endif
#include <cstdint>
#include "moe.h"
#include <cub/block/block_scan.cuh>
#include <nvtx3/nvToolsExt.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// ==================== CUDA Kernels ====================
constexpr int MAX_NUM_EXPERT = 256;
constexpr int MAX_CACHE_EXPERT = 128;
constexpr int BYTES_PER_VEC = 16;
constexpr int BN = 32;
constexpr int BK = 32;
constexpr int BM = 16;
constexpr int warpSize = 32;

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    #if __CUDA_ARCH__ >= 800
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
        unsigned long long gmem_addr = (unsigned long long)gmem_src;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
    #else
        // fallback
        *reinterpret_cast<int4*>(smem_dst) = *reinterpret_cast<const int4*>(gmem_src);
    #endif
    }
    
    __device__ __forceinline__ void cp_async_commit() {
    #if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    #endif
    }
    
    template<int n>
    __device__ __forceinline__ void cp_async_wait_impl() {
    #if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group %0;\n" :: "n"(n) : "memory");
    #endif
    }
    

__global__ void submit_kernel(AsyncState* data, int layer_idx, int batch_idx, int num_tokens) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        data->layer_idx = layer_idx;
        data->batch_idx = batch_idx;
        data->num_tokens = num_tokens;
        data->callback_completed = 0;
        data->gpu_signal = 1;
        data->submit_count += 1;
        __threadfence_system();  // 确保写入对 CPU 可见
    }
}

__global__ void sync_kernel(AsyncState* data) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (data->callback_completed == 0) {
            __threadfence_system();  // 确保从 CPU 读取最新值
        }
        data->sync_count += 1;
    }
}

__global__ void cache_policy_kernel(
    int* cache_map,    // [256] 输入映射表
    int* miss_map,
    int* copy_map,
    int* sort,         // [128] 输入输出排序数组
    int* topk,   // [32]  输入topk数组
    int* cpu_topk,        // [32]  输出处理后topk
    const int C,   // total num of cached expert  < 128
    const int N,   // total num of expert  < 256
    const int active_expert_num,   // total num of expert in topk < 64
    const int K,
    const int update_expert_num
) {
    const int tid = threadIdx.x;

    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage temp_buf;
    // 共享内存分配
    __shared__ bool s_in_topk[MAX_NUM_EXPERT];   // topk标记
    __shared__ bool s_in_match[MAX_NUM_EXPERT];  // match标记
    __shared__ int  s_miss_list[MAX_NUM_EXPERT];  // miss列表
    __shared__ int  s_cache_map[MAX_NUM_EXPERT];
    __shared__ int  s_sort[MAX_CACHE_EXPERT];

    // 初始化共享内存
    s_in_topk[tid] = false;
    s_in_match[tid] = false;
    s_miss_list[tid] = -1;
    s_cache_map[tid] = cache_map[tid];

    copy_map[tid] = -1;
    int top_k_id = -1;
    bool is_match=true;
    int val = 0;
    __syncthreads();

    // 阶段1: 标记topk元素
    if (tid < active_expert_num) {
        top_k_id = topk[tid];
        if (tid % 8 < K){
            s_in_topk[top_k_id] = true;
        }
    }
    __syncthreads();


    // 阶段2: 计算miss/match
    int offset;
    bool match = s_cache_map[tid] > -1;
    bool active = s_in_topk[tid];
    const bool is_miss  = active & !match;
    s_in_match[tid] = active & match;

    {
        BlockScan(temp_buf).ExclusiveSum(is_miss ? 1 : 0, offset);
        if(is_miss){
            s_miss_list[offset] = tid;
        }
    }

    // 阶段3: 重新排序，match的数据移动到末尾
    if(tid < C){
        val = sort[tid];
        //printf("tid:%3d, val:%3d\n", tid, val);
        is_match = s_in_match[val];
    }
    int unused_offset;
    int used_offset;

    BlockScan(temp_buf).ExclusiveSum(is_match ? 0 : 1, unused_offset);
    BlockScan(temp_buf).ExclusiveSum(is_match ? 1 : 0, used_offset);

    int new_pos = is_match ? (C - 1  - used_offset) : unused_offset;
    if(tid < C){
        //printf("tid:%3d, val:%3d, pos:%3d\n", tid, val, new_pos);
        s_sort[new_pos] = val;
    }
    __syncthreads();

    // 处理swap数据，
    if(tid < update_expert_num){
        int miss_id = s_miss_list[tid];
        int evict_id = s_sort[tid];
        int new_pos = s_cache_map[evict_id];

        s_sort[tid] = miss_id >= 0 ? miss_id : evict_id;

        if(miss_id >= 0)
        {
            //printf("missid:  %d\n", miss_id);
            int evict_pos = miss_map[miss_id];
            copy_map[miss_id] = new_pos;

            s_cache_map[miss_id] = new_pos;
            s_cache_map[evict_id] = -1;
            miss_map[evict_id] = evict_pos;
            miss_map[miss_id] = -1;

            s_in_match[miss_id] = true;
        }
    }
    __syncthreads();

    // 阶段5: 更新load expert后的sort
    if(tid < C){
        val = s_sort[tid];
        is_match = s_in_match[val];
    }
    BlockScan(temp_buf).ExclusiveSum(is_match ? 0 : 1, unused_offset);
    BlockScan(temp_buf).ExclusiveSum(is_match ? 1 : 0, used_offset);

    new_pos = is_match ? (C - 1 - used_offset) : unused_offset;

    if (tid < C){
        sort[new_pos] = val;
    }

    // 计算topk
    if(tid < active_expert_num)
    {
        int flag = (s_cache_map[top_k_id] >= 0);
        cpu_topk[tid] = flag ? -1 : top_k_id;
    }
    cache_map[tid] = s_cache_map[tid];
    __syncthreads();
}

// ==================== CpuMoeLayer CUDA 方法 ====================

__global__ void update_expert_cache_kernel(
    const uint8_t* __restrict__ src_w13,
    const uint8_t* __restrict__ src_w2,
    const float* __restrict__ src_scales_w13,
    const float* __restrict__ src_scales_w2,
    uint8_t* __restrict__ dst_w13,
    uint8_t* __restrict__ dst_w2,
    float* __restrict__ dst_scales_w13,
    float* __restrict__ dst_scales_w2,
    const int* __restrict__ map,
    const int64_t w13_bytes_per_expert,
    const int64_t w2_bytes_per_expert,
    const int64_t w13_scale_bytes_per_expert,
    const int64_t w2_scale_bytes_per_expert,
    const int num_experts,
    const int W13_N,
    const int W13_K,
    const int W2_N,
    const int W2_K,
    const int W13_scale_N,
    const int W13_scale_K,
    const int W2_scale_N,
    const int W2_scale_K,
    const int map_size,           // 新增：map 数组大小
    const int cache_expert_num,   // 新增：缓存中的专家数量
    const int source_expert_num   // 新增：源数据中的专家数量
)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t grid_stride = gridDim.x * blockDim.x * BYTES_PER_VEC;
    const int64_t thread_stride = blockDim.x * gridDim.x;

    const uint8_t* src_w13_scales_bytes = reinterpret_cast<const uint8_t*>(src_scales_w13);
    const uint8_t* src_w2_scales_bytes = reinterpret_cast<const uint8_t*>(src_scales_w2);
    uint8_t* dst_w13_scales_bytes = reinterpret_cast<uint8_t*>(dst_scales_w13);
    uint8_t* dst_w2_scales_bytes = reinterpret_cast<uint8_t*>(dst_scales_w2);
    const int block_id = blockIdx.x;
    const int64_t tid_warp = threadIdx.x % warpSize;
    const int64_t warp_id  = threadIdx.x / warpSize;
    const int64_t n_warp = blockDim.x / warpSize;
    const int64_t thread_offset  = tid_warp * BYTES_PER_VEC;
    const int64_t start_tile = block_id * n_warp + warp_id;

    __shared__ int16_t tile[8][2][32][8];
    int tmp;
    int k_mid;
    int k_outer;
    int n_outer;
    int n_dst;
    int k_dst;
    int16_t out16[8];
    const int lane_row = (tid_warp >> 2) + ((tid_warp & 3) << 3);


    auto copy_chunk = [&](const uint8_t* __restrict__ src_base,
                          uint8_t* __restrict__ dst_base,
                          const int64_t chunk_size,
                          const int N,
                          const int K
                        ) {
        for (int64_t byte_offset = tid * BYTES_PER_VEC; byte_offset < chunk_size; byte_offset += grid_stride) {
            const float4* s = reinterpret_cast<const float4*>(src_base + byte_offset);
            float4* d = reinterpret_cast<float4*>(dst_base + byte_offset);
            *d = *s;
        }  
    };


    auto copy_chunk_vnni = [&](const uint8_t* __restrict__ src_base,
                              uint8_t* __restrict__ dst_base,
                              const int64_t chunk_size,
                              const int N,
                              const int K
                            ) {
        const int tile_size = BM * BN;  // 512 字节
        const int64_t n_tiles = chunk_size / tile_size;
        const int64_t k_outer_dim = (int64_t)K / (int64_t)32;  // BK=32
        const int64_t tile_stride = gridDim.x * blockDim.x * BYTES_PER_VEC / tile_size;
        
        int buf_id = 0;
        int next_buf = buf_id ^ 1;
        
        // 第一个 tile 的异步拷贝
        const int4* s = reinterpret_cast<const int4*>(src_base + start_tile * tile_size + thread_offset);
        cp_async_16B(reinterpret_cast<int4*>(&tile[warp_id][buf_id][tid_warp]), s);
        cp_async_commit();
        cp_async_wait_impl<0>();
        __syncwarp();
        
        int64_t tile_id = start_tile;
        for(; tile_id < n_tiles - tile_stride; tile_id += tile_stride){
            // 先进行下一个 tile 的异步拷贝
            const int4* s = reinterpret_cast<const int4*>(src_base + (tile_id + tile_stride) * tile_size + thread_offset);
            cp_async_16B(reinterpret_cast<int4*>(&tile[warp_id][next_buf][tid_warp]), s);
            cp_async_commit();
            cp_async_wait_impl<1>();

            // 进行转换（从当前 buf_id 读取）
            #pragma unroll 
            for(int i = 0; i < 8; i++){
                out16[i] = tile[warp_id][buf_id][(i << 2) + (tid_warp & 3)][tid_warp >> 2]; 
            }

            // 计算索引（基于当前 tile_id）
            tmp = tile_id << 3;
            k_mid = tmp & 15;
            tmp >>= 4;
            k_outer = tmp % k_outer_dim;
            n_outer = tmp / k_outer_dim;
            n_dst = (n_outer << 5) + lane_row;
            k_dst = (k_outer << 5) + (k_mid << 1);
            int4* d = reinterpret_cast<int4*>(dst_base + n_dst * K + k_dst);
            const int4 data = *reinterpret_cast<const int4*>(&out16[0]);
            *d = data;

            // 流水线更替 buffer_id
            buf_id = buf_id ^ 1;
            next_buf = next_buf ^ 1;
            __syncwarp();
        } 
        
        // 处理最后一个 tile
        cp_async_wait_impl<0>();
        #pragma unroll 
        for(int i = 0; i < 8; i++){
            out16[i] = tile[warp_id][buf_id][(i << 2) + (tid_warp & 3)][tid_warp >> 2];
        }  
        tmp = tile_id << 3;
        k_mid = tmp & 15;
        tmp >>= 4;
        k_outer = tmp % k_outer_dim;
        n_outer = tmp / k_outer_dim;
        n_dst = (n_outer << 5) + lane_row;
        k_dst = (k_outer << 5) + (k_mid << 1);
        const int4 data = *reinterpret_cast<const int4*>(&out16[0]);
        int4* d = reinterpret_cast<int4*>(dst_base + n_dst * K + k_dst);
        *d = data;
    };

    for (int e = 0; e < num_experts; ++e) {
        const int dst_idx = map[e];
        if (dst_idx < 0) continue;
        copy_chunk_vnni(src_w13 + e * w13_bytes_per_expert, dst_w13 + dst_idx * w13_bytes_per_expert, w13_bytes_per_expert,W13_N,W13_K);
        copy_chunk_vnni(src_w2 + e * w2_bytes_per_expert, dst_w2 + dst_idx * w2_bytes_per_expert, w2_bytes_per_expert,W2_N,W2_K);
        copy_chunk(src_w13_scales_bytes + e * w13_scale_bytes_per_expert, dst_w13_scales_bytes + dst_idx * w13_scale_bytes_per_expert, w13_scale_bytes_per_expert,W13_scale_N,W13_scale_K);
        copy_chunk(src_w2_scales_bytes + e * w2_scale_bytes_per_expert, dst_w2_scales_bytes + dst_idx * w2_scale_bytes_per_expert, w2_scale_bytes_per_expert,W2_scale_N,W2_scale_K);
    }
}


void Moe::update_expert_cache(
    torch::Tensor w13_cache,
    torch::Tensor w2_cache,
    torch::Tensor w13_scale_cache,
    torch::Tensor w2_scale_cache,
    torch::Tensor map,
    int64_t num_experts)
{
    // Sanity Checks
    TORCH_CHECK(w13_cache.is_cuda() && w2_cache.is_cuda() && map.is_cuda());
    TORCH_CHECK(w13_scale_cache.is_cuda() && w2_scale_cache.is_cuda());

    // Compute bytes per expert
    const int64_t w13_bytes_per_expert = w13_cache.size(1) * w13_cache.size(2) * w13_cache.element_size();
    const int64_t w2_bytes_per_expert = w2_cache.size(1) * w2_cache.size(2) * w2_cache.element_size();
    const int64_t w13_scale_bytes_per_expert = w13_scale_cache.size(1) * w13_scale_cache.size(2) * w13_scale_cache.element_size();
    const int64_t w2_scale_bytes_per_expert = w2_scale_cache.size(1) * w2_scale_cache.size(2) * w2_scale_cache.element_size();
    const int64_t actual_num_experts  = std::min(map.size(0), num_experts);

    
    // 2. 检查缓存大小（第一维是专家数量）
    const int64_t cache_expert_num = w13_cache.size(0);
    // Dynamic Launch Configuration
    constexpr int64_t BLOCK_SIZE = 256;
    const int64_t GRID_SIZE = 4;

    const int W13_N = w13_cache.size(1);
    const int W13_K = w13_cache.size(2);
    const int W2_N = w2_cache.size(1);
    const int W2_K = w2_cache.size(2);
    const int W13_scale_N = w13_scale_cache.size(1);
    const int W13_scale_K = w13_scale_cache.size(2);
    const int W2_scale_N = w2_scale_cache.size(1);
    const int W2_scale_K = w2_scale_cache.size(2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    update_expert_cache_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(m_w13_weights),  
        reinterpret_cast<const uint8_t*>(m_w2_weights),  
        reinterpret_cast<const float*>(m_w13_scale),     
        reinterpret_cast<const float*>(m_w2_scale),   
        reinterpret_cast<uint8_t*>(w13_cache.data_ptr()),
        reinterpret_cast<uint8_t*>(w2_cache.data_ptr()),
        w13_scale_cache.data_ptr<float>(),
        w2_scale_cache.data_ptr<float>(),
        map.data_ptr<int>(),
        w13_bytes_per_expert, w2_bytes_per_expert,
        w13_scale_bytes_per_expert, w2_scale_bytes_per_expert,
        num_experts,
        W13_N,
        W13_K,
        W2_N,
        W2_K,
        W13_scale_N,
        W13_scale_K,
        W2_scale_N,
        W2_scale_K,
        static_cast<int>(map.size(0)),    // map_size
        static_cast<int>(cache_expert_num), // cache_expert_num
        config_.expert_num                 // source_expert_num
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(launch_err));
    }
    
}

void MoeOffloadEngine::expert_cache_policy(
    torch::Tensor cache_map,
    torch::Tensor miss_map,
    torch::Tensor policy_sort,
    torch::Tensor topk_ids,
    torch::Tensor cpu_topk,
    torch::Tensor copy_map
){
    TORCH_CHECK(cache_map.device().is_cuda(), "cache_map mast be CUDA");
    TORCH_CHECK(miss_map.device().is_cuda(), "miss_map mast be CUDA");
    TORCH_CHECK(policy_sort.device().is_cuda(), "policy_sort mast be CUDA");
    TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids mast be CUDA");
    TORCH_CHECK(cpu_topk.device().is_cuda(), "cpu_topk mast be CUDA");
    TORCH_CHECK(copy_map.device().is_cuda(), "copy_map mast be CUDA");

    const int threads = 256;
    const int blocks = 1;
    auto num_tokens = topk_ids.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cache_policy_kernel<<<blocks, threads, 0, stream>>>(
        cache_map.data_ptr<int>(),
        miss_map.data_ptr<int>(),
        copy_map.data_ptr<int>(),
        policy_sort.data_ptr<int>(),
        topk_ids.data_ptr<int>(),
        cpu_topk.data_ptr<int>(),
        config_.cache_expert_num,
        config_.expert_num,
        num_tokens * config_.num_experts_per_tok,
        config_.cache_topk,
        config_.update_expert_num
    );
}

// ==================== MoeOffloadEngine CUDA 方法 ====================

void MoeOffloadEngine::initialize_async_state() {
    if (async_state_initialized_) {
        throw std::runtime_error("Async state already initialized");
    }

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties: " +
                                 std::string(cudaGetErrorString(err)));
    }

    if (!prop.canMapHostMemory) {
        throw std::runtime_error("GPU does not support mapped host memory");
    }

    // 分配零拷贝内存
    err = cudaHostAlloc((void**)&cpu_state_, sizeof(AsyncState), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostAlloc failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // 获取设备指针
    err = cudaHostGetDevicePointer((void**)&gpu_state_, (void*)cpu_state_, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(cpu_state_);
        cpu_state_ = nullptr;
        throw std::runtime_error("cudaHostGetDevicePointer failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // 初始化状态
    *cpu_state_ = AsyncState();  // 使用构造函数初始化

    // 启动轮询线程
    try {
        polling_thread_ = std::thread(&MoeOffloadEngine::cpu_polling_loop, this);
    } catch (const std::system_error& e) {
        cudaFreeHost(cpu_state_);
        cpu_state_ = nullptr;
        gpu_state_ = nullptr;
        throw std::runtime_error("Failed to create thread: " + std::string(e.what()));
    }

    async_state_initialized_ = true;
    fprintf(stderr, "[MoeOffloadEngine] Async state initialized: gpu_state=%p cpu_state=%p\n",
            gpu_state_, cpu_state_);
}

void MoeOffloadEngine::cleanup_async_state() {
    if (cpu_state_) {
        cudaFreeHost(cpu_state_);
        cpu_state_ = nullptr;
        gpu_state_ = nullptr;
    }
}

cudaError_t MoeOffloadEngine::submit(int layer_idx, int batch_idx, int num_tokens) {
    nvtxRangePushA("MoeOffloadEngine::submit");
    if (!async_state_initialized_) {
        fprintf(stderr, "[ERROR] submit_kernel engine not initialized\n");
        return cudaErrorNotInitialized;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    submit_kernel<<<1, 1, 0, stream>>>(gpu_state_, layer_idx, batch_idx, num_tokens);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[ERROR] submit_kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
    }

    nvtxRangePop();
    return launch_err;
}

cudaError_t MoeOffloadEngine::sync() {
    nvtxRangePushA("MoeOffloadEngine::sync");
    if (!async_state_initialized_) {
        return cudaErrorNotInitialized;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sync_kernel<<<1, 1, 0, stream>>>(gpu_state_);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[ERROR] wait_callback_completion_kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
        nvtxRangePop();
        return launch_err;
    }
    nvtxRangePop();
    return launch_err;
}

void MoeOffloadEngine::get_output(torch::Tensor gpu_output) {
    TORCH_CHECK(gpu_output.device().is_cuda(), "Output must be on CUDA device");
    int64_t n = gpu_output.size(0);
    TORCH_CHECK(output_.size(0) >= n, "CPU output buffer too small");
    //gpu_output.copy_(output_.slice(0, 0, n), true);

    size_t copy_bytes = std::min(gpu_output.nbytes(), output_.nbytes());
    const cudaStream_t copyStream = at::cuda::getCurrentCUDAStream(gpu_output.device().index());
    AT_CUDA_CHECK(cudaMemcpyAsync(gpu_output.data_ptr(),
                                  output_.data_ptr(),
                                  copy_bytes,
                                  cudaMemcpyHostToDevice,
                                  copyStream));

}

void MoeOffloadEngine::set_input(torch::Tensor gpu_hidden_states,
                                torch::Tensor gpu_topk_ids,
                                torch::Tensor gpu_topk_weights) {
    int64_t n = gpu_hidden_states.size(0);
    TORCH_CHECK(gpu_hidden_states.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(gpu_topk_ids.device().is_cuda(), "topk_ids must be on CUDA");
    TORCH_CHECK(gpu_topk_weights.device().is_cuda(), "topk_weights must be on CUDA");
    TORCH_CHECK(n <= config_.max_batch_token, "max_batch_token");
    //hidden_states_.slice(0, 0, n).copy_(gpu_hidden_states.slice(0, 0, n), true);
    //topk_ids_.slice(0, 0, n).copy_(gpu_topk_ids.slice(0, 0, n), true);
    //topk_weights_.slice(0, 0, n).copy_(gpu_topk_weights.slice(0, 0, n), true);


    size_t input_copy_bytes = std::min(gpu_hidden_states.nbytes(), hidden_states_.nbytes());
    size_t ids_copy_bytes = std::min(gpu_topk_ids.nbytes(), topk_ids_.nbytes());
    size_t weights_copy_bytes = std::min(gpu_topk_weights.nbytes(), topk_weights_.nbytes());

    // 3. 取目标 stream（优先使用传入的 stream）
    const cudaStream_t copyStream = at::cuda::getCurrentCUDAStream(gpu_hidden_states.device().index());

    // 4. 异步拷贝（仅拷贝适配部分）
    cudaMemcpyAsync(hidden_states_.data_ptr(), gpu_hidden_states.data_ptr(), input_copy_bytes, cudaMemcpyDeviceToHost, copyStream);
    cudaMemcpyAsync(topk_ids_.data_ptr(), gpu_topk_ids.data_ptr(), ids_copy_bytes, cudaMemcpyDeviceToHost, copyStream);
    cudaMemcpyAsync(topk_weights_.data_ptr(), gpu_topk_weights.data_ptr(), weights_copy_bytes, cudaMemcpyDeviceToHost, copyStream);

}