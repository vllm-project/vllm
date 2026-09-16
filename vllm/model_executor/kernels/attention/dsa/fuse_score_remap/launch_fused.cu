// Host launch + torch binding for sm90_fp8_paged_mqa_logits_fused.
// TMA construction matches DeepGEMM 2.6.1 sm90_fp8_paged_mqa_logits host.

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "merge_cta_topk.cuh"
#include "sm90_fp8_paged_mqa_logits_fused.cuh"

namespace {

constexpr int kNextN = 1;
constexpr int kNumHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kBlockKV = 64;
constexpr int kNumQStages = 3;
constexpr int kNumKVStages = 3;
constexpr int kSplitKV = 256;
constexpr int kNumTMAThreads = 128;
constexpr int kNumMathThreads = 512;
constexpr int kNumMathWarpGroups = kNumMathThreads / 128;
constexpr int kPageSize = 64;
constexpr int kHeadDimWithScale = 132;

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

void cu_check(CUresult err, const char* what) {
    if (err != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        throw std::runtime_error(std::string(what) + ": " + (msg ? msg : "cuda driver error"));
    }
}

CUtensorMapDataType aten_to_tma_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case torch::kInt:
            return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kFloat8_e4m3fn:
            return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        default:
            throw std::runtime_error("unsupported TMA dtype");
    }
}

CUtensorMapSwizzle swizzle_of(int mode) {
    switch (mode) {
        case 0:
        case 16:
            return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:
            return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:
            return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128:
            return CU_TENSOR_MAP_SWIZZLE_128B;
        default:
            throw std::runtime_error("unsupported TMA swizzle");
    }
}

CUtensorMap make_tma_2d(
    const torch::Tensor& t,
    int gmem_inner,
    int gmem_outer,
    int smem_inner,
    int smem_outer,
    int gmem_outer_stride,
    int swizzle_mode) {
    const int elem = static_cast<int>(t.element_size());
    if (swizzle_mode != 0) {
        smem_inner = swizzle_mode / elem;
    }
    CUtensorMap map{};
    const cuuint64_t gmem_dims[2] = {
        static_cast<cuuint64_t>(gmem_inner),
        static_cast<cuuint64_t>(gmem_outer),
    };
    const cuuint32_t smem_dims[2] = {
        static_cast<cuuint32_t>(smem_inner),
        static_cast<cuuint32_t>(smem_outer),
    };
    const cuuint64_t gmem_strides[1] = {
        static_cast<cuuint64_t>(gmem_outer_stride) * static_cast<cuuint64_t>(elem),
    };
    const cuuint32_t elem_strides[2] = {1, 1};
    cu_check(
        cuTensorMapEncodeTiled(
            &map,
            aten_to_tma_dtype(t.scalar_type()),
            2,
            t.data_ptr(),
            gmem_dims,
            gmem_strides,
            smem_dims,
            elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_of(swizzle_mode),
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled 2D");
    return map;
}

CUtensorMap make_tma_3d(
    const torch::Tensor& t,
    int g0,
    int g1,
    int g2,
    int s0,
    int s1,
    int s2,
    int stride0,
    int stride1,
    int swizzle_mode) {
    const int elem = static_cast<int>(t.element_size());
    if (swizzle_mode != 0) {
        s0 = swizzle_mode / elem;
    }
    CUtensorMap map{};
    const cuuint64_t gmem_dims[3] = {
        static_cast<cuuint64_t>(g0),
        static_cast<cuuint64_t>(g1),
        static_cast<cuuint64_t>(g2),
    };
    const cuuint32_t smem_dims[3] = {
        static_cast<cuuint32_t>(s0),
        static_cast<cuuint32_t>(s1),
        static_cast<cuuint32_t>(s2),
    };
    const cuuint64_t gmem_strides[2] = {
        static_cast<cuuint64_t>(stride0) * static_cast<cuuint64_t>(elem),
        static_cast<cuuint64_t>(stride1) * static_cast<cuuint64_t>(elem),
    };
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    cu_check(
        cuTensorMapEncodeTiled(
            &map,
            aten_to_tma_dtype(t.scalar_type()),
            3,
            t.data_ptr(),
            gmem_dims,
            gmem_strides,
            smem_dims,
            elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_of(swizzle_mode),
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled 3D");
    return map;
}

constexpr int align_up(int x, int a) { return (x + a - 1) / a * a; }

int original_smem_bytes() {
    const int swizzle_alignment = kHeadDim * 8;
    const int smem_q = kNextN * kNumHeads * kHeadDim;  // fp8
    const int aligned_w = align_up(kNextN * kNumHeads * 4, swizzle_alignment);
    const int q_pipe =
        kNumQStages * (smem_q + aligned_w) + align_up(kNumQStages * 8 * 2, swizzle_alignment);
    const int smem_kv = kBlockKV * kHeadDim;
    const int aligned_kv_scale = align_up(kBlockKV * 4, swizzle_alignment);
    const int kv_pipe = kNumKVStages * (smem_kv + aligned_kv_scale) +
                        align_up(kNumKVStages * 8 * 2, swizzle_alignment);
    const int umma = kNumMathWarpGroups * 2 * 8;
    return q_pipe + kNumMathWarpGroups * kv_pipe + umma + 4;
}

using FusedKernel = void (*)(
    uint32_t,
    uint32_t,
    const uint32_t*,
    const uint32_t*,
    const uint32_t*,
    const uint32_t*,
    float*,
    int32_t*,
    uint32_t,
    uint32_t,
    const cute::TmaDescriptor,
    const cute::TmaDescriptor,
    const cute::TmaDescriptor,
    const cute::TmaDescriptor);

FusedKernel fused_fn() {
    return &deep_gemm::sm90_fp8_paged_mqa_logits_fused<
        kNextN,
        kNumHeads,
        kHeadDim,
        kBlockKV,
        false,
        false,
        kNumQStages,
        kNumKVStages,
        kSplitKV,
        kNumTMAThreads,
        kNumMathThreads,
        true>;
}

void ensure_ready() {
    static bool ready = false;
    if (ready) {
        return;
    }
    cu_check(cuInit(0), "cuInit");
    cuda_check(
        cudaFuncSetAttribute(
            fused_fn(),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            original_smem_bytes()),
        "cudaFuncSetAttribute fused smem");
    auto set_merge = [](auto* fn) {
        cuda_check(
            cudaFuncSetAttribute(
                fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024),
            "cudaFuncSetAttribute merge smem");
    };
    set_merge(dsa_opt::merge_cta_topk_kernel<512, 16>);
    set_merge(dsa_opt::merge_cta_topk_kernel<512, 32>);
    set_merge(dsa_opt::merge_cta_topk_kernel<1024, 16>);
    set_merge(dsa_opt::merge_cta_topk_kernel<1024, 32>);
    set_merge(dsa_opt::merge_cta_topk_kernel<2048, 16>);
    set_merge(dsa_opt::merge_cta_topk_kernel<2048, 32>);
    ready = true;
}

void warmup_fused() {
    ensure_ready();
}

struct TmaPack {
    CUtensorMap q{};
    CUtensorMap kv{};
    CUtensorMap kv_scale{};
    CUtensorMap weights{};
};

struct LaunchPtrs {
    const void* q = nullptr;
    const void* kv = nullptr;
    const void* kv_scale = nullptr;
    const void* weights = nullptr;
    const void* ctx = nullptr;
    const void* table = nullptr;
    const void* sched = nullptr;
    const void* pack_s = nullptr;
    const void* pack_i = nullptr;
    const void* out = nullptr;
    int batch = 0;
    int num_pages = 0;
    int num_sms = 0;
    int max_parts = 0;
    int bt_stride = 0;
    int topk = 0;
};

bool same_launch(const LaunchPtrs& a, const LaunchPtrs& b) {
    return a.q == b.q && a.kv == b.kv && a.kv_scale == b.kv_scale &&
           a.weights == b.weights && a.ctx == b.ctx && a.table == b.table &&
           a.sched == b.sched && a.pack_s == b.pack_s && a.pack_i == b.pack_i &&
           a.out == b.out && a.batch == b.batch && a.num_pages == b.num_pages &&
           a.num_sms == b.num_sms && a.max_parts == b.max_parts &&
           a.bt_stride == b.bt_stride && a.topk == b.topk;
}

TmaPack make_tmas(
    const torch::Tensor& q,
    const torch::Tensor& kv_fp8,
    const torch::Tensor& kv_scale,
    const torch::Tensor& weights,
    int batch,
    int num_pages) {
    TmaPack t;
    t.q = make_tma_2d(
        q, kHeadDim, batch * kNextN * kNumHeads, kHeadDim, kNextN * kNumHeads,
        static_cast<int>(q.stride(2)), kHeadDim);
    t.kv = make_tma_3d(
        kv_fp8, kHeadDim, kBlockKV, num_pages, kHeadDim, kBlockKV, 1,
        static_cast<int>(kv_fp8.stride(1)), static_cast<int>(kv_fp8.stride(0)),
        kHeadDim);
    t.kv_scale = make_tma_2d(
        kv_scale, kBlockKV, num_pages, kBlockKV, 1,
        static_cast<int>(kv_scale.stride(0)), 0);
    t.weights = make_tma_2d(
        weights, kNumHeads, batch * kNextN, kNumHeads, kNextN,
        static_cast<int>(weights.stride(0)), 0);
    return t;
}

bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    return st != cudaStreamCaptureStatusNone;
}

void launch_score(
    FusedKernel fn,
    const TmaPack& tma,
    cudaStream_t stream,
    uint32_t batch,
    uint32_t bt_stride,
    const uint32_t* ctx,
    const uint32_t* table,
    const uint32_t* sched,
    float* pack_s,
    int32_t* pack_i,
    uint32_t max_parts,
    uint32_t num_rows,
    int num_sms,
    int smem) {
    const dim3 grid(num_sms);
    const dim3 block(kNumTMAThreads + kNumMathThreads);
    const auto tq = *reinterpret_cast<const cute::TmaDescriptor*>(&tma.q);
    const auto tkv = *reinterpret_cast<const cute::TmaDescriptor*>(&tma.kv);
    const auto tkvs = *reinterpret_cast<const cute::TmaDescriptor*>(&tma.kv_scale);
    const auto tw = *reinterpret_cast<const cute::TmaDescriptor*>(&tma.weights);
    if (stream_is_capturing(stream)) {
        fn<<<grid, block, smem, stream>>>(
            batch, bt_stride, ctx, table, nullptr, sched, pack_s, pack_i,
            max_parts, num_rows, tq, tkv, tkvs, tw);
        cuda_check(cudaGetLastError(), "sm90_fp8_paged_mqa_logits_fused launch");
        return;
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cuda_check(
        cudaLaunchKernelEx(
            &cfg, fn, batch, bt_stride, ctx, table, nullptr, sched, pack_s, pack_i,
            max_parts, num_rows, tq, tkv, tkvs, tw),
        "sm90_fp8_paged_mqa_logits_fused launch");
}

template <int kTopK, int kIpt>
void launch_merge(
    const float* pack_s,
    const int32_t* pack_i,
    const int32_t* ctx,
    int32_t* out,
    int num_rows,
    int max_parts,
    cudaStream_t stream) {
    ensure_ready();
    auto* merge_fn = dsa_opt::merge_cta_topk_kernel<kTopK, kIpt>;
    if (stream_is_capturing(stream)) {
        merge_fn<<<num_rows, 256, 0, stream>>>(
            pack_s, pack_i, ctx, out, num_rows, max_parts);
        cuda_check(cudaGetLastError(), "merge_cta_topk launch");
        return;
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(num_rows);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cuda_check(
        cudaLaunchKernelEx(&cfg, merge_fn, pack_s, pack_i, ctx, out, num_rows, max_parts),
        "merge_cta_topk launch");
}

template <int kTopK>
void launch_device_topk(
    float* pack_s,
    int32_t* pack_i,
    const int32_t* ctx,
    int32_t* out,
    int num_rows,
    int max_parts,
    cudaStream_t stream) {
    const int stride = max_parts * dsa_opt::kPackWidth;
    const int num_items = num_rows * stride;
    struct DevSortWS {
        torch::Tensor keys_out;
        torch::Tensor vals_out;
        torch::Tensor begin;
        torch::Tensor end;
        torch::Tensor temp;
        size_t temp_bytes = 0;
        int items = 0;
        int rows = 0;
    };
    static DevSortWS ws;
    const auto dev = torch::Device(torch::kCUDA, c10::cuda::current_device());
    auto opts_f = torch::dtype(torch::kFloat).device(dev);
    auto opts_i = torch::dtype(torch::kInt).device(dev);
    auto opts_u8 = torch::dtype(torch::kUInt8).device(dev);
    if (ws.items != num_items || ws.rows != num_rows || ws.keys_out.numel() == 0) {
        TORCH_CHECK(
            !stream_is_capturing(stream),
            "DeviceSegmentedRadixSort workspace alloc during CUDA graph capture");
        ws.keys_out = torch::empty({num_items}, opts_f);
        ws.vals_out = torch::empty({num_items}, opts_i);
        ws.begin = torch::empty({num_rows}, opts_i);
        ws.end = torch::empty({num_rows}, opts_i);
        ws.items = num_items;
        ws.rows = num_rows;
        ws.temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, ws.temp_bytes,
            pack_s, ws.keys_out.data_ptr<float>(),
            pack_i, ws.vals_out.data_ptr<int32_t>(),
            num_items, num_rows,
            ws.begin.data_ptr<int>(), ws.end.data_ptr<int>(),
            0, static_cast<int>(sizeof(float) * 8), stream);
        ws.temp = torch::empty({static_cast<int64_t>(ws.temp_bytes)}, opts_u8);
    }

    dsa_opt::fill_seg_offsets_kernel<<<1, 256, 0, stream>>>(
        ctx, ws.begin.data_ptr<int>(), ws.end.data_ptr<int>(), num_rows, stride);
    cuda_check(cudaGetLastError(), "fill_seg_offsets launch");

    size_t temp_bytes = ws.temp_bytes;
    cuda_check(
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            ws.temp.data_ptr(), temp_bytes,
            pack_s, ws.keys_out.data_ptr<float>(),
            pack_i, ws.vals_out.data_ptr<int32_t>(),
            num_items, num_rows,
            ws.begin.data_ptr<int>(), ws.end.data_ptr<int>(),
            0, static_cast<int>(sizeof(float) * 8), stream),
        "DeviceSegmentedRadixSort");

    dsa_opt::take_seg_topk_kernel<kTopK><<<num_rows, 256, 0, stream>>>(
        ws.vals_out.data_ptr<int32_t>(), ws.begin.data_ptr<int>(),
        ws.end.data_ptr<int>(), out, num_rows);
    cuda_check(cudaGetLastError(), "take_seg_topk launch");
}

template <int kTopK>
void launch_merge_auto(
    const float* pack_s,
    const int32_t* pack_i,
    const int32_t* ctx,
    int32_t* out,
    int num_rows,
    int max_parts,
    cudaStream_t stream) {
    if (max_parts <= 16) {
        launch_merge<kTopK, 16>(pack_s, pack_i, ctx, out, num_rows, max_parts, stream);
        return;
    }
    if (max_parts <= 32) {
        launch_merge<kTopK, 32>(pack_s, pack_i, ctx, out, num_rows, max_parts, stream);
        return;
    }
    launch_device_topk<kTopK>(
        const_cast<float*>(pack_s), const_cast<int32_t*>(pack_i), ctx, out,
        num_rows, max_parts, stream);
}

template <int kTopK>
void launch_fused(
    const torch::Tensor& q,
    const torch::Tensor& kv_fp8,
    const torch::Tensor& kv_scale,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    torch::Tensor& pack_scores,
    torch::Tensor& pack_indices,
    torch::Tensor& out_indices,
    int num_sms,
    int max_parts) {
    const int batch = static_cast<int>(q.size(0));
    const int num_pages = static_cast<int>(kv_fp8.size(0));
    const int block_table_stride = static_cast<int>(block_table.stride(0));
    const int num_rows = batch * kNextN;
    TORCH_CHECK(max_parts * kSplitKV <= 262144, "packed candidates too many");

    LaunchPtrs key;
    key.q = q.data_ptr();
    key.kv = kv_fp8.data_ptr();
    key.kv_scale = kv_scale.data_ptr();
    key.weights = weights.data_ptr();
    key.ctx = context_lens.data_ptr();
    key.table = block_table.data_ptr();
    key.sched = schedule_meta.data_ptr();
    key.pack_s = pack_scores.data_ptr();
    key.pack_i = pack_indices.data_ptr();
    key.out = out_indices.data_ptr();
    key.batch = batch;
    key.num_pages = num_pages;
    key.num_sms = num_sms;
    key.max_parts = max_parts;
    key.bt_stride = block_table_stride;
    key.topk = kTopK;

    struct Cache {
        LaunchPtrs key{};
        TmaPack tma{};
        bool tma_ok = false;
    };
    static Cache cache;

    ensure_ready();
    auto* fn = fused_fn();
    const int smem = original_smem_bytes();

    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (!cache.tma_ok || !same_launch(cache.key, key)) {
        cache.tma = make_tmas(q, kv_fp8, kv_scale, weights, batch, num_pages);
        cache.tma_ok = true;
        cache.key = key;
    }

    launch_score(
        fn, cache.tma, stream, static_cast<uint32_t>(batch),
        static_cast<uint32_t>(block_table_stride),
        reinterpret_cast<const uint32_t*>(context_lens.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(block_table.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(schedule_meta.data_ptr<int32_t>()),
        pack_scores.data_ptr<float>(), pack_indices.data_ptr<int32_t>(),
        static_cast<uint32_t>(max_parts), static_cast<uint32_t>(num_rows),
        num_sms, smem);

    launch_merge_auto<kTopK>(
        pack_scores.data_ptr<float>(), pack_indices.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(), out_indices.data_ptr<int32_t>(),
        num_rows, max_parts, stream);
}

struct Prepared {
    torch::Tensor q;
    torch::Tensor kv_fp8;
    torch::Tensor kv_scale;
    torch::Tensor w;
    torch::Tensor ctx;
    torch::Tensor table;
    torch::Tensor sched;
    int batch = 0;
    int num_sms = 0;
    int max_parts = 0;
};

Prepared prepare_inputs(
    torch::Tensor q,
    torch::Tensor kv_cache,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor schedule_meta) {
    ensure_ready();
    TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(q.dim() == 4);
    TORCH_CHECK(q.size(1) == kNextN && q.size(2) == kNumHeads && q.size(3) == kHeadDim);
    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat);
    TORCH_CHECK(weights.dim() == 2 && weights.size(1) == kNumHeads);
    TORCH_CHECK(weights.stride(1) == 1);
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt);
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt);
    TORCH_CHECK(schedule_meta.is_cuda() && schedule_meta.scalar_type() == torch::kInt);

    const int batch = static_cast<int>(q.size(0));
    TORCH_CHECK(weights.size(0) == batch);
    TORCH_CHECK(seq_lens.numel() == batch);
    TORCH_CHECK(block_table.size(0) == batch && block_table.stride(1) == 1);

    auto kv = kv_cache;
    if (kv.scalar_type() != torch::kByte) {
        kv = kv.view(torch::kUInt8);
    }
    TORCH_CHECK(kv.size(1) == kPageSize);
    TORCH_CHECK(kv.size(-1) == kHeadDimWithScale);
    const int num_pages = static_cast<int>(kv.size(0));
    const int page_stride_bytes = static_cast<int>(kv.stride(0));
    TORCH_CHECK(page_stride_bytes >= kPageSize * kHeadDimWithScale);
    TORCH_CHECK(page_stride_bytes % 4 == 0);

    Prepared p;
    p.q = q;
    p.kv_fp8 = torch::from_blob(
        kv.data_ptr(),
        {num_pages, kPageSize, kHeadDim},
        {page_stride_bytes, kHeadDim, 1},
        kv.options().dtype(torch::kFloat8_e4m3fn));
    p.kv_scale = torch::from_blob(
        kv.data_ptr<uint8_t>() + kPageSize * kHeadDim,
        {num_pages, kPageSize},
        {page_stride_bytes / 4, 1},
        kv.options().dtype(torch::kFloat));
    p.num_sms = static_cast<int>(schedule_meta.size(0)) - 1;
    TORCH_CHECK(schedule_meta.size(1) == 2);
    TORCH_CHECK(p.num_sms > 0);
    TORCH_CHECK(
        seq_lens.is_contiguous() && weights.is_contiguous() &&
            block_table.is_contiguous() && schedule_meta.is_contiguous(),
        "fuse_score_remap tensors must be contiguous (no alloc during CUDA graph capture)");
    p.ctx = seq_lens;
    p.w = weights;
    p.table = block_table;
    p.sched = schedule_meta;
    p.batch = batch;
    const int max_seq = static_cast<int>(block_table.size(1)) * kPageSize;
    p.max_parts = std::max(1, (max_seq + kSplitKV - 1) / kSplitKV);
    return p;
}

void run_score_only(Prepared& p, torch::Tensor& pack_scores, torch::Tensor& pack_indices) {
    const int num_pages = static_cast<int>(p.kv_fp8.size(0));
    const int block_table_stride = static_cast<int>(p.table.stride(0));
    const int num_rows = p.batch * kNextN;
    TORCH_CHECK(p.max_parts * kSplitKV <= 262144, "packed candidates too many");
    TORCH_CHECK(pack_scores.size(0) == p.batch);
    TORCH_CHECK(pack_scores.size(1) == p.max_parts);
    TORCH_CHECK(pack_scores.size(2) == kSplitKV);

    LaunchPtrs key;
    key.q = p.q.data_ptr();
    key.kv = p.kv_fp8.data_ptr();
    key.kv_scale = p.kv_scale.data_ptr();
    key.weights = p.w.data_ptr();
    key.ctx = p.ctx.data_ptr();
    key.table = p.table.data_ptr();
    key.sched = p.sched.data_ptr();
    key.pack_s = pack_scores.data_ptr();
    key.pack_i = pack_indices.data_ptr();
    key.out = nullptr;
    key.batch = p.batch;
    key.num_pages = num_pages;
    key.num_sms = p.num_sms;
    key.max_parts = p.max_parts;
    key.bt_stride = block_table_stride;
    key.topk = 0;

    struct Cache {
        LaunchPtrs key{};
        TmaPack tma{};
        bool tma_ok = false;
    };
    static Cache cache;

    ensure_ready();
    auto* fn = fused_fn();
    const int smem = original_smem_bytes();

    auto stream = c10::cuda::getCurrentCUDAStream(p.q.get_device()).stream();
    if (!cache.tma_ok || !same_launch(cache.key, key)) {
        cache.tma = make_tmas(p.q, p.kv_fp8, p.kv_scale, p.w, p.batch, num_pages);
        cache.tma_ok = true;
        cache.key = key;
    }

    launch_score(
        fn, cache.tma, stream, static_cast<uint32_t>(p.batch),
        static_cast<uint32_t>(block_table_stride),
        reinterpret_cast<const uint32_t*>(p.ctx.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(p.table.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(p.sched.data_ptr<int32_t>()),
        pack_scores.data_ptr<float>(), pack_indices.data_ptr<int32_t>(),
        static_cast<uint32_t>(p.max_parts), static_cast<uint32_t>(num_rows),
        p.num_sms, smem);
}

void gather_physical(
    torch::Tensor pack_indices,
    torch::Tensor logical,
    torch::Tensor out_indices) {
    TORCH_CHECK(pack_indices.is_cuda() && logical.is_cuda() && out_indices.is_cuda());
    TORCH_CHECK(pack_indices.scalar_type() == torch::kInt);
    TORCH_CHECK(logical.scalar_type() == torch::kInt);
    TORCH_CHECK(out_indices.scalar_type() == torch::kInt);
    const int batch = static_cast<int>(logical.size(0));
    const int topk = static_cast<int>(logical.size(1));
    TORCH_CHECK(out_indices.size(0) == batch && out_indices.size(1) == topk);
    const int pack_cols = static_cast<int>(pack_indices.numel() / batch);
    auto stream = c10::cuda::getCurrentCUDAStream(logical.get_device()).stream();
    dsa_opt::gather_physical_kernel<<<batch, 256, 0, stream>>>(
        pack_indices.data_ptr<int32_t>(), logical.data_ptr<int32_t>(),
        out_indices.data_ptr<int32_t>(), pack_cols, topk);
    cuda_check(cudaGetLastError(), "gather_physical launch");
}

void fused_paged_mqa_score(
    torch::Tensor q,
    torch::Tensor kv_cache,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor schedule_meta,
    torch::Tensor pack_scores,
    torch::Tensor pack_indices) {
    TORCH_CHECK(pack_scores.is_cuda() && pack_scores.scalar_type() == torch::kFloat);
    TORCH_CHECK(pack_indices.is_cuda() && pack_indices.scalar_type() == torch::kInt);
    auto p = prepare_inputs(q, kv_cache, weights, seq_lens, block_table, schedule_meta);
    run_score_only(p, pack_scores, pack_indices);
}

void fused_paged_mqa_topk(
    torch::Tensor q,
    torch::Tensor kv_cache,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor schedule_meta,
    torch::Tensor out_indices) {
    ensure_ready();
    TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(q.dim() == 4);
    TORCH_CHECK(q.size(1) == kNextN && q.size(2) == kNumHeads && q.size(3) == kHeadDim);
    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat);
    TORCH_CHECK(weights.dim() == 2 && weights.size(1) == kNumHeads);
    TORCH_CHECK(weights.stride(1) == 1);
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt);
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt);
    TORCH_CHECK(schedule_meta.is_cuda() && schedule_meta.scalar_type() == torch::kInt);
    TORCH_CHECK(out_indices.is_cuda() && out_indices.scalar_type() == torch::kInt);

    const int batch = static_cast<int>(q.size(0));
    const int topk = static_cast<int>(out_indices.size(1));
    TORCH_CHECK(out_indices.size(0) == batch);
    TORCH_CHECK(topk == 512 || topk == 1024 || topk == 2048);
    TORCH_CHECK(weights.size(0) == batch);
    TORCH_CHECK(seq_lens.numel() == batch);
    TORCH_CHECK(block_table.size(0) == batch && block_table.stride(1) == 1);

    auto kv = kv_cache;
    if (kv.scalar_type() != torch::kByte) {
        kv = kv.view(torch::kUInt8);
    }
    TORCH_CHECK(kv.size(1) == kPageSize);
    TORCH_CHECK(kv.size(-1) == kHeadDimWithScale);
    const int num_pages = static_cast<int>(kv.size(0));
    const int page_stride_bytes = static_cast<int>(kv.stride(0));
    TORCH_CHECK(page_stride_bytes >= kPageSize * kHeadDimWithScale);
    TORCH_CHECK(page_stride_bytes % 4 == 0);

    auto kv_fp8 = torch::from_blob(
        kv.data_ptr(),
        {num_pages, kPageSize, kHeadDim},
        {page_stride_bytes, kHeadDim, 1},
        kv.options().dtype(torch::kFloat8_e4m3fn));
    auto kv_scale = torch::from_blob(
        kv.data_ptr<uint8_t>() + kPageSize * kHeadDim,
        {num_pages, kPageSize},
        {page_stride_bytes / 4, 1},
        kv.options().dtype(torch::kFloat));

    const int num_sms = static_cast<int>(schedule_meta.size(0)) - 1;
    TORCH_CHECK(schedule_meta.size(1) == 2);
    TORCH_CHECK(num_sms > 0);

    TORCH_CHECK(seq_lens.is_contiguous() && weights.is_contiguous() &&
                block_table.is_contiguous() && schedule_meta.is_contiguous());
    auto ctx = seq_lens;
    auto w = weights;
    auto table = block_table;
    auto sched = schedule_meta;

    const int max_seq = static_cast<int>(block_table.size(1)) * kPageSize;
    const int max_parts = std::max(1, (max_seq + kSplitKV - 1) / kSplitKV);

    struct PackWS {
        torch::Tensor scores;
        torch::Tensor indices;
    };
    static PackWS ws;
    const auto need = std::vector<int64_t>{batch, max_parts, kSplitKV};
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (ws.scores.numel() == 0 || ws.scores.sizes() != need ||
        ws.scores.device() != q.device()) {
        TORCH_CHECK(
            !stream_is_capturing(stream),
            "fused_paged_mqa_topk PackWS alloc during CUDA graph capture");
        ws.scores = torch::empty(need, q.options().dtype(torch::kFloat));
        ws.indices = torch::empty(need, q.options().dtype(torch::kInt));
    }

    if (topk == 512) {
        launch_fused<512>(
            q, kv_fp8, kv_scale, w, ctx, table, sched, ws.scores, ws.indices,
            out_indices, num_sms, max_parts);
    } else if (topk == 1024) {
        launch_fused<1024>(
            q, kv_fp8, kv_scale, w, ctx, table, sched, ws.scores, ws.indices,
            out_indices, num_sms, max_parts);
    } else {
        launch_fused<2048>(
            q, kv_fp8, kv_scale, w, ctx, table, sched, ws.scores, ws.indices,
            out_indices, num_sms, max_parts);
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_paged_mqa_topk", &fused_paged_mqa_topk);
    m.def("fused_paged_mqa_score", &fused_paged_mqa_score);
    m.def("gather_physical", &gather_physical);
    m.def("warmup_fused", &warmup_fused);
}
