#pragma once

#include <cub/cub.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "utils.cuh"
#include "cuda_kernels/config.h"
#include "cuda_kernels/bit_utils.cuh"

using transac_bar_t = kerutils::transac_bar_t;

namespace topk_select_common {

#ifdef KERUTILS_ENABLE_SM100
    #define IS_LDG_STG_256_AVAILABLE 1
#else
    #define IS_LDG_STG_256_AVAILABLE 0
#endif

// Choose the loading & storing instruction
#if IS_LDG_STG_256_AVAILABLE
    // Use LDG256 / STG256 on sm100+
    static constexpr uint32_t NUM_BYTES_PER_GMEM_LOAD = 256 / 8;
    static constexpr uint32_t NUM_BYTES_PER_GMEM_STORE = 256 / 8;
    #define LOAD_FROM_GMEM(src_ptr, dst_reg) KU_LDG_256(src_ptr, dst_reg, ".nc", "no_allocate", "evict_last", "256B")
    #define STORE_TO_GMEM(dst_ptr, src_reg) KU_STG_256(dst_ptr, src_reg, "no_allocate", "evict_first")
#else
    // Use LDG128 on sm90 and below
    static constexpr uint32_t NUM_BYTES_PER_GMEM_LOAD = 128 / 8;
    static constexpr uint32_t NUM_BYTES_PER_GMEM_STORE = 128 / 8;
    #define LOAD_FROM_GMEM(src_ptr, dst_reg) KU_LDG_128(src_ptr, dst_reg, ".nc", "no_allocate", "256B")
    #define STORE_TO_GMEM(dst_ptr, src_reg) KU_STG_128(dst_ptr, src_reg, "no_allocate", "")
#endif

static constexpr uint32_t NUM_BYTES_PER_SMEM_LOAD = 16;
static constexpr uint32_t NUM_BYTES_PER_SMEM_STORE = 16;


template<uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__
void ld_shared(ValueT res[NUM_VALUES], const ValueT* ptr) {
    constexpr uint32_t NUM_BYTES_TO_LOAD = NUM_VALUES * sizeof(ValueT);
    static_assert(NUM_BYTES_TO_LOAD == 16 || NUM_BYTES_TO_LOAD == 8 || NUM_BYTES_TO_LOAD == 4 || NUM_BYTES_TO_LOAD == 2);
    uint32_t addr = cute::cast_smem_ptr_to_uint(ptr);
    if constexpr (NUM_BYTES_TO_LOAD == 16) {
        *(__int128_t*)res = ku::ld_shared(ptr);
    } else if constexpr (NUM_BYTES_TO_LOAD == 8) {
        asm volatile ("ld.weak.shared::cta.b64 %0, [%1];\n" : "=l"(*(int64_t*)res) : "r"(addr) : "memory");
    } else if constexpr (NUM_BYTES_TO_LOAD == 4) {
        asm volatile ("ld.weak.shared::cta.b32 %0, [%1];\n" : "=r"(*(int32_t*)res) : "r"(addr) : "memory");
    } else if constexpr (NUM_BYTES_TO_LOAD == 2) {
        asm volatile ("ld.weak.shared::cta.b16 %0, [%1];\n" : "=h"(*(int16_t*)res) : "r"(addr) : "memory");
    } else {
        __builtin_unreachable();
    }
}

template<uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__
void ld_shared_with_loop(ValueT res[NUM_VALUES], const ValueT* ptr) {
    constexpr uint32_t NUM_BYTES_TO_LOAD = NUM_VALUES * sizeof(ValueT);
    if constexpr (NUM_BYTES_TO_LOAD <= 16) {
        ld_shared<NUM_VALUES>(res, ptr);
    } else {
        static constexpr uint32_t NUM_LOAD_VALUES_THIS_ROUND = 16 / sizeof(ValueT);
        static_assert(16 % sizeof(ValueT) == 0);
        ld_shared<NUM_LOAD_VALUES_THIS_ROUND>(res, ptr);
        ld_shared_with_loop<NUM_VALUES-NUM_LOAD_VALUES_THIS_ROUND>(res+NUM_LOAD_VALUES_THIS_ROUND, ptr+NUM_LOAD_VALUES_THIS_ROUND);
    }
}

template<uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__
void st_global(ValueT* ptr, ValueT src[NUM_VALUES]) {
    constexpr uint32_t NUM_BYTES_TO_STORE = NUM_VALUES * sizeof(ValueT);
    static_assert((IS_LDG_STG_256_AVAILABLE && NUM_BYTES_TO_STORE == 32) || NUM_BYTES_TO_STORE == 16 || NUM_BYTES_TO_STORE == 8 || NUM_BYTES_TO_STORE == 4 || NUM_BYTES_TO_STORE == 2);
    if constexpr (IS_LDG_STG_256_AVAILABLE && NUM_BYTES_TO_STORE == 32) {
        KU_STG_256(ptr, src, "no_allocate", "evict_first");
    } else if constexpr (NUM_BYTES_TO_STORE == 16) {
        KU_STG_128(ptr, src, "no_allocate", "");
    } else if constexpr (NUM_BYTES_TO_STORE == 8) {
        *(uint64_t*)ptr = *(uint64_t*)src;
    } else if constexpr (NUM_BYTES_TO_STORE == 4) {
        *(uint32_t*)ptr = *(uint32_t*)src;
    } else if constexpr (NUM_BYTES_TO_STORE == 2) {
        *(uint16_t*)ptr = *(uint16_t*)src;
    } else {
        __builtin_unreachable();
    }
}

template<
    typename Config,
    uint32_t MAX_TOPK,
    uint32_t NUM_THREADS,
    uint32_t TARGET_OCCUPANCY
>
struct EpilogueRunner {
    using ValueT = typename Config::ValueT;
    using OutIdxT = typename Config::OutIdxT;
    static_assert(cute::is_same_v<ValueT, float> || cute::is_same_v<ValueT, nv_bfloat16>);
    static_assert(cute::is_same_v<OutIdxT, int32_t> || cute::is_same_v<OutIdxT, int64_t>);
    using UIntValueT = cute::conditional_t<cute::is_same_v<ValueT, float>, uint32_t, uint16_t>;

    static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
    static_assert(NUM_THREADS % 32 == 0);
    static_assert(NUM_WARPS % 2 == 0);

    static constexpr uint32_t NUM_VALUES_PER_LOAD_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(ValueT);
    static constexpr uint32_t NUM_VALUES_PER_THREAD_FOR_SORT = MAX_TOPK / NUM_THREADS;
    static_assert(MAX_TOPK % NUM_THREADS == 0);
    static constexpr uint32_t NUM_REGS_PER_THREAD = (65536 / (NUM_THREADS * TARGET_OCCUPANCY)) / 8 * 8;
    using BlockRadixSortT = cub::BlockRadixSort<
        UIntValueT,
        NUM_THREADS,
        NUM_VALUES_PER_THREAD_FOR_SORT,
        uint32_t,
        4,
        !(NUM_REGS_PER_THREAD <= 80 && NUM_VALUES_PER_THREAD_FOR_SORT >= 16)  // MemoizeOuterScan. Setting this to `false` can reduce register pressure. We use a simple heuristic here
    >;
    // Gated on `sorted_value`: the sort only runs in the `sorted_value` branch, so this keeps the
    // temp storage out of the smem plan of `sorted_value == false` kernels.
    using BlockRadixSortTempStorageT = cute::conditional_t<Config::sorted_value, typename BlockRadixSortT::TempStorage, char>;

    // [SGLang] The page transform is only wired into the unsorted, index-only epilogue
    static_assert(!Config::page_transform || (cute::is_same_v<OutIdxT, int32_t> && !Config::return_value && !Config::sorted_value));

    // [SGLang] Output slot for vocab index `idx` under the page transform. An invalid slot still loads
    // (page 0, which the host guarantees exists) and then stores `fill`, so callers stay branch-free.
    static __device__ __forceinline__ uint32_t page_transform_index(
        uint32_t idx, bool valid, int32_t fill, const int* page_table_row, uint32_t page_bits
    ) {
        idx = valid ? idx : 0u;
        int32_t mapped = (__ldg(page_table_row + (idx >> page_bits)) << page_bits) | (int32_t)(idx & ((1u << page_bits) - 1));
        return (uint32_t)(valid ? mapped : fill);
    }

    static __device__ __forceinline__ uint64_t pack_u32x2(uint32_t lo, uint32_t hi) {
        return (uint64_t)lo | ((uint64_t)hi << 32);
    }

    // [SGLang] The page-transform counterpart of the unsorted index epilogue. Indices are packed into
    // `uint64_t` pairs by hand: routing loaded values through a punned `int32_t` array, as the default
    // path does, puts that array on the stack, and the resulting stack frame also costs the main loop.
    template<bool IS_SHORTCUT>
    static __device__ __forceinline__ void page_transform_epilogue(
        const uint32_t* smem_index_buf, const TopkSelectArgs &args, uint32_t batch_idx, uint32_t end_vocab_idx
    ) {
        int32_t* result_indices = (int32_t*)args.output_index + (uint64_t)batch_idx * args.stride_output_index_batch;
        const int* page_table_row = args.page_table + (uint64_t)batch_idx * args.stride_page_table_batch;
        const uint32_t page_bits = args.page_bits;
        if constexpr (IS_SHORTCUT) {
            constexpr uint32_t NUM_OUTPUT_IDXS_PER_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(int32_t);
            #pragma unroll 2
            for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDXS_PER_STORE; i < args.topk; i += NUM_THREADS * NUM_OUTPUT_IDXS_PER_STORE) {
                uint64_t out[NUM_OUTPUT_IDXS_PER_STORE / 2];
                CUTE_UNROLL
                for (uint32_t j = 0; j < NUM_OUTPUT_IDXS_PER_STORE; j += 2) {
                    out[j/2] = pack_u32x2(
                        page_transform_index(i+j, i+j < end_vocab_idx, args.idx_oob_fill_value, page_table_row, page_bits),
                        page_transform_index(i+j+1, i+j+1 < end_vocab_idx, args.idx_oob_fill_value, page_table_row, page_bits)
                    );
                }
                STORE_TO_GMEM(result_indices + i, out);
            }
        } else {
            page_transform_dispatch<PAGE_TRANSFORM_VEC, 1>(smem_index_buf, result_indices, args.topk, page_table_row, page_bits);
        }
    }

    // [SGLang] Vector width of the staged transform, fixed per kernel: the widest that still gives every thread a
    // round at the top-k this bucket is meant for (512 / 1024, and 2048 for the 4096 bucket). Picking it at runtime
    // instead instantiates the transform once per width, and that extra code alone measurably slows the kernel.
    static constexpr uint32_t PAGE_TRANSFORM_VEC =
        cute::max(cute::min((MAX_TOPK >= 4096 ? 2048u : MAX_TOPK) / NUM_THREADS, 4u), 1u);

    // [SGLang] Resolve the (block-uniform) number of live rounds to a template argument, so that the staged
    // transform below is straight-line code: a branch per round lets the register allocator reuse one round's
    // gather registers for the next, which serializes the rounds on the scoreboard.
    template<uint32_t VEC, uint32_t ROUNDS>
    static __device__ __forceinline__ void page_transform_dispatch(
        const uint32_t* smem_index_buf, int32_t* result_indices, uint32_t topk, const int* page_table_row, uint32_t page_bits
    ) {
        constexpr uint32_t MAX_ROUNDS = cute::max(MAX_TOPK / (NUM_THREADS * VEC), 1u);
        if constexpr (ROUNDS < MAX_ROUNDS) {
            if (topk > ROUNDS * NUM_THREADS * VEC) {
                page_transform_dispatch<VEC, ROUNDS + 1>(smem_index_buf, result_indices, topk, page_table_row, page_bits);
                return;
            }
        }
        page_transform_staged<VEC, ROUNDS>(smem_index_buf, result_indices, topk, page_table_row, page_bits);
    }

    // [SGLang] Transform the staged `smem_index_buf[:topk]`. Thread `t` owns slots `(r * NUM_THREADS + t) * VEC + [0, VEC)`
    // for rounds `r < ROUNDS`. The rounds are unrolled and split into phases -- every smem load, then every
    // page-table gather, then every store -- so each thread keeps all of its gathers in flight at once.
    template<uint32_t VEC, uint32_t ROUNDS>
    static __device__ __forceinline__ void page_transform_staged(
        const uint32_t* smem_index_buf, int32_t* result_indices, uint32_t topk, const int* page_table_row, uint32_t page_bits
    ) {
        static_assert(VEC == 1 || VEC == 2 || VEC == 4);
        // Slots up to ROUNDS * NUM_THREADS * VEC stay inside the MAX_TOPK smem buffer
        static_assert(ROUNDS * NUM_THREADS * VEC <= cute::max(MAX_TOPK, NUM_THREADS * VEC));

        uint32_t idx[ROUNDS][VEC];
        CUTE_UNROLL
        for (uint32_t r = 0; r < ROUNDS; ++r) {
            const uint32_t* src = smem_index_buf + (r * NUM_THREADS + threadIdx.x) * VEC;
            if constexpr (VEC == 4) {
                __int128_t v = ku::ld_shared(src);
                CUTE_UNROLL
                for (uint32_t j = 0; j < 4; ++j) idx[r][j] = (uint32_t)(v >> (32 * j));
            } else if constexpr (VEC == 2) {
                uint64_t v = *(const uint64_t*)src;
                idx[r][0] = (uint32_t)v;
                idx[r][1] = (uint32_t)(v >> 32);
            } else {
                idx[r][0] = *src;
            }
        }
        // Slots past `topk` hold stale smem, which must not index the page table
        CUTE_UNROLL
        for (uint32_t r = 0; r < ROUNDS; ++r) {
            const uint32_t base = (r * NUM_THREADS + threadIdx.x) * VEC;
            CUTE_UNROLL
            for (uint32_t j = 0; j < VEC; ++j)
                idx[r][j] = page_transform_index(idx[r][j], base + j < topk, 0, page_table_row, page_bits);
        }
        // The host requires `topk` to end on a 32-byte boundary, so a live vector never crosses it
        CUTE_UNROLL
        for (uint32_t r = 0; r < ROUNDS; ++r) {
            const uint32_t base = (r * NUM_THREADS + threadIdx.x) * VEC;
            if (base < topk) {
                if constexpr (VEC == 1) {
                    result_indices[base] = (int32_t)idx[r][0];
                } else {
                    uint64_t out[VEC / 2];
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < VEC; j += 2) out[j / 2] = pack_u32x2(idx[r][j], idx[r][j + 1]);
                    st_global<VEC / 2>((uint64_t*)(result_indices + base), out);
                }
            }
        }
    }

    template<bool IS_SHORTCUT>  // "shortcut" means `end_vocab_idx` <= `topk`
    static __device__ __forceinline__ void topk_select_epilogue(
        typename Config::ValueT* smem_value_buf,    // [MAX_TOPK]
        uint32_t* smem_index_buf,   // [MAX_TOPK]
        const TopkSelectArgs &args,
        uint32_t batch_idx,
        uint32_t end_vocab_idx,
        uint32_t warp_idx,
        BlockRadixSortTempStorageT &radix_sort_temp_storage
    ) {
        // General epilogue for topk select
        // This epilogue has two modes: shortcut mode and non-shortcut mode.
        // In shortcut mode, it
        //  - Generates indices
        //  - Load values from global memory, if `Config::return_value` is `true`
        //  - Perform sorting, if `Config::sorted_value` is `true`
        //  - Save indices & values to global memory
        // In non-shortcut mode, it
        //  - Assume end_vocab_idx > topk holds, i.e. there are enough values for topk selection
        //  - Assume topk indices locate at smem_index_buf[:topk] and topk values locate at smem_value_buf[:topk] (an extra `__syncthreads()` might be necessary)
        //  - Perform sorting, if `Config::sorted_value` is `true`
        //  - Save indices & values to global memory (values are un-distorted if `UNDISTORT_VALUES` is `true`)

        ValueT* input_values = Config::return_value ? (ValueT*)args.input + (uint64_t)batch_idx * args.stride_input_batch : nullptr;
        OutIdxT* result_indices = (OutIdxT*)args.output_index + (uint64_t)batch_idx * args.stride_output_index_batch;
        ValueT* result_values = Config::return_value ? (ValueT*)args.output_value + (uint64_t)batch_idx * args.stride_output_value_batch : nullptr;
        int32_t output_idx_offset = args.output_idx_offset != nullptr ? __ldg(args.output_idx_offset + batch_idx) : 0;
        ValueT oob_fill_value = Config::return_value && IS_SHORTCUT ? (ValueT)args.value_oob_fill_value : (ValueT)0.0f;

        if constexpr (Config::page_transform) {
            page_transform_epilogue<IS_SHORTCUT>(smem_index_buf, args, batch_idx, end_vocab_idx);
        } else if constexpr (!Config::sorted_value) {
            if constexpr (IS_SHORTCUT) {
                // Saves indices 0...end_vocab_idx (plus `output_idx_offset`) followed by (args.topk-end_vocab_idx) `args.idx_oob_fill_value` into `result_indices`
                constexpr uint32_t NUM_OUTPUT_IDXS_PER_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(OutIdxT);
                #pragma unroll 2
                for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDXS_PER_STORE; i < args.topk; i += NUM_THREADS * NUM_OUTPUT_IDXS_PER_STORE) {
                    OutIdxT out[NUM_OUTPUT_IDXS_PER_STORE];
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < NUM_OUTPUT_IDXS_PER_STORE; ++j)
                        out[j] = i+j < end_vocab_idx ? (OutIdxT)(i+j) + output_idx_offset : args.idx_oob_fill_value;
                    STORE_TO_GMEM(result_indices + i, out);
                }

                // Save values into `result_values`, if `RETURN_VALUE` is `True`
                if constexpr (Config::return_value) {
                    #pragma unroll 2
                    for (uint32_t i = threadIdx.x * NUM_VALUES_PER_LOAD_STORE; i < args.topk; i += NUM_THREADS * NUM_VALUES_PER_LOAD_STORE) {
                        ValueT values[NUM_VALUES_PER_LOAD_STORE];
                        if (i < end_vocab_idx) {
                            LOAD_FROM_GMEM(input_values + i, values);
                        }
                        CUTE_UNROLL
                        for (uint32_t j = 0; j < NUM_VALUES_PER_LOAD_STORE; ++j)
                            values[j] = i+j < end_vocab_idx ? values[j] : oob_fill_value;
                        STORE_TO_GMEM(result_values + i, values);
                    }
                }
            } else {
                // Load & cast & save indices
                constexpr uint32_t NUM_OUTPUT_IDX_PER_ROUND = cute::min(NUM_BYTES_PER_SMEM_STORE / sizeof(uint32_t), NUM_BYTES_PER_GMEM_STORE / sizeof(OutIdxT));
                #pragma unroll 2
                for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDX_PER_ROUND; i < args.topk; i += NUM_THREADS * NUM_OUTPUT_IDX_PER_ROUND) {
                    uint32_t indices_u32[NUM_OUTPUT_IDX_PER_ROUND];
                    OutIdxT out[NUM_OUTPUT_IDX_PER_ROUND];
                    ld_shared<NUM_OUTPUT_IDX_PER_ROUND>(indices_u32, smem_index_buf + i);
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < NUM_OUTPUT_IDX_PER_ROUND; ++j)
                        out[j] = (OutIdxT)indices_u32[j] + output_idx_offset;
                    st_global<NUM_OUTPUT_IDX_PER_ROUND>(result_indices + i, out);
                }

                if constexpr (Config::return_value) {
                    // Load & store values
                    constexpr uint32_t NUM_VALUES_PER_ROUND = cute::min(NUM_BYTES_PER_SMEM_STORE, NUM_BYTES_PER_GMEM_STORE) / sizeof(ValueT);
                    #pragma unroll 2
                    for (uint32_t i = threadIdx.x * NUM_VALUES_PER_ROUND; i < args.topk; i += NUM_THREADS * NUM_VALUES_PER_ROUND) {
                        UIntValueT values[NUM_VALUES_PER_ROUND];
                        ld_shared<NUM_VALUES_PER_ROUND>(values, (UIntValueT*)(smem_value_buf + i));
                        st_global<NUM_VALUES_PER_ROUND>((UIntValueT*)(result_values + i), values);
                    }
                }
            }
        } else {
            // Load indices & values into shared memory
            if constexpr (IS_SHORTCUT) {
                if (warp_idx < NUM_WARPS/2) {
                    #pragma unroll 8
                    for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS/2) {
                        smem_index_buf[i] = i < end_vocab_idx ? i : args.idx_oob_fill_value - output_idx_offset;
                    }
                } else {
                    constexpr uint32_t NUM_VALUES_PER_LDG128 = NUM_BYTES_PER_SMEM_LOAD / sizeof(ValueT);
                    #pragma unroll 2
                    for (uint32_t i = (threadIdx.x-NUM_THREADS/2) * NUM_VALUES_PER_LDG128; i < end_vocab_idx; i += (NUM_THREADS/2) * NUM_VALUES_PER_LDG128) {
                        ValueT values[NUM_VALUES_PER_LDG128];
                        KU_LDG_128(input_values + i, values, ".nc", "no_allocate", "256B");
                        ku::st_shared(smem_value_buf + i, *(__int128_t*)values);
                    }
                }
                __syncthreads();
            }

            // Sort
            UIntValueT local_values[NUM_VALUES_PER_THREAD_FOR_SORT];
            uint32_t local_indices[NUM_VALUES_PER_THREAD_FOR_SORT];
            uint32_t thread_offset = threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT;
            ld_shared_with_loop<NUM_VALUES_PER_THREAD_FOR_SORT>(local_values, (UIntValueT*)smem_value_buf + threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT);
            ld_shared_with_loop<NUM_VALUES_PER_THREAD_FOR_SORT>(local_indices, smem_index_buf + threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT);
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                local_values[i] = distort(local_values[i]);
            if (IS_SHORTCUT || args.topk < MAX_TOPK) {
                uint32_t limit = IS_SHORTCUT ? end_vocab_idx : args.topk;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                    local_values[i] = thread_offset + i < limit ? local_values[i] : 0;
            }
            __syncthreads();    // To allow `radix_sort_temp_storage` overlap with `smem_value_buf` and `smem_index_buf`
            BlockRadixSortT(radix_sort_temp_storage).SortDescending(local_values, local_indices);

            // Store directly to gmem (no smem buffering)
            // For common cases, our MAX_TOPK <= 1024 and NUM_THREADS >= 256, so we have NUM_VALUES_PER_THREAD_FOR_SORT <= 8, which means each thread only need to call STG once. So we don't use SMEM to buffer here
            if (thread_offset < args.topk) {
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                    local_values[i] = !IS_SHORTCUT || threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT + i < end_vocab_idx ? un_distort(local_values[i]) : *(UIntValueT*)&oob_fill_value;
                OutIdxT local_indices_new[NUM_VALUES_PER_THREAD_FOR_SORT];
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                    local_indices_new[i] = (OutIdxT)((int32_t)local_indices[i] + output_idx_offset);
                auto store = [&]<uint32_t NUM_VALUES, typename T>(T* dst, T src[NUM_VALUES]) {
                    constexpr uint32_t NUM_BYTES_TO_STORE = NUM_VALUES * sizeof(T);
                    if constexpr (NUM_BYTES_TO_STORE <= NUM_BYTES_PER_GMEM_STORE) {
                        // Don't need to do OOB check because the output array is at least padded to the maximum width of global store.
                        st_global<NUM_VALUES>(dst, src);
                    } else {
                        static_assert(NUM_BYTES_TO_STORE % NUM_BYTES_PER_GMEM_STORE == 0);
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_BYTES_TO_STORE; i += NUM_BYTES_PER_GMEM_STORE) {
                            uint32_t elem_offset = i / sizeof(T);
                            if (thread_offset + elem_offset < args.topk)
                                STORE_TO_GMEM(dst+elem_offset, src+elem_offset);
                        }
                    }
                };
                store.template operator()<NUM_VALUES_PER_THREAD_FOR_SORT>((UIntValueT*)result_values + thread_offset, local_values);
                store.template operator()<NUM_VALUES_PER_THREAD_FOR_SORT>(result_indices + thread_offset, local_indices_new);
            }
        }
    }
};

/*
// TODO Give a brief introduction to the algorithm here
*/
template<typename Config>
class TopkSelectKernelBase {
public:
    using ValueT = Config::ValueT;
    using OutIdxT = Config::OutIdxT;
    static_assert(std::is_same_v<ValueT, nv_bfloat16> || std::is_same_v<ValueT, float>);

    static constexpr uint32_t PLACEHOLDER_U32 = std::is_same_v<ValueT, nv_bfloat16> ? 0xff80 : 0xff800000; // -INF
    static constexpr uint64_t PLACEHOLDER_B64 = std::is_same_v<ValueT, nv_bfloat16> ? 0xff80ff80ff80ff80 : 0xff800000ff800000;
    static constexpr uint64_t PLACEHOLDER_PAIR = (uint64_t)PLACEHOLDER_U32 << 32;

    static constexpr uint32_t TARGET_OCCUPANCY = Config::target_occupancy;
    static constexpr uint32_t NUM_THREADS = Config::num_threads;
    static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
    static constexpr uint32_t MAX_TOPK = Config::max_topk;
    static_assert(NUM_THREADS == 256 || NUM_THREADS == 512);

    static constexpr uint32_t NUM_ELEMS_PER_128b = 16 / sizeof(ValueT);
    static constexpr uint32_t NUM_UINT32_PER_128b = 4;
    static constexpr uint32_t NUM_ELEMS_PER_ROUND = Config::elements_per_round;
    static constexpr uint32_t NUM_ELEMS_PER_THREAD_PER_ROUND = NUM_ELEMS_PER_ROUND / NUM_THREADS;   // TODO Unify "values", "elements", and "elems" to "elems"
    static constexpr uint32_t NUM_128b_PER_THREAD_PER_ROUND = NUM_ELEMS_PER_THREAD_PER_ROUND / NUM_ELEMS_PER_128b;
    static_assert(NUM_ELEMS_PER_ROUND % (NUM_ELEMS_PER_128b * NUM_THREADS) == 0);
    static constexpr uint32_t ELEMS_PER_THREAD_PER_ROUND_MASK = NUM_ELEMS_PER_THREAD_PER_ROUND - 1;
    // NUM_128b_PER_THREAD_PER_ROUND is the canonical name; no separate NUM_UNITS alias is kept.

    static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND == 16);

    // Each segment/window contains this many elements. It is the granularity used by both the
    // pseudo-random visit order and the sorted-index epilogue.
    static constexpr uint32_t NUM_ELEMS_PER_SEG = Config::elements_per_segment; // shuffle rule
    static_assert(NUM_ELEMS_PER_SEG == 512);
    // The last NUM_TAIL_ELEMS elements of a row are always visited in their original order.
    // 4096 keeps the tail small enough to fit alongside the init-window perm segments.
    static constexpr uint32_t NUM_TAIL_ELEMS = 4096;
    static constexpr uint32_t NUM_TAIL_SEGS = NUM_TAIL_ELEMS / NUM_ELEMS_PER_SEG;
    // The init window is 32 KiB of input data
    static constexpr uint32_t NUM_ELEMS_IN_INIT_WINDOW = 32768 / sizeof(ValueT);
    static constexpr uint32_t NUM_INIT_ROUNDS_MAX = NUM_ELEMS_IN_INIT_WINDOW / NUM_ELEMS_PER_ROUND;
    static_assert(NUM_ELEMS_IN_INIT_WINDOW % NUM_ELEMS_PER_ROUND == 0);
    static_assert(NUM_ELEMS_IN_INIT_WINDOW % NUM_ELEMS_PER_SEG == 0);

    static constexpr uint32_t NUM_SEGS_PER_ROUND = NUM_ELEMS_PER_ROUND / NUM_ELEMS_PER_SEG;
    static_assert(NUM_ELEMS_PER_ROUND % NUM_ELEMS_PER_SEG == 0);
    static_assert(NUM_ELEMS_PER_SEG % NUM_ELEMS_PER_128b == 0);
    static constexpr uint32_t NUM_128b_PER_SEG = NUM_ELEMS_PER_SEG / NUM_ELEMS_PER_128b;
    static_assert(NUM_THREADS % NUM_128b_PER_SEG == 0);
    static_assert(NUM_128b_PER_SEG % 32 == 0); // every lane in a warp should in the same seg

    static constexpr uint32_t NUM_ISSUE_WARPS = NUM_SEGS_PER_ROUND < 4 ? NUM_SEGS_PER_ROUND : 4;
    static constexpr uint32_t NUM_SEGS_PER_ISSUE_WARP = NUM_SEGS_PER_ROUND / NUM_ISSUE_WARPS;
    static_assert(NUM_SEGS_PER_ROUND % NUM_ISSUE_WARPS == 0);
    static_assert(NUM_TAIL_SEGS % NUM_ISSUE_WARPS == 0);
    static_assert(NUM_WARPS >= NUM_ISSUE_WARPS);

    static constexpr uint32_t NUM_TMA_LOAD_BUFS = Config::tma_buffer_depth;
    static constexpr uint32_t TMA_PREFETCH_DEPTH = NUM_TMA_LOAD_BUFS - 1;
    static_assert(NUM_TMA_LOAD_BUFS >= 2);

    static constexpr uint32_t RECONSTRUCT_THRESHOLD = Config::reconstruct_threshold;
    static_assert(RECONSTRUCT_THRESHOLD >= MAX_TOPK);

    // Parameters of the pseudo-random permutation over permuted segments:
    //   permuted_id = (linear_id * (PERM_MUL_PRIME % perm_len) + PERM_ADD_BASE) % perm_len.
    static constexpr uint32_t PERM_ADD_BASE = 0x22262226u;
    static constexpr uint32_t PERM_MUL_PRIME = 0xB559EB75u;

    static constexpr uint32_t NUM_SEGS_IN_INIT_WINDOW = NUM_ELEMS_IN_INIT_WINDOW / NUM_ELEMS_PER_SEG;
    // Number of permuted segments included in the init window after reserving room for the tail.
    static constexpr uint32_t NUM_PERM_SEGS_IN_INIT_WINDOW =
        (NUM_ELEMS_IN_INIT_WINDOW - NUM_TAIL_ELEMS) / NUM_ELEMS_PER_SEG;
    static_assert(NUM_PERM_SEGS_IN_INIT_WINDOW % NUM_TAIL_SEGS == 0);
    static_assert(NUM_PERM_SEGS_IN_INIT_WINDOW > 0);
    static constexpr uint32_t NUM_UINT32_IN_INIT_WINDOW_PER_THREAD = NUM_ELEMS_IN_INIT_WINDOW * sizeof(ValueT) / sizeof(uint32_t) / NUM_THREADS;
    static constexpr uint32_t NUM_128b_INIT_PER_THREAD = NUM_UINT32_IN_INIT_WINDOW_PER_THREAD / NUM_UINT32_PER_128b;
    static_assert(NUM_ELEMS_IN_INIT_WINDOW % (sizeof(uint32_t) / sizeof(ValueT) * NUM_THREADS) == 0);
    static_assert(NUM_UINT32_IN_INIT_WINDOW_PER_THREAD % (16 * sizeof(ValueT) / sizeof(uint32_t)) == 0);
    static_assert(2 * NUM_UINT32_IN_INIT_WINDOW_PER_THREAD <= 256);

    static constexpr uint32_t NUM_EXTRA_SLOTS = RECONSTRUCT_THRESHOLD + NUM_ELEMS_PER_ROUND;

    static constexpr uint32_t NUM_RECONSTRUCT_RADIX_BITS = 8;
    static constexpr uint32_t NUM_RECONSTRUCT_BUCKETS = 1u << NUM_RECONSTRUCT_RADIX_BITS;
    // Each row also holds one sink slot (index NUM_RECONSTRUCT_BUCKETS) that the LSB histogram sends the
    // out-of-bucket elements to; the +4 rounds a row up to a multiple of 16 bytes so that both rows stay
    // 16B-aligned, since the reader relies on for its 128-bit shared-memory loads.
    static constexpr uint32_t NUM_RECONSTRUCT_BUCKET_SLOTS = NUM_RECONSTRUCT_BUCKETS + 4;
    static_assert(sizeof(uint32_t) * NUM_RECONSTRUCT_BUCKET_SLOTS % 16 == 0);
    static_assert(MAX_TOPK + NUM_EXTRA_SLOTS <= 0xFFFF);
    static_assert(NUM_EXTRA_SLOTS % 2 == 0);

    static constexpr uint32_t NUM_RECONSTRUCT_UNITS_MAX = (MAX_TOPK + NUM_EXTRA_SLOTS) / 2;

    using EpilogueT = EpilogueRunner<Config, MAX_TOPK, NUM_THREADS, TARGET_OCCUPANCY>;
    static_assert(sizeof(typename EpilogueT::BlockRadixSortTempStorageT) <= NUM_EXTRA_SLOTS * sizeof(uint64_t));

    static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) >= NUM_ELEMS_IN_INIT_WINDOW * sizeof(ValueT),
                "the init window must fit in the extra pairs region");
    static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) / sizeof(uint32_t) >= MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG,
                "the index sort's window delta table must fit in the extra pairs region");
    // array `incoming_topk_pairs` will be used as
    //     1. tma buffer during init phase
    //     2. extra array during main phase
    //     3. windows_delta counter during epilogue phase
    static_assert(2 * MAX_TOPK * sizeof(uint64_t) % 1024 == 0);

    static constexpr uint32_t SWIZZLE_SHIFT = 3;
    static constexpr uint32_t SWIZZLE_MASK = 7 * (16 / sizeof(ValueT));

    static constexpr uint32_t NUM_BYTES_PER_TMA_ROW = 128;  // Limit the innermost box dim of TMA to 1) prevent OOB 2) swizzling has limitations on the innermost box dim
    static_assert(NUM_BYTES_PER_TMA_ROW <= INPUT_STRIDE_ALIGNMENT_REQUIREMENT); // To prevent OOB
    static constexpr uint32_t NUM_ELEMS_PER_TMA_ROW = NUM_BYTES_PER_TMA_ROW / sizeof(ValueT);
    static_assert(NUM_ELEMS_PER_SEG % NUM_ELEMS_PER_TMA_ROW == 0);
    static constexpr uint32_t NUM_TMA_ROWS_PER_SEG = NUM_ELEMS_PER_SEG / NUM_ELEMS_PER_TMA_ROW;
    static_assert(SWIZZLE_MASK == ((1u << SWIZZLE_SHIFT) - 1) * NUM_ELEMS_PER_128b);
    static_assert(NUM_ELEMS_PER_TMA_ROW == (1u << SWIZZLE_SHIFT) * NUM_ELEMS_PER_128b);
    static_assert(NUM_TMA_ROWS_PER_SEG % (1u << SWIZZLE_SHIFT) == 0);

    static_assert(NUM_SEGS_PER_ROUND == NUM_WARPS);
    static_assert(NUM_SEGS_PER_ROUND >= NUM_TAIL_SEGS);
    static_assert(NUM_ELEMS_PER_SEG % (Config::elements_per_round/NUM_WARPS) == 0);

    struct TmaParams {
        CUtensorMap tensor_map;
    };

    static CUtensorMap make_topk_tensor_map(const TopkSelectArgs &args) {
        static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT >= NUM_ELEMS_PER_TMA_ROW * sizeof(ValueT));
        constexpr CUtensorMapDataType dtype = std::is_same_v<ValueT, nv_bfloat16>
            ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            : CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        return ku::make_tensor_map(
            // Split args.vocab_size into two dims because
            //   - TMA requires innermost box dim <= swizzle size
            //   - To avoid OOB as we have `INPUT_STRIDE_ALIGNMENT_REQUIREMENT`
            {
                NUM_ELEMS_PER_TMA_ROW,
                ku::ceil_div((uint64_t)args.vocab_size, (uint64_t)NUM_ELEMS_PER_TMA_ROW),
                args.batch_size
            },
            ku::make_stride_helper<uint64_t>({NUM_ELEMS_PER_TMA_ROW, args.stride_input_batch}, sizeof(ValueT)),
            {
                NUM_ELEMS_PER_TMA_ROW,
                NUM_TMA_ROWS_PER_SEG,
                1
            },
            args.input,
            dtype,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    struct SharedMemoryPlanBase {
        // bf16: pair(64b) = unused(16b) | value(16b) | index(32b)
        // fp32: pair(64b) = value(32b) | index(32b)
        // candidate[0]/candidate[1] are the A/B buffers swapped on each reconstruct;
        // incoming holds the pairs scanned since the last reconstruct.
        CUTE_ALIGNAS(1024) uint64_t surviving_topk_pairs[2][MAX_TOPK];
        CUTE_ALIGNAS(1024) uint64_t incoming_topk_pairs[NUM_EXTRA_SLOTS];
        CUTE_ALIGNAS(1024) ValueT tma_load_buf[NUM_TMA_LOAD_BUFS][NUM_ELEMS_PER_ROUND];
        transac_bar_t tma_load_full_bar[NUM_TMA_LOAD_BUFS];
        transac_bar_t init_full_bar[NUM_INIT_ROUNDS_MAX];
        uint32_t warp_cnt[NUM_WARPS];
        uint32_t reconstruct_pivot_bucket;
        uint32_t reconstruct_num_should_select;
        CUTE_ALIGNAS(16) uint32_t reconstruct_bucket_counter[2][NUM_RECONSTRUCT_BUCKET_SLOTS];
    };

    // Map a linear visit index to the actual permuted segment id.
    static __device__ __forceinline__
    uint32_t get_permuted_seg_idx(uint32_t linear_id, uint32_t perm_len, uint32_t perm_mul) {
        return ((linear_id % perm_len) * perm_mul + PERM_ADD_BASE) % perm_len;
    }

    static __device__ __forceinline__
    void advance_perm_state(uint32_t &state, uint32_t stride, uint32_t perm_len) {
        state += stride;
        if (state >= perm_len) state -= perm_len;
    }

    static __device__ __forceinline__
    uint32_t sw_msk(uint32_t e) {
        return (e >> SWIZZLE_SHIFT) & SWIZZLE_MASK;
    }
    static __device__ __forceinline__
    uint32_t sw_elem(uint32_t e) {
        return e ^ sw_msk(e);
    }
    static __device__ __forceinline__
    uint32_t sw_b128(uint32_t u) {
        return u ^ ((u >> 3) & (7));
    }

    struct EqGtPrefix { uint32_t start_pos_in_collector; uint32_t eq_quota; };
    // Given local > pivot and == pivot counts, compute this thread's output prefix (start position in the final output buffer) and how many equal
    // elements it may still append (eq_quota).
    // Should be called by all threads
    // cnt_gt and cnt_eq must be < 65536.
    static __device__ __forceinline__
    EqGtPrefix compute_equal_quota_and_prefix(uint32_t cnt_gt, uint32_t cnt_eq, uint32_t topk, uint32_t warp_idx, uint32_t lane_idx, uint32_t *warp_cnt) {
        static_assert(NUM_WARPS <= 32);
        uint32_t cnt_packed = (cnt_gt << 16) | cnt_eq;
        uint32_t warp_total_packed = __reduce_add_sync(0xFFFFFFFF, cnt_packed);
        if (lane_idx == 0) {
            warp_cnt[warp_idx] = warp_total_packed;
        }
        __syncthreads();

        uint32_t off_lane_packed = warp_level_exclusive_prefix_sum(cnt_packed, lane_idx);
        uint32_t stored_warp_packed = lane_idx < NUM_WARPS ? warp_cnt[lane_idx] : 0u;
        uint32_t off_warp_packed = __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_packed : 0u);
        uint32_t total_packed = __reduce_add_sync(0xFFFFFFFF, stored_warp_packed);
        uint32_t num_total_ge = total_packed >> 16;
        uint32_t num_total_eq_quota = topk - num_total_ge;
        uint32_t off_packed = off_warp_packed + off_lane_packed;
        uint32_t num_gt_before = off_packed >> 16;
        uint32_t num_eq_before = off_packed & 0xFFFFu;
        uint32_t eq_quota = num_total_eq_quota > num_eq_before ? num_total_eq_quota - num_eq_before : 0u;
        uint32_t num_elems_should_select_before = num_gt_before + min(num_eq_before, num_total_eq_quota);
        return {num_elems_should_select_before, eq_quota};
    }

    // Load this thread's slice from the swizzled smem round buffer into registers.
    template<uint32_t NUM_128b>
    static __device__ __forceinline__
    void load_swizzled_slice(
        ValueT dst[NUM_128b * NUM_ELEMS_PER_128b],
        const ValueT *smem_buf,
        uint32_t smem_slice_elem_offset, // element offset into smem_buf where this thread's slice starts
        uint32_t unit_swizzle_mask,      // swizzle mask applied to each 128-bit unit inside the slice
        uint32_t first_128b_unit_in_slice = 0 // which 128-bit unit of this slice to start loading from
    ) {
        CUTE_UNROLL
        for (uint32_t u = 0; u < NUM_128b; ++u) {
            ld_shared<NUM_ELEMS_PER_128b>(
                dst + u * NUM_ELEMS_PER_128b,
                smem_buf + smem_slice_elem_offset +
                    (((first_128b_unit_in_slice + u) * NUM_ELEMS_PER_128b) ^ unit_swizzle_mask));
        }
    }

    // Find the bucket id that the topk-th element resides in, and get how many elements we should select from that bucket
    // Save result to `smem.reconstruct_pivot_bucket` and `smem.reconstruct_num_should_select`
    // If `CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET`, return whether we should select all elements in that bucket.
    // Should be called by a whole warp
    // [SGLang] Any plan with `reconstruct_pivot_bucket` and `reconstruct_num_should_select` will do
    template<bool CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET, typename PlanT>
    static __device__ __forceinline__
    bool find_pivot_in_histogram(PlanT &smem, const uint32_t *bucket_counter, uint32_t topk, uint32_t lane_idx) {
        bool should_select_whole_bucket = false;
        // Each lane loads 8 counters
        static_assert(NUM_RECONSTRUCT_BUCKETS == 32 * 8);
        uint32_t counts[8];
        const uint32_t *bucket_ptr = bucket_counter + lane_idx * 8;
        ld_shared<4>(counts, bucket_ptr);
        ld_shared<4>(counts + 4, bucket_ptr + 4);

        uint32_t local_sum = 0;
        CUTE_UNROLL
        for (uint32_t j = 0; j < 8; ++j)
            local_sum += counts[j];

        // Get the inclusive suffix sum of the histogram
        uint32_t suffix_count[9];
        suffix_count[8] = warp_level_exclusive_suffix_sum(local_sum, lane_idx);
        CUTE_UNROLL
        for (int32_t j = 7; j >= 0; --j)
            suffix_count[j] = suffix_count[j + 1] + counts[j];

        if (suffix_count[8] < topk && topk <= suffix_count[0]) {
            uint32_t j = 0;
            CUTE_UNROLL
            for (uint32_t k = 1; k < 8; ++k)
                j += (uint32_t)(suffix_count[k] >= topk);

            // A depth-3 SEL tree picking suffix_count[j+1] by the bits of j
            // j is the bucket that contains the top-k element
            uint32_t b0 = (j & 1) ? suffix_count[2] : suffix_count[1], b1 = (j & 1) ? suffix_count[4] : suffix_count[3];
            uint32_t b2 = (j & 1) ? suffix_count[6] : suffix_count[5], b3 = (j & 1) ? suffix_count[8] : suffix_count[7];
            b0 = (j & 2) ? b1 : b0;  b2 = (j & 2) ? b3 : b2;
            uint32_t suffix_count_j_plus_1 = (j & 4) ? b2 : b0;   // = s[j+1]
            smem.reconstruct_pivot_bucket = lane_idx * 8 + j;
            smem.reconstruct_num_should_select = topk - suffix_count_j_plus_1;

            if constexpr (CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET) {
                // topk == suffix_count[j] means the whole pivot bucket is selected
                uint32_t d0 = (j & 1) ? suffix_count[1] : suffix_count[0], d1 = (j & 1) ? suffix_count[3] : suffix_count[2];
                uint32_t d2 = (j & 1) ? suffix_count[5] : suffix_count[4], d3 = (j & 1) ? suffix_count[7] : suffix_count[6];
                d0 = (j & 2) ? d1 : d0;  d2 = (j & 2) ? d3 : d2;
                uint32_t suffix_count_j = (j & 4) ? d2 : d0;    // = suffix_count[j]
                should_select_whole_bucket = (topk == suffix_count_j);
            }
        }

        if constexpr (CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET) {
            // The owning lane holds the result; reduce so all lanes in the warp see the same answer.
            return __reduce_or_sync(0xFFFFFFFFu, should_select_whole_bucket);
        }
        return false;
    }

    // Clean every bucket counter using 16-byte shared-memory writes.
    // Need to be called by all threads.
    static __device__ __forceinline__
    void clear_reconstruct_histograms(SharedMemoryPlanBase &smem, uint32_t thread_idx) {
        static_assert((2 * NUM_RECONSTRUCT_BUCKET_SLOTS) % 4 == 0);
        __int128_t zero = (__int128_t)0ull;
        for (uint32_t i = thread_idx; i < (2 * NUM_RECONSTRUCT_BUCKET_SLOTS) / 4; i += NUM_THREADS) {
            reinterpret_cast<__int128_t*>(smem.reconstruct_bucket_counter)[i] = zero;
        }
    }

    template<typename ExtraBarInitF>
    static __device__ __forceinline__
    void init_shared_memory(SharedMemoryPlanBase &smem, uint32_t warp_idx, ExtraBarInitF &&init_extra_bar_func) {
        // Pad unused elements in `surviving_topk_pairs` as `-INF` to avoid being selected
        static_assert((2 * MAX_TOPK) % NUM_THREADS == 0);
        CUTE_UNROLL
        for (uint32_t i = 0; i < MAX_TOPK / NUM_THREADS; ++i) {
            smem.surviving_topk_pairs[0][i * NUM_THREADS + threadIdx.x] = PLACEHOLDER_PAIR;
            smem.surviving_topk_pairs[1][i * NUM_THREADS + threadIdx.x] = PLACEHOLDER_PAIR;
        }

        clear_reconstruct_histograms(smem, threadIdx.x);

        if (warp_idx == 0 && cute::elect_one_sync()) {
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_TMA_LOAD_BUFS; ++i) {
                smem.tma_load_full_bar[i].init(NUM_ISSUE_WARPS);
            }
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; ++i) {
                smem.init_full_bar[i].init(NUM_ISSUE_WARPS);
            }
            init_extra_bar_func();
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();
    }

    static __device__ __forceinline__
    void init_shared_memory(SharedMemoryPlanBase &smem, uint32_t warp_idx) {
        init_shared_memory(smem, warp_idx, [] {});
    }

    // Issue TMA loads for one round
    // Use IS_INIT to control whether is the init round.
    // For init round: copy the "tail" [num_perm_segs * NUM_ELEMS_PER_SEG, end_vocab_idx), if HAVE_TAIL is true, and also the first NUM_SEGS_PER_ROUND - NUM_TAIL_SEGS segments according to the perm generator
    // For non-init round: copy the next NUM_SEGS_PER_ROUND segs according to the perm generator
    template<bool IS_INIT, bool HAVE_TAIL>
    static __device__ __forceinline__
    void issue_tma_loads_for_round(SharedMemoryPlanBase &smem, const CUtensorMap &tensor_map,
                        uint32_t batch_idx, uint32_t end_vocab_idx,
                        uint32_t num_perm_segs, uint32_t round_idx,
                        uint32_t &next_tma_permuted_segment, uint32_t tma_permuted_segment_stride,
                        uint32_t warp_idx
    ) {
        uint32_t perm_len = max(num_perm_segs, 1u);
        uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;

        static_assert(IS_INIT || !HAVE_TAIL);
        if (warp_idx >= NUM_ISSUE_WARPS) return;

        ValueT *dst_base;
        ku::transac_bar_t *bar_ptr;
        if constexpr (IS_INIT) {
            // Use `smem.incoming_topk_pairs` as a buffer for elements in the "init" rounds
            // We've asserted the size of `incoming_topk_pairs` is large enough
            dst_base = (ValueT*)smem.incoming_topk_pairs + round_idx * NUM_ELEMS_PER_ROUND;
            bar_ptr = &smem.init_full_bar[round_idx];
        } else {
            dst_base = smem.tma_load_buf[round_idx % NUM_TMA_LOAD_BUFS];
            bar_ptr = &smem.tma_load_full_bar[round_idx % NUM_TMA_LOAD_BUFS];
        }
        ku::transac_bar_t &bar = *bar_ptr;

        constexpr uint32_t NUM_BYTES_PER_SEG = NUM_ELEMS_PER_SEG * (uint32_t)sizeof(ValueT);
        auto copy_one_seg = [&](uint32_t dst_seg_idx, uint32_t global_seg_idx) {
            cute::SM90_TMA_LOAD_3D::copy(
                &tensor_map, reinterpret_cast<uint64_t*>(&bar),
                (uint64_t)cute::TMA::CacheHintSm90::EVICT_FIRST,
                dst_base + dst_seg_idx * NUM_ELEMS_PER_SEG,
                0, global_seg_idx * NUM_TMA_ROWS_PER_SEG, batch_idx);
        };

        if (IS_INIT && num_perm_segs == 0) {
            // Do not permute if `num_perm_seg == 0`, just copy in all the elements in-a-row
            uint32_t num_global_segs = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
            uint32_t num_segs_issued = 0;
            CUTE_UNROLL
            for (uint32_t k = 0; k < NUM_SEGS_PER_ISSUE_WARP; k++) {
                uint32_t local_seg_idx = warp_idx + k * NUM_ISSUE_WARPS;
                uint32_t global_seg_idx = round_idx * NUM_SEGS_PER_ROUND + local_seg_idx;
                if (global_seg_idx < num_global_segs) {
                    copy_one_seg(local_seg_idx, global_seg_idx);
                    num_segs_issued += 1;
                }
            }
            bar.arrive_and_expect_tx(num_segs_issued * NUM_BYTES_PER_SEG);
        } else {
            static_assert(NUM_TAIL_SEGS % NUM_ISSUE_WARPS == 0);
            constexpr uint32_t NUM_TAIL_SEGS_PER_ISSUE_WARP = HAVE_TAIL ? NUM_TAIL_SEGS / NUM_ISSUE_WARPS : 0;
            if constexpr (HAVE_TAIL) {
                // Copy tail segs
                // To save memory BW, only copy segs that have at least one valid element. The other segs are handled by `fill_padded_tail_segments`
                uint32_t num_tail_tma_segs = ku::ceil_div(end_vocab_idx - num_perm_elems, (uint32_t)NUM_ELEMS_PER_SEG);
                uint32_t num_real_tail_segs_issued = 0;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_TAIL_SEGS_PER_ISSUE_WARP; i++) {
                    uint32_t cur_seg_idx = warp_idx + i * NUM_ISSUE_WARPS;
                    if (cur_seg_idx < num_tail_tma_segs) {
                        copy_one_seg(cur_seg_idx, (num_perm_segs + cur_seg_idx));
                        num_real_tail_segs_issued += 1;
                    }
                }
                bar.arrive_and_expect_tx((num_real_tail_segs_issued + NUM_SEGS_PER_ISSUE_WARP - NUM_TAIL_SEGS_PER_ISSUE_WARP) * NUM_BYTES_PER_SEG);
            } else {
                bar.arrive_and_expect_tx(NUM_SEGS_PER_ISSUE_WARP * NUM_BYTES_PER_SEG);
            }
            CUTE_UNROLL
            for (uint32_t i = NUM_TAIL_SEGS_PER_ISSUE_WARP; i < NUM_SEGS_PER_ISSUE_WARP; i++) {
                uint32_t cur_global_seg_idx = next_tma_permuted_segment;
                uint32_t local_seg_idx = warp_idx + i * NUM_ISSUE_WARPS;
                copy_one_seg(local_seg_idx, cur_global_seg_idx);
                advance_perm_state(next_tma_permuted_segment, tma_permuted_segment_stride, perm_len);
            }
        }
    }

    // Fill the invalid (padded) tail segments with PLACEHOLDER (-INF).
    // TMA only loads the real tail elements; the unused padded tail region must still be initialized as -INF
    static __device__ __forceinline__ void fill_padded_tail_segments(ValueT *init_buf, uint32_t num_tail_elems) {
        uint32_t tail_covered_by_tma = ku::ceil(num_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG);
        constexpr __int128_t FILLER_128b = ((__int128_t)PLACEHOLDER_B64 << 64) | PLACEHOLDER_B64;
        for (uint32_t i = tail_covered_by_tma + threadIdx.x * NUM_ELEMS_PER_128b; i < NUM_TAIL_ELEMS; i += NUM_THREADS * NUM_ELEMS_PER_128b) {
            ku::st_shared(init_buf + i, FILLER_128b);
        }
    }

    // Stage the final top-k candidates from the survivor buffer into a contiguous smem layout
    // and run the epilogue.
    //
    // When SORT_BY_INDEX is true, each 512-element window is visited as a whole in the permuted
    // order, so the survivors are already sorted inside each window. Only the window groups need
    // to be reordered by their window id
    //
    // Input cannot contain NaN
    template<bool SORT_BY_INDEX>
    static __device__ __forceinline__
    void stage_output_and_epilogue(SharedMemoryPlanBase &smem, typename EpilogueT::BlockRadixSortTempStorageT &sort_temp_storage,
                                   const TopkSelectArgs &args, uint32_t batch_idx, uint32_t end_vocab_idx,
                                   uint32_t survivor_buf_idx, uint32_t warp_idx, uint32_t lane_idx) {

        uint32_t staging_buf_idx = survivor_buf_idx ^ 1;
        bool is_already_sorted_in_index = end_vocab_idx <= NUM_ELEMS_IN_INIT_WINDOW;
        uint32_t* window_scatter_delta = (uint32_t*)smem.incoming_topk_pairs;
        ValueT* smem_value_buf = (ValueT*)smem.surviving_topk_pairs[staging_buf_idx];
        uint32_t* smem_index_buf = (uint32_t*)(smem.surviving_topk_pairs[staging_buf_idx] + MAX_TOPK * (uint32_t)sizeof(ValueT) / 8);

        // Index-sort shuffle: each window's selected candidates are already sorted and contiguous;
        // only the whole window groups are in visit order, so reorder them by their window id.
        if constexpr (SORT_BY_INDEX) if (!is_already_sorted_in_index) {
            uint32_t num_windows = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
            auto get_window_id = [&](uint32_t offset) -> uint32_t {
                return (uint32_t)smem.surviving_topk_pairs[survivor_buf_idx][offset] / NUM_ELEMS_PER_SEG;
            };

            // Reset all window counters.
            for (uint32_t i = threadIdx.x; i < num_windows; i += NUM_THREADS) {
                window_scatter_delta[i] = 0;
            }
            __syncthreads();

            // Count how many selected candidates fall into each window.
            for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS) {
                atomicAdd_block(window_scatter_delta + get_window_id(i), 1u);
            }
            __syncthreads();

            // Convert the per-window counts into exclusive prefix sums (starting offsets).
            uint32_t windows_per_thread = ku::ceil_div(num_windows, (uint32_t)NUM_THREADS);
            uint32_t w_lo = min(threadIdx.x * windows_per_thread, num_windows);
            uint32_t w_hi = min(w_lo + windows_per_thread, num_windows);
            uint32_t my_cnt = 0;
            for (uint32_t i = w_lo; i < w_hi; ++i) {
                // TODO Performance can be optimized by using wider reads (pay attn to alignment issues!)
                my_cnt += window_scatter_delta[i];
            }
            uint32_t warp_total = __reduce_add_sync(0xFFFFFFFF, my_cnt);
            if (lane_idx == 0) {
                smem.warp_cnt[warp_idx] = warp_total;
            }
            __syncthreads();

            static_assert(NUM_WARPS <= 32);
            uint32_t cur_prefix_sum = __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? smem.warp_cnt[lane_idx] : 0u) + warp_level_exclusive_prefix_sum(my_cnt, lane_idx);
            for (uint32_t i = w_lo; i < w_hi; ++i) {
                uint32_t cnt = window_scatter_delta[i];
                window_scatter_delta[i] = cur_prefix_sum;
                cur_prefix_sum += cnt;
            }
            __syncthreads();

            // Subtract window_delta[i] by the index of the first element in this window around all select elements
            // So that later we can use "index + window_delta[i]" to choose the index
            for (uint32_t i = threadIdx.x+1; i < args.topk; i += NUM_THREADS) { // +1 to skip the first element
                uint32_t idx = (uint32_t)smem.surviving_topk_pairs[survivor_buf_idx][i];
                uint32_t window_idx = idx / NUM_ELEMS_PER_SEG;
                uint32_t prev_elem_window_id = get_window_id(i - 1u);
                if (window_idx != prev_elem_window_id) {
                    // i is the first element in its window
                    window_scatter_delta[window_idx] -= i;
                }
            }

            __syncthreads();
        }

        CUTE_UNROLL
        for (uint32_t i = threadIdx.x; i < MAX_TOPK; i += NUM_THREADS) {
            uint64_t pair = smem.surviving_topk_pairs[survivor_buf_idx][i];
            uint32_t idx = (uint32_t)pair;
            ValueT val;
            if constexpr (sizeof(ValueT) == 2) {
                val = __ushort_as_bfloat16((uint16_t)(pair >> 32));
            } else {
                val = __uint_as_float((uint32_t)(pair >> 32));
            }
            if (SORT_BY_INDEX && !is_already_sorted_in_index) {
                if (i < args.topk) {
                    uint32_t dst_pos = (i + window_scatter_delta[idx / NUM_ELEMS_PER_SEG]) & (MAX_TOPK - 1);
                    smem_value_buf[dst_pos] = val;
                    smem_index_buf[dst_pos] = idx;
                    continue;
                }
            }
            smem_value_buf[i] = val;
            smem_index_buf[i] = idx;
        }

        __syncthreads();

        EpilogueT::template topk_select_epilogue<false>(
            smem_value_buf,
            smem_index_buf,
            args,
            batch_idx, end_vocab_idx, warp_idx,
            sort_temp_storage
        );
    }

    // Take the appropriate action when we see a NaN: `trap()`, or [SGLang] write the row as if every selected
    // index were 0, so downstream gathers stay in range. Called by the whole CTA; out of line, since inlined copies
    // of this cold path only cost instruction fetch on every launch.
    static __device__ __noinline__
    void take_action_when_have_nan(const TopkSelectArgs &args, uint32_t batch_idx) {
        if (args.abort_when_nan_found) {
            if (threadIdx.x == 0) {
                printf("[topk_select] NaN detected. Calling `trap;` which will result in \"unspecified launch failure\"\n");
            }
            __syncthreads();
            ku::trap();
        } else {
            OutIdxT* result_indices = (OutIdxT*)args.output_index + (uint64_t)batch_idx * args.stride_output_index_batch;
            OutIdxT index0;
            if constexpr (Config::page_transform) {
                const int* page_table_row = args.page_table + (uint64_t)batch_idx * args.stride_page_table_batch;
                index0 = (OutIdxT)EpilogueT::page_transform_index(0u, true, 0, page_table_row, args.page_bits);
            } else {
                index0 = (OutIdxT)(args.output_idx_offset != nullptr ? __ldg(args.output_idx_offset + batch_idx) : 0);
            }
            for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS) result_indices[i] = index0;
            if constexpr (Config::return_value) {
                const ValueT value0 = ((const ValueT*)args.input)[(uint64_t)batch_idx * args.stride_input_batch];
                ValueT* result_values = (ValueT*)args.output_value + (uint64_t)batch_idx * args.stride_output_value_batch;
                for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS) result_values[i] = value0;
            }
        }
    }
};

template<typename Config>
class TopkSelectKernelBF16Base : public TopkSelectKernelBase<Config> {
    using Base = TopkSelectKernelBase<Config>;
public:
    using ValueT = typename Base::ValueT;
    using OutIdxT = typename Base::OutIdxT;
    using TmaParams = typename Base::TmaParams;
    using SharedMemoryPlanBase = typename Base::SharedMemoryPlanBase;
    using EpilogueT = typename Base::EpilogueT;
    using Base::NUM_THREADS;
    using Base::NUM_WARPS;
    using Base::MAX_TOPK;
    using Base::NUM_ELEMS_PER_128b;
    using Base::NUM_UINT32_PER_128b;
    using Base::NUM_ELEMS_PER_ROUND;
    using Base::NUM_ELEMS_PER_THREAD_PER_ROUND;
    using Base::ELEMS_PER_THREAD_PER_ROUND_MASK;
    using Base::NUM_128b_PER_THREAD_PER_ROUND;
    using Base::NUM_ELEMS_PER_SEG;
    using Base::NUM_TAIL_ELEMS;
    using Base::NUM_TAIL_SEGS;
    using Base::NUM_ELEMS_IN_INIT_WINDOW;
    using Base::NUM_SEGS_IN_INIT_WINDOW;
    using Base::NUM_INIT_ROUNDS_MAX;
    using Base::NUM_UINT32_IN_INIT_WINDOW_PER_THREAD;
    using Base::NUM_128b_INIT_PER_THREAD;
    using Base::NUM_SEGS_PER_ROUND;
    using Base::NUM_SEGS_PER_ISSUE_WARP;
    using Base::NUM_ISSUE_WARPS;
    using Base::NUM_TMA_LOAD_BUFS;
    using Base::TMA_PREFETCH_DEPTH;
    using Base::RECONSTRUCT_THRESHOLD;
    using Base::PERM_ADD_BASE;
    using Base::PERM_MUL_PRIME;
    using Base::NUM_EXTRA_SLOTS;
    using Base::PLACEHOLDER_PAIR;
    using Base::NUM_RECONSTRUCT_BUCKETS;
    using Base::NUM_RECONSTRUCT_UNITS_MAX;

    static_assert(std::is_same_v<ValueT, nv_bfloat16>);
    static_assert(!Config::sorted_value);
    static constexpr uint32_t NUM_UINT32_RECONSTRUCT_PER_THREAD =
        ((NUM_RECONSTRUCT_UNITS_MAX + NUM_THREADS - 1) / NUM_THREADS) | 1u; // padding
    static_assert(2 * NUM_UINT32_RECONSTRUCT_PER_THREAD <= 256);
    static constexpr uint16_t PLACEHOLDER = 0xff80; // -INF
    static constexpr uint32_t NEG_INF_X2_BITS = 0xFF80FF80u;

    // Histogram pass 1 (MSB radix): for each distorted bf16x2 value, retrieve the upper 8 bits and performs atomicAdd on the bucket counter
    // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    template<uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    void histogram_radix_msb(uint32_t *bucket_counter, const nv_bfloat162 (&packed_values)[N], uint32_t num_packed_values) {
        uint32_t bucket_base = cute::cast_smem_ptr_to_uint(bucket_counter);
        CUTE_UNROLL
        for (uint32_t i = 0; i < N; i++) {
            if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values) break;
            uint32_t raw = bf16x2_to_u32(packed_values[i]);
            uint32_t distorted;
            distort_x2<uint16_t>((uint16_t*)&distorted, (const uint16_t*)&raw);
            // bfe + mad form each bucket address in two instructions, then two shared-memory adds.
            asm volatile (
                "{\n"
                ".reg .b32 i0, i1, a0, a1;\n"
                "bfe.u32 i0, %0, 8, 8;\n"
                "bfe.u32 i1, %0, 24, 8;\n"
                "mad.lo.u32 a0, i0, 4, %1;\n"
                "mad.lo.u32 a1, i1, 4, %1;\n"
                "red.shared.add.u32 [a0], 1;\n"
                "red.shared.add.u32 [a1], 1;\n"
                "}\n"
                :
                : "r"(distorted), "r"(bucket_base)
                : "memory"
            );
        }
    }

    // Histogram pass 2 (LSB radix refinement): among values whose distorted MSB equals pivot_hi8, retrieve the lower 8 bits and performs atomicAdd on the bucket counter
    // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    // `pivot_hi8` is the highest 8 bit of the DISTORTED pivot
    template<bool USE_CLUSTER_ADDRESSING, uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    void histogram_radix_lsb_for_pivot_msb(uint32_t *bucket_counter, uint32_t pivot_hi8, const nv_bfloat162 (&packed_values)[N], uint32_t num_packed_values) {
        uint32_t bucket1_base = cute::cast_smem_ptr_to_uint(bucket_counter);
        // Every elements in the pivot bucket must have the same sign, so we can use this information to optimize instead of
        bool bucket_negative = pivot_hi8 < 0x80;
        uint32_t orig_pivot_hi8 = bucket_negative ? 0xFF - pivot_hi8 : pivot_hi8 - 0x80;    // Un-distort
        uint32_t raw_hi8_x2 = (orig_pivot_hi8 << 16) | orig_pivot_hi8;
        uint32_t low_byte_xor = bucket_negative ? 0xFFFFFFFFu : 0u;
        // Elements outside the bucket are counted in this row's sink slot (index NUM_RECONSTRUCT_BUCKETS) and never read.
        uint32_t scratch_bucket_idx_x2 = 0x01000100;    // 2x 256
        CUTE_UNROLL
        for (uint32_t i = 0; i < N; i++) {
            if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values) break;
            asm volatile (
                "{\n"
                ".reg .b32 hi, lo, sel, t, t0, t1, a0, a1;\n"
                "prmt.b32 hi, %0, 0, 0x5341;\n"             // bytes {1, zero, 3, zero}: both raw high bytes as 0x00hh halves
                "set.eq.s32.bf16x2 sel, hi, %1;\n"          // 0xFFFF per half whose raw high byte equals the pivot's
                "lop3.b32 lo, %0, %4, 0x00ff00ff, 0x28;\n"  // lo = (%0 ^ %4) & 0x00ff00ff
                "lop3.b32 t, sel, lo, %3, 0xca;\n"          // t = sel ? lo : %3, per-bit
                "and.b32 t0, t, 0xFFFF;\n"
                "shr.u32 t1, t, 16;\n"
                "mad.lo.u32 a0, t0, 4, %2;\n"               // base + bucket * 4
                "mad.lo.u32 a1, t1, 4, %2;\n"
                "red.shared.add.u32 [a0], 1;\n"
                "red.shared.add.u32 [a1], 1;\n"
                "}\n"
                :
                : "r"(bf16x2_to_u32(packed_values[i])), "r"(raw_hi8_x2), "r"(bucket1_base), "r"(scratch_bucket_idx_x2), "r"(low_byte_xor)
                : "memory"
            );
        }
    }

    // Count the number of elements that are 1) > pivot 2) = pivot 3) NaN in the given array
    struct CensusCounts { uint32_t cnt_gt; uint32_t cnt_eq; uint32_t cnt_nan; };
    // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    template<uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    CensusCounts get_census_counts(const nv_bfloat162 (&values)[N], uint32_t num_packed_values, uint32_t pivot_value_x2_bits) {
        static_assert(2 * N <= 256);    // Since we're going to use bf16 for accumulation
        nv_bfloat162 gt_accum = {0.0f, 0.0f};
        nv_bfloat162 eq_accum = {0.0f, 0.0f};
        nv_bfloat162 nan_accum = {0.0f, 0.0f};
        CUTE_UNROLL
        for (uint32_t i = 0; i < N; i++) {
            if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values) break;
            uint32_t raw = bf16x2_to_u32(values[i]);
            asm volatile (
                "{\n"
                ".reg .b32 gt_result, eq_result;\n"
                "set.gt.bf16x2.bf16x2 gt_result, %3, %4;\n"
                "add.rn.bf16x2 %0, gt_result, %0;\n"
                "set.eq.bf16x2.bf16x2 eq_result, %3, %4;\n"
                "add.rn.bf16x2 %1, eq_result, %1;\n"
                "min.NaN.bf16x2 %2, %2, %3;\n"  // We use `min.NaN` for NaN detection
                "}\n"
                : "+r"(*(uint32_t*)&gt_accum), "+r"(*(uint32_t*)&eq_accum), "+r"(*(uint32_t*)&nan_accum)
                : "r"(raw), "r"(pivot_value_x2_bits)
            );
        }
        uint32_t cnt_gt = (uint32_t)(float)(gt_accum.x + gt_accum.y);
        uint32_t cnt_eq = (uint32_t)(float)(eq_accum.x + eq_accum.y);
        uint32_t nan_flag;
        asm ("set.nan.bf16x2.bf16x2 %0, %1, %1;" : "=r"(nan_flag) : "r"(*(uint32_t*)&nan_accum));
        uint32_t cnt_nan = nan_flag != 0 ? 1u : 0u;
        return {cnt_gt, cnt_eq, cnt_nan};
    }

    // Decide whether or not to accept an element in the old survivor buffer
    // If taken, update relevant status (`dst_ptr` and `eq_quota`)
    template<bool USE_CLUSTER_ADDRESSING>
    static __device__ __forceinline__
    void copy_selected_pairs_to_survivor(
        uint32_t &dst_ptr,  // The pointer to the new survivor buffer. May increase. Marked as uint32 since it's obtained from cast_smem_ptr_to_uint
        uint32_t &eq_quota, // The EQ quota of this thread. May decrease
        nv_bfloat162 packed_values,
        uint32_t pivot_value_x2_bits,
        uint32_t src_ptr    // The pointer to the location of src. `src` contains 8B pairs in {index, value}
    ) {
        uint32_t values_raw = bf16x2_to_u32(packed_values);
        #define MOVE_SELECTED_PAIR2(ADD)                \
            asm volatile (                                          \
                "{\n"                                               \
                ".reg .pred g0, g1, e0, e1, q, t0, t1, s0, s1;\n"   \
                ".reg .b32 pair0_lo, pair0_hi, pair1_lo, pair1_hi;\n" \
                "ld.shared.v4.b32 {pair0_lo, pair0_hi, pair1_lo, pair1_hi}, [%4];\n" \
                "setp.gt.bf16x2 g0|g1, %2, %3;\n"                   \
                "setp.eq.bf16x2 e0|e1, %2, %3;\n"                   \
                /* element 0 (low half) */                          \
                "setp.ne.and.u32 t0, %1, 0, e0;\n"                                                  \
                "@t0 sub.u32 %1, %1, 1;\n"                          \
                "or.pred s0, g0, t0;\n"                             \
                "@s0 st.shared.v2.b32 [%0], {pair0_lo, pair0_hi};\n" \
                "@s0 " ADD "\n"                                     \
                /* element 1 (high half; quota is re-tested after element 0's decrement) */ \
                "setp.ne.and.u32 t1, %1, 0, e1;\n"                                                  \
                "@t1 sub.u32 %1, %1, 1;\n"                          \
                "or.pred s1, g1, t1;\n"                             \
                "@s1 st.shared.v2.b32 [%0], {pair1_lo, pair1_hi};\n" \
                "@s1 " ADD "\n"                                     \
                "}\n"                                               \
                : "+r"(dst_ptr), "+r"(eq_quota)                     \
                : "r"(values_raw), "r"(pivot_value_x2_bits), "r"(src_ptr) \
                : "memory"                                          \
            )
        if constexpr (USE_CLUSTER_ADDRESSING) {
            MOVE_SELECTED_PAIR2("add.u32 %0, %0, 8;");
        } else {
            MOVE_SELECTED_PAIR2("add.f32 %0, %0, 0f00000008;");
        }
        #undef MOVE_SELECTED_PAIR2
    }

    // The src-in-register version of `copy_selected_pairs_to_survivor`
    template<bool USE_CLUSTER_ADDRESSING>
    static __device__ __forceinline__
    void append_selected_pairs_from_registers(
        uint32_t &dst_ptr,
        uint32_t &eq_quota,
        nv_bfloat162 packed_values,
        uint32_t pivot_value_x2_bits,
        uint32_t index0,
        uint32_t index1
    ) {
        uint32_t values_raw = bf16x2_to_u32(packed_values);
        uint32_t val_word1 = values_raw >> 16;   // element 1 low, clean high
        #define APPEND_SELECTED_PAIR2(ADD)              \
            asm volatile (                                          \
                "{\n"                                               \
                ".reg .pred g0, g1, e0, e1, q, t0, t1, s0, s1;\n"   \
                "setp.gt.bf16x2 g0|g1, %2, %3;\n"                   \
                "setp.eq.bf16x2 e0|e1, %2, %3;\n"                   \
                /* element 0 (low half) */                          \
                "setp.ne.and.u32 t0, %1, 0, e0;\n"                                                  \
                "@t0 sub.u32 %1, %1, 1;\n"                          \
                "or.pred s0, g0, t0;\n"                             \
                "@s0 st.shared.v2.u32 [%0], {%4, %2};\n"            \
                "@s0 " ADD "\n"                                     \
                /* element 1 (high half; quota is re-tested after element 0's decrement) */ \
                "setp.ne.and.u32 t1, %1, 0, e1;\n"                                                  \
                "@t1 sub.u32 %1, %1, 1;\n"                          \
                "or.pred s1, g1, t1;\n"                             \
                "@s1 st.shared.v2.u32 [%0], {%5, %6};\n"            \
                "@s1 " ADD "\n"                                     \
                "}\n"                                               \
                : "+r"(dst_ptr), "+r"(eq_quota)                     \
                : "r"(values_raw), "r"(pivot_value_x2_bits),               \
                "r"(index0), "r"(index1), "r"(val_word1)          \
                : "memory"                                          \
            )
        if constexpr (USE_CLUSTER_ADDRESSING) {
            APPEND_SELECTED_PAIR2("add.u32 %0, %0, 8;");
        } else {
            APPEND_SELECTED_PAIR2("add.f32 %0, %0, 0f00000008;");
        }
        #undef APPEND_SELECTED_PAIR2
    }

    // Get the pivot via a two-stage histogram and get PivotAndQuota
    // The MSB histogram should be ready and all threads should be able to see it (e.g. via a `__syncthreads()`) before we call the function
    // We don't absorb MSB histogram's logic inside since sometimes we'd like to overlap TMA load and MSB histogram's modification
    struct PivotAndQuota {
        uint32_t pivot_value_x2_bits;
        uint32_t start_pos_in_collector;
        uint32_t eq_quota;
        uint32_t cnt_nan;
    };
    // `num_my` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    template<bool USE_CLUSTER_ADDRESSING, uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    PivotAndQuota compute_pivot_and_quota(uint32_t topk, const nv_bfloat162 (&values)[N],
                                       uint32_t num_values, uint32_t warp_idx, uint32_t lane_idx, SharedMemoryPlanBase &smem) {
        if (warp_idx == 0) {
            Base::template find_pivot_in_histogram<false>(smem, smem.reconstruct_bucket_counter[0], topk, lane_idx);
        }
        __syncthreads();
        uint32_t pivot_hi8 = smem.reconstruct_pivot_bucket;
        uint32_t num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;

        histogram_radix_lsb_for_pivot_msb<USE_CLUSTER_ADDRESSING, N, NUM_PACKED_VALUES_ALIGNMENT>(smem.reconstruct_bucket_counter[1], pivot_hi8, values, num_values);
        __syncthreads();

        if (warp_idx == 0) {
            Base::template find_pivot_in_histogram<false>(smem, smem.reconstruct_bucket_counter[1], num_elem_should_select_in_pivot_bucket, lane_idx);
        }
        __syncthreads();
        uint32_t pivot_lo8 = smem.reconstruct_pivot_bucket;
        uint32_t pivot_distorted = (pivot_hi8 << 8) | pivot_lo8;
        uint16_t pivot_value = un_distort((uint16_t)pivot_distorted);
        uint32_t pivot_value_x2_bits = ((uint32_t)pivot_value << 16) | pivot_value;

        auto census = get_census_counts<N, NUM_PACKED_VALUES_ALIGNMENT>(values, num_values, pivot_value_x2_bits);

        static_assert(NUM_WARPS <= NUM_RECONSTRUCT_BUCKETS);
        auto eqgt = Base::compute_equal_quota_and_prefix(census.cnt_gt, census.cnt_eq, topk, warp_idx, lane_idx, smem.reconstruct_bucket_counter[0]);
        return {pivot_value_x2_bits, eqgt.start_pos_in_collector, eqgt.eq_quota, census.cnt_nan};
    }

    // Main part of the normal kernel and phase A of the cluster kernel.
    // The input is divided into:
    //  - a "perm" prefix [0, num_perm_segs * NUM_ELEMS_PER_SEG), visited in pseudo-random order;
    //  - a "tail" suffix [num_perm_segs * NUM_ELEMS_PER_SEG, end_vocab_idx), visited sequentially.
    // The init phase loads the tail plus the first perm segments into one init window, selects
    // the initial top-k from it, then the main loop continues over the remaining perm segments.
    //
    // Returns the number of real selected pairs in the final survivor buffer; this is normally
    // `topk`, but can be smaller in a cluster CTA whose local range has fewer real elements.
    template<bool USE_CLUSTER_ADDRESSING, typename IsWarpActiveF>
    static __device__ __forceinline__
    uint32_t scan_segs(const TmaParams &tma_params, SharedMemoryPlanBase &smem,
                       uint32_t batch_idx, uint32_t end_vocab_idx, uint32_t topk,
                       uint32_t warp_idx, uint32_t lane_idx,
                       uint32_t num_perm_segs,                  // The number of segments to be permuted, globally
                       uint32_t local_start_seg_idx,            // The index of the first "local" segment ("local" means "belong to this CTA"). 0 if not the cluster-based implementation
                       uint32_t num_local_perm_segs,            // The number of permuted segments to be processed, locally
                       uint32_t num_local_tail_elems_padded,    // num_local_tail_elems, padded to NUM_TAIL_ELEMS if num_perm_segs > 0, otherwise padded to NUM_ELEMS_PER_SEG
                       uint32_t num_local_tail_elems,           // The number of tail elems locally
                       uint32_t &survivor_buf_idx,
                       bool &have_nan,
                       IsWarpActiveF &&is_warp_active_f) {
        uint32_t num_local_elems_padded = num_local_tail_elems_padded + num_local_perm_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_local_rounds = ku::ceil_div(num_local_elems_padded, (uint32_t)NUM_ELEMS_PER_ROUND);
        uint32_t num_init_rounds = min(num_local_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);
        uint32_t num_local_tail_segs_padded = num_local_tail_elems_padded / NUM_ELEMS_PER_SEG;
        uint32_t num_main_rounds = num_local_rounds - num_init_rounds;

        // Permutation generation arguments
        uint32_t perm_len = max(num_perm_segs, 1u);
        uint32_t perm_mul = PERM_MUL_PRIME % perm_len;

        // Permutation state
        // Only valid for the lane elected by `elect_one_sync`
        uint32_t tma_permuted_segment_stride = NUM_ISSUE_WARPS * perm_mul % perm_len;
        uint32_t linear_tma_segment_start = local_start_seg_idx + warp_idx;
        uint32_t next_tma_permuted_segment = Base::get_permuted_seg_idx(linear_tma_segment_start, perm_len, perm_mul);

        auto issue_tma_copy = [&]<bool IS_INIT, bool HAVE_TAIL>(uint32_t round_idx) {
            Base::template issue_tma_loads_for_round<IS_INIT, HAVE_TAIL>(
                smem, tma_params.tensor_map,
                batch_idx, end_vocab_idx,
                num_perm_segs, round_idx,
                next_tma_permuted_segment, tma_permuted_segment_stride,
                warp_idx
            );
        };
        auto issue_tma_copy_for_main_rounds = [&](uint32_t round_idx) {
            issue_tma_copy.template operator()<false, false>(round_idx);
        };

        uint32_t threshold_x2_bits = NEG_INF_X2_BITS;
        uint32_t num_incomers = 0;
        uint32_t num_survivors = 0;

        // Init phase
        if (num_local_elems_padded != 0) {
            // Issue copy: Copy the tail and the first few segments, result in init_buf[0: NUM_ELEMS_IN_INIT_WINDOW] (or init_buf[0: ])
            ValueT *init_buf = (ValueT*)smem.incoming_topk_pairs;
            if (cute::elect_one_sync()) {
                if (num_local_tail_elems_padded != 0) {
                    issue_tma_copy.template operator()<true, true>(0);
                } else {
                    issue_tma_copy.template operator()<true, false>(0);
                }

                CUTE_UNROLL
                for (uint32_t i = 1; i < NUM_INIT_ROUNDS_MAX; i++) {
                    if (i < num_init_rounds) {
                        issue_tma_copy.template operator()<true, false>(i);
                    }
                }

                // Pre-issue TMA copies for the main part
                CUTE_UNROLL
                for (uint32_t i = 0; i < TMA_PREFETCH_DEPTH; i++) {
                    if (i < num_main_rounds) {
                        issue_tma_copy_for_main_rounds(i);
                    }
                }
            }

            if (num_perm_segs != 0 && num_local_tail_elems_padded != 0) {
                Base::fill_padded_tail_segments(init_buf, num_local_tail_elems);
                __syncthreads();
            }

            // Decide how many 128-bit elements each thread processes.
            uint32_t num_elems_in_init_window_padded = num_local_tail_elems_padded + min(num_local_perm_segs, (uint32_t)NUM_SEGS_IN_INIT_WINDOW - num_local_tail_segs_padded) * NUM_ELEMS_PER_SEG;  // Init window size, padded to NUM_ELEMS_PER_SEG
            uint32_t num_128b_in_init_window_padded = num_elems_in_init_window_padded / NUM_ELEMS_PER_128b;
            uint32_t cnt_floor = num_128b_in_init_window_padded / NUM_THREADS;
            uint32_t cnt_rem = num_128b_in_init_window_padded % NUM_THREADS;
            uint32_t my_elem_start_idx = (threadIdx.x * cnt_floor + min(threadIdx.x, cnt_rem)) * NUM_ELEMS_PER_128b;
            uint32_t num_my_elems = NUM_ELEMS_PER_128b * cnt_floor + (threadIdx.x < cnt_rem ? NUM_ELEMS_PER_128b : 0u);
            uint32_t tail_padding_elems = num_local_tail_elems_padded - num_local_tail_elems;
            uint32_t num_real_init_elems = num_elems_in_init_window_padded - tail_padding_elems;
            num_survivors = min(topk, num_real_init_elems);

            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i == num_init_rounds) break;
                smem.init_full_bar[i].wait(0);

                // Fill the incomplete segment with PLACEHOLDER (-INF)
                if (num_local_tail_elems % NUM_ELEMS_PER_SEG != 0 && i == num_local_tail_elems / NUM_ELEMS_PER_ROUND) {
                    uint32_t box_end = ku::ceil_div(num_local_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG) * NUM_ELEMS_PER_SEG;
                    for (uint32_t e = num_local_tail_elems + threadIdx.x; e < box_end; e += NUM_THREADS) {
                        init_buf[Base::sw_elem(e)] = __ushort_as_bfloat16(PLACEHOLDER);
                    }
                    __syncthreads();
                }
                nv_bfloat162 my_values[4 * NUM_128b_PER_THREAD_PER_ROUND];
                uint32_t num_my_values = 0;
                CUTE_UNROLL
                for (uint32_t k = 0; k < NUM_128b_PER_THREAD_PER_ROUND; ++k) {
                    uint32_t pos = threadIdx.x + (i * NUM_128b_PER_THREAD_PER_ROUND + k) * NUM_THREADS;
                    if (pos < num_128b_in_init_window_padded) {
                        ld_shared<4>(my_values + k * 4, reinterpret_cast<const nv_bfloat162*>(init_buf + Base::sw_b128(pos) * NUM_ELEMS_PER_128b));
                        num_my_values = 4 * (k + 1);
                    }
                }
                histogram_radix_msb<4 * NUM_128b_PER_THREAD_PER_ROUND, 4>(smem.reconstruct_bucket_counter[0], my_values, num_my_values);
            }

            nv_bfloat162 init_values[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t j = 0; j < NUM_128b_INIT_PER_THREAD; ++j) {
                if (j * NUM_ELEMS_PER_128b == num_my_elems) break;
                uint32_t u0 = my_elem_start_idx / NUM_ELEMS_PER_128b + j;
                ld_shared<4>(init_values + j * NUM_UINT32_PER_128b, reinterpret_cast<const nv_bfloat162*>(init_buf + Base::sw_b128(u0) * NUM_ELEMS_PER_128b));
            }

            __syncthreads();    // publish the round-1 histogram

            static_assert(NUM_ELEMS_IN_INIT_WINDOW <= 0xFFFF);
            // The CTA's init-window slice may contain fewer real elements than K (e.g. a cluster
            // rank whose visit range mostly overlaps the padded tail). In that case the pivot
            // selection must degrade to "select all real elements": cap the K used in the pivot /
            // quota computation at the real count
            uint32_t effective_topk = min(topk, num_real_init_elems);
            auto [pivot_value_x2_bits, start_pos_in_collector, eq_quota, cnt_nan] =
                compute_pivot_and_quota<USE_CLUSTER_ADDRESSING, NUM_UINT32_IN_INIT_WINDOW_PER_THREAD, 4>(effective_topk, init_values, num_my_elems / 2, warp_idx, lane_idx, smem);
            // The init census walks the thread's whole slice of the window, i.e. every element of the init
            // window exactly once, so NaNs inside the window are caught here.
            have_nan |= cnt_nan != 0;

            {
                uint32_t smem_base = cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[0]);
                uint32_t out_ptr = smem_base + start_pos_in_collector * (uint32_t)sizeof(uint64_t);
                uint32_t s0 = local_start_seg_idx + (max(my_elem_start_idx, num_local_tail_elems_padded) - num_local_tail_elems_padded) / NUM_ELEMS_PER_SEG;
                uint32_t perm_state = Base::get_permuted_seg_idx(s0, perm_len, perm_mul);
                float unit_base_f = 0.0f;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i++) {
                    // num_my_elems is a multiple of NUM_ELEMS_PER_128b (= 8), so i*2 == num_my_elems
                    // only at i % 4 == 0; the extra condition lets the compiler unroll by 4.
                    if (i % 4 == 0 && i*2 == num_my_elems) break;
                    if (i % 4 == 0) {
                        uint32_t g_u = my_elem_start_idx + i / 4 * NUM_ELEMS_PER_128b;
                        if (g_u < num_local_tail_elems_padded) {
                            unit_base_f = __uint_as_float(num_perm_segs * NUM_ELEMS_PER_SEG + g_u);
                        } else {
                            unit_base_f = __uint_as_float(perm_state * NUM_ELEMS_PER_SEG + g_u % NUM_ELEMS_PER_SEG);
                            if (g_u % NUM_ELEMS_PER_SEG == NUM_ELEMS_PER_SEG - NUM_ELEMS_PER_128b) {
                                Base::advance_perm_state(perm_state, perm_mul, perm_len);
                            }
                        }
                    }
                    // fp32-subnormal FP-pipe int-add: index = base + element offset
                    uint32_t index0 = __float_as_uint(unit_base_f + __uint_as_float(i % 4 * 2));
                    uint32_t index1 = __float_as_uint(unit_base_f + __uint_as_float(i % 4 * 2 + 1));
                    nv_bfloat162 cur_value = init_values[i];      // two bf16 payloads packed as bf16x2
                    append_selected_pairs_from_registers<USE_CLUSTER_ADDRESSING>(out_ptr, eq_quota, cur_value, pivot_value_x2_bits, index0, index1);
                }
            }

            threshold_x2_bits = pivot_value_x2_bits;
        }
        __syncthreads();


        uint32_t linear_segment_start =
            local_start_seg_idx
            + num_init_rounds * NUM_SEGS_PER_ROUND
            - num_local_tail_segs_padded
            + warp_idx;
        uint32_t current_permuted_segment = Base::get_permuted_seg_idx(linear_segment_start, perm_len, perm_mul);
        uint32_t permuted_segment_stride_per_round = NUM_SEGS_PER_ROUND % perm_len * perm_mul % perm_len;

        // Re-select the top-k elements from the smem.surviving_topk_pairs[survivor_buf_idx][:num_survivors] + smem.incoming_topk_pairs[:num_incomers],
        // Write the selected pairs into the other survivor buffer, and return the new pivot
        auto reconstruct = [&](uint32_t cur_extra_len) -> uint32_t {
            uint32_t tid = threadIdx.x;
            Base::clear_reconstruct_histograms(smem, tid);

            uint32_t padded_extra_len = ku::ceil(cur_extra_len, 2u);    // Since we process elements in pairs (uint32_t)
            if (warp_idx == 0 && (cur_extra_len & 1u)) {
                smem.incoming_topk_pairs[cur_extra_len] = PLACEHOLDER_PAIR;
            }
            __syncthreads();

            // units [0, MAX_TOPK/2): the current survivor buffer
            // The rest: incomers
            uint32_t num_uint32 = (MAX_TOPK + padded_extra_len) / 2;
            uint32_t num_uint32_per_thread = ku::ceil_div(num_uint32, (uint32_t)NUM_THREADS) | 1u;   // force odd => bank-conflict-free smem gathers
            uint32_t unit_base = tid * num_uint32_per_thread;
            uint32_t num_my_units = unit_base < num_uint32 ? min(num_uint32_per_thread, num_uint32 - unit_base) : 0u;

            auto unit_to_pair_addr = [&](uint32_t offset) -> uint32_t {
                return offset < MAX_TOPK / 2
                     ? cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx] + 2 * offset)
                     : cute::cast_smem_ptr_to_uint(smem.incoming_topk_pairs + (2 * offset - MAX_TOPK));
            };

            nv_bfloat162 values[NUM_UINT32_RECONSTRUCT_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t m = 0; m < NUM_UINT32_RECONSTRUCT_PER_THREAD; m++) {
                if (m == num_my_units) break;
                uint32_t pair2[4];
                ld_shared<4>(pair2, (const uint32_t*)unit_to_pair_addr(unit_base + m));
                // One b128 unit = 2 pairs {index0, value0, index1, value1}; pick the low halves of the
                // two value words (the bf16 payloads) and pack them into one bf16x2 word.
                values[m] = u32_to_bf16x2(__byte_perm(pair2[1], pair2[3], 0x5410));
            }

            histogram_radix_msb(smem.reconstruct_bucket_counter[0], values, num_my_units);
            __syncthreads();

            auto [pivot_value_x2_bits, out_prefix, eq_quota, cnt_nan] =
                compute_pivot_and_quota<USE_CLUSTER_ADDRESSING>(topk, values, num_my_units, warp_idx, lane_idx, smem);
            // Every NaN that the main loop collected is part of the buffer this census just walked (the hit
            // test is `.gtu`, so NaN always becomes an incomer).
            have_nan |= cnt_nan != 0;

            { // write back
                uint32_t out_ptr = cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx ^ 1]) + out_prefix * (uint32_t)sizeof(uint64_t);
                CUTE_UNROLL
                for (uint32_t m = 0; m < NUM_UINT32_RECONSTRUCT_PER_THREAD; m++) {
                    if (m == num_my_units) break;
                    copy_selected_pairs_to_survivor<USE_CLUSTER_ADDRESSING>(out_ptr, eq_quota, values[m], pivot_value_x2_bits, unit_to_pair_addr(unit_base + m));
                }
            }

            survivor_buf_idx ^= 1;
            return pivot_value_x2_bits;
        };

        uint32_t logical_elem_offset = threadIdx.x * NUM_ELEMS_PER_THREAD_PER_ROUND;
        uint32_t offset_in_segment = logical_elem_offset % NUM_ELEMS_PER_SEG;

        uint32_t swizzle_mask = Base::sw_msk(logical_elem_offset);
        uint32_t chunk_swizzle_mask = swizzle_mask & ELEMS_PER_THREAD_PER_ROUND_MASK;
        uint32_t smem_read_offset = logical_elem_offset ^ (swizzle_mask & ~ELEMS_PER_THREAD_PER_ROUND_MASK);

        // Track the TMA buffer index and its phase incrementally instead of dividing every round.
        // On long rows the init window is a tiny fraction of the row, so its pivot stays far above the
        // row's true one and the append rate (hence the store path) stays high for the whole scan:
        // reconstruct early to tighten the pivot. Short rows would pay more for the reconstruct than the
        // extra appends cost, so they keep the plain threshold.
        uint32_t reconstruct_trigger = num_main_rounds > 16 ? RECONSTRUCT_THRESHOLD / 4 : RECONSTRUCT_THRESHOLD;

        uint32_t tma_buf_idx = 0;
        uint32_t tma_buf_phase = 0;

        for (uint32_t main_round_idx = 0; main_round_idx < num_main_rounds; ++main_round_idx) {
            if (main_round_idx + TMA_PREFETCH_DEPTH < num_main_rounds && cute::elect_one_sync()) {
                issue_tma_copy_for_main_rounds(main_round_idx + TMA_PREFETCH_DEPTH);
            }

            bool is_warp_active = is_warp_active_f(main_round_idx);

            const ValueT *buf = smem.tma_load_buf[tma_buf_idx];
            // One bit per element of this thread's slice, in element order.
            // Hits are rare, so appending through the set bits of this mask is faster than testing + storing every element again.
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND <= 32);        // hit_mask is a uint32_t
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND % 4 == 0);     // the loop below packs 4 elements per nibble
            uint32_t hit_mask = 0;
            if (is_warp_active) {
                smem.tma_load_full_bar[tma_buf_idx].wait(tma_buf_phase);
                ValueT values[NUM_ELEMS_PER_THREAD_PER_ROUND];
                Base::template load_swizzled_slice<NUM_128b_PER_THREAD_PER_ROUND>(values, buf, smem_read_offset, chunk_swizzle_mask);
                CUTE_UNROLL
                for (int32_t j = NUM_ELEMS_PER_THREAD_PER_ROUND - 4; j >= 0; j -= 4) {
                    asm volatile (
                        "{\n"
                        ".reg .b32 r0, r1, ind, nib;\n"
                        "set.gtu.s32.bf16x2 r0, %1, %3;\n"   // .gtu: NaN always counts as a hit, so every NaN reaches the incoming buffer and the census below reports it.
                        "set.gtu.s32.bf16x2 r1, %2, %3;\n"   // `set` with .s32 fills the corresponding 16bit to 0xFFFF (-1) when the condition holds. We use `prmt` to extract one bit from each
                        "prmt.b32 ind, r0, r1, 0x7531;\n"
                        "dp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\n"   // ind's bytes are 0x00/0xFF, so with signed byte weights each hit contributes 1/2/4/8
                        "mad.lo.u32 %0, %0, 16, nib;\n"
                        "}\n"
                        : "+r"(hit_mask)
                        : "r"(*(const uint32_t*)(values + j)),
                          "r"(*(const uint32_t*)(values + j + 2)),
                          "r"(threshold_x2_bits)
                    );
                }
            }
            uint32_t num_new_incomers = __popc(hit_mask);

            uint32_t warp_total_hits = __reduce_add_sync(0xFFFFFFFF, num_new_incomers);
            if (lane_idx == 0) {
                smem.warp_cnt[warp_idx] = warp_total_hits;
            }
            __syncthreads();

            static_assert(NUM_WARPS <= 32);
            uint32_t stored_warp_hits = lane_idx < NUM_WARPS ? smem.warp_cnt[lane_idx] : 0u;
            uint32_t num_total_hits_in_this_round = __reduce_add_sync(0xFFFFFFFF, stored_warp_hits);

            uint32_t seg_elem_base = current_permuted_segment * NUM_ELEMS_PER_SEG + offset_in_segment;

            // Element e of this thread's slice lives at smem_read_offset + (e ^ chunk_swizzle_mask)
            uint32_t elem_addr_base = cute::cast_smem_ptr_to_uint(buf + smem_read_offset);

            // Start the first hit's smem load before the prefix scan / count exchange below, so that its
            // latency (and the barrier wait) overlaps with them instead of delaying the first store.
            uint32_t first_hit_e = hit_mask != 0 ? __ffs(hit_mask) - 1u : 0u;
            uint32_t first_hit_val = 0;
            if (is_warp_active && warp_total_hits != 0) {
                asm volatile ("ld.shared.u16 %0, [%1];" : "=r"(first_hit_val) : "r"(elem_addr_base + ((first_hit_e ^ chunk_swizzle_mask) << 1)));
            }

            // Lane prefix via one ballot per bit of the (<= 16) hit count: the ballots are independent,
            // so this is much shorter on the critical path than a 5-step shuffle scan.
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND < 32);     // one ballot per bit of num_new_incomers
            uint32_t lane_prefix = 0;
            CUTE_UNROLL
            for (uint32_t k = 0; k < 5; ++k) {
                uint32_t bit = __ballot_sync(0xFFFFFFFF, (num_new_incomers >> k) & 1u) & ((1u << lane_idx) - 1u);
                lane_prefix += (uint32_t)__popc(bit) << k;
            }

            uint32_t dst_slot =
                num_incomers +
                __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_hits : 0u) +
                lane_prefix;

            if (is_warp_active && warp_total_hits != 0) {
                uint32_t dst_ptr = cute::cast_smem_ptr_to_uint(smem.incoming_topk_pairs) + dst_slot * (uint32_t)sizeof(uint64_t);
                if (hit_mask != 0) {
                    asm volatile ("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(dst_ptr), "r"(seg_elem_base + first_hit_e), "r"(first_hit_val) : "memory");
                    dst_ptr += (uint32_t)sizeof(uint64_t);
                }
                uint32_t mask = hit_mask & (hit_mask - 1u);
                while (mask != 0) {
                    uint32_t e = __ffs(mask) - 1u;
                    mask &= mask - 1u;
                    uint32_t val_word;
                    asm volatile ("ld.shared.u16 %0, [%1];" : "=r"(val_word) : "r"(elem_addr_base + ((e ^ chunk_swizzle_mask) << 1)));
                    asm volatile ("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(dst_ptr), "r"(seg_elem_base + e), "r"(val_word) : "memory");
                    dst_ptr += (uint32_t)sizeof(uint64_t);
                }
            }
            Base::advance_perm_state(current_permuted_segment, permuted_segment_stride_per_round, perm_len);

            num_incomers += num_total_hits_in_this_round;
            // No warp may run ahead into the next round: the TMA copy issued there reuses the smem
            // buffer this round is reading, and warp_cnt must not be overwritten while others read it.
            __syncthreads();

            if (++tma_buf_idx == NUM_TMA_LOAD_BUFS) {
                tma_buf_idx = 0;
                tma_buf_phase ^= 1u;
            }

            // On long rows the init window is a tiny fraction of the row, so its pivot stays far above the
            // row's true one and the append rate (hence the store path) stays high for the whole scan:
            // reconstruct early to tighten the pivot. Short rows would pay more for the reconstruct than
            // the extra appends cost, so they keep the plain threshold.
            if (num_incomers >= reconstruct_trigger && main_round_idx + 1 < num_main_rounds) {
                threshold_x2_bits = reconstruct(num_incomers);
                num_incomers = 0;
            }
        }

        if (num_incomers > 0) {
            reconstruct(num_incomers);
            num_incomers = 0;
        }

        return num_survivors;
    }
};

} // namespace topk_select_common
