#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <vector>

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#endif

#include "kerutils/common/common.h"

namespace kerutils {

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;

        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                                     \
    do {                                                                                         \
        if (not (cond)) {                                                                        \
            char _ku_buf[1024];                                                                  \
            int _ku_len = snprintf(_ku_buf, sizeof(_ku_buf),                                    \
                                   "Assertion `%s` failed (%s:%d)", #cond, __FILE__, __LINE__); \
            __VA_OPT__(                                                                          \
                _ku_len += snprintf(_ku_buf + _ku_len, sizeof(_ku_buf) - _ku_len,              \
                                    ": " __VA_ARGS__);                                          \
            )                                                                                    \
            fprintf(stderr, "%s\n", _ku_buf);                                                    \
            THROW_KU_EXCEPTION("Assertion", _ku_buf);                                            \
        }                                                                                        \
    } while(0)

#ifdef KERUTILS_IS_BUILD_ON_CUDA

#define KU_CUDA_CHECK(call)                                                                                   \
do {                                                                                                          \
    cudaError_t status_ = (call);                                                                             \
    if (status_ != cudaSuccess) {                                                                             \
        char _ku_buf[1024];                                                                                   \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUDA error (%s:%d): %s", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                    \
        THROW_KU_EXCEPTION("CUDA", _ku_buf);                                                                  \
    }                                                                                                         \
} while(0)

#define KU_CUTLASS_CHECK(call)                                                                                   \
do {                                                                                                             \
    cutlass::Status status_ = (call);                                                                            \
    if (status_ != cutlass::Status::kSuccess) {                                                                 \
        char _ku_buf[1024];                                                                                      \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUTLASS error (%s:%d): %d", __FILE__, __LINE__, static_cast<int>(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                       \
        THROW_KU_EXCEPTION("CUTLASS", _ku_buf);                                                                 \
    }                                                                                                            \
} while(0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template<typename T>
inline __host__ __device__ constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline __host__ __device__ constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

template<typename T, T LOWER_BOUND = 1>
inline __host__ __device__ constexpr T find_next_power_of_2(const T& x) {
    if (x <= LOWER_BOUND)
        return LOWER_BOUND;
    return find_next_power_of_2<T, LOWER_BOUND*2>(x);
}

// A wrapper for make_tensor_map
static inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t> &size,
    const std::vector<uint64_t> &strides,   // PAY ATTENTION: In BYTES
    const std::vector<uint32_t> &box_size,
    void* global_ptr,
    CUtensorMapDataType data_type,
    CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t> &element_strides_ = {}
) {
    int dim = size.size();
    KU_ASSERT(dim >= 1);

    std::vector<uint32_t> element_strides;
    if (element_strides_.empty()) {
        for (int i = 0; i < dim; ++i)
            element_strides.push_back(1);
    } else {
        element_strides = element_strides_;
    }
    KU_ASSERT(strides.size() == (uint32_t)dim-1 && box_size.size() == (uint32_t)dim && element_strides.size() == (uint32_t)dim);

    auto call_cuTensorMapEncodeTiled = [&]<typename... Args>(Args... args) {
        cudaDriverEntryPointQueryResult cuda_status;
        void* pfn = nullptr;
#if (__CUDACC_VER_MAJOR__ > 12)
        KU_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &pfn, 12000,
            cudaEnableDefault,
            &cuda_status));
#else
        KU_CUDA_CHECK(cudaGetDriverEntryPoint(
            "cuTensorMapEncodeTiled",
            &pfn,
            cudaEnableDefault,
            &cuda_status));
#endif
        if (cuda_status != cudaDriverEntryPointSuccess) {
            KU_ASSERT(false, "Failed to load `cuTensorMapEncodeTiled`. cuda_status = %d", cuda_status);
        }
        return reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(pfn)(args...); \
    };

    CUtensorMap result;
    CUresult ret_code = call_cuTensorMapEncodeTiled(
        &result,
        data_type,
        dim,
        global_ptr,
        size.data(),
        strides.data(),
        box_size.data(),
        element_strides.data(),
        interleave_mode,
        swizzle_mode,
        l2_promotion,
        oob_fill
    );
    if (ret_code != CUresult::CUDA_SUCCESS) {
        auto print_vector = [&](auto t, const char* fmt, const char end='\n') {
            for (auto elem : t) {
                printf(fmt, elem);
            }
            printf("%c", end);
        };
        fprintf(stderr, "Failed to create tensormap\n");
        fprintf(stderr, "Dim: %d\n", dim);
        printf("size: "); print_vector(size, "%lu ");
        printf("strides: "); print_vector(strides, "%lu ");
        printf("box_size: "); print_vector(box_size, "%u ");
        printf("element_strides: "); print_vector(element_strides, "%u ");
        printf("global ptr: 0x%lx\n", (int64_t)global_ptr);
        printf("data_type: %d\n", (int)data_type);
        printf("swizzle_mode: %d\n", (int)swizzle_mode);
        printf("l2_promotion: %d\n", (int)l2_promotion);
        printf("interleave_mode: %d\n", (int)interleave_mode);
        printf("oob_fill: %d\n", (int)oob_fill);
        KU_ASSERT(false);
    }
    return result;
}

// Given strides (in number of elements), this function converts their datatype in uint64_t and then multiplies by elem_size
template<typename T>
static inline std::vector<uint64_t> make_stride_helper(const std::vector<T> &strides_in_elems, size_t elem_size) {
    std::vector<uint64_t> res;
    for (auto stride : strides_in_elems) {
        res.push_back(((uint64_t)stride) * elem_size);
    }
    return res;
}

struct KernelLaunchConfig {
    dim3 grid;
    dim3 block;
    size_t dynamic_smem;
    cudaStream_t stream;
    dim3 cluster{1, 1, 1};
    bool use_pdl{false};
    bool cooperative{false};
};

namespace detail {

template <class Arg>
void* kernel_arg_ptr(Arg&& arg) {
    return const_cast<void*>(reinterpret_cast<const void*>(std::addressof(arg)));
}

} // namespace detail

template<typename KernelFunc, typename... Args>
void launch_kernel(const KernelLaunchConfig &cfg, KernelFunc kernel, Args&&... args) {
    if (cfg.dynamic_smem > 0) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(cfg.dynamic_smem)));
    }

    const bool is_cluster     = !(cfg.cluster.x == 1 && cfg.cluster.y == 1 && cfg.cluster.z == 1);
    const bool need_kernel_ex = is_cluster || cfg.use_pdl;

    if (is_cluster && cfg.cooperative) {
        KU_ASSERT(false, "cluster and cooperative launch are mutually exclusive");
    }

    if (cfg.cooperative) {
        void* kernel_args[sizeof...(Args) > 0 ? sizeof...(Args) : 1] = {};
        if constexpr (sizeof...(Args) > 0) {
            size_t i = 0;
            ((kernel_args[i++] = detail::kernel_arg_ptr(std::forward<Args>(args))), ...);
        }
        KU_CUDA_CHECK(cudaLaunchCooperativeKernel(
            kernel, cfg.grid, cfg.block,
            sizeof...(Args) > 0 ? kernel_args : nullptr,
            static_cast<unsigned int>(cfg.dynamic_smem), cfg.stream));
    } else if (need_kernel_ex) {
        if (is_cluster) {
            KU_ASSERT(cfg.grid.x % cfg.cluster.x == 0 &&
                      cfg.grid.y % cfg.cluster.y == 0 &&
                      cfg.grid.z % cfg.cluster.z == 0);
            if (cfg.cluster.x * cfg.cluster.y * cfg.cluster.z > 8) {
                KU_CUDA_CHECK(cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(kernel),
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            }
        }

        const unsigned int num_attrs = is_cluster ? 2 : 1;
        cudaLaunchAttribute attrs[2];
        if (is_cluster) {
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim = {cfg.cluster.x, cfg.cluster.y, cfg.cluster.z};
        }
        attrs[num_attrs - 1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[num_attrs - 1].val.programmaticStreamSerializationAllowed = cfg.use_pdl ? 1 : 0;

        cudaLaunchConfig_t config = {
            {cfg.grid.x, cfg.grid.y, cfg.grid.z},
            {cfg.block.x, cfg.block.y, cfg.block.z},
            cfg.dynamic_smem,
            cfg.stream,
            attrs,
            num_attrs
        };
        KU_CUDA_CHECK(cudaLaunchKernelEx(
            &config, kernel, std::forward<Args>(args)...));
    } else {
        kernel<<<cfg.grid, cfg.block, cfg.dynamic_smem, cfg.stream>>>(
            std::forward<Args>(args)...);
    }

    KU_CHECK_KERNEL_LAUNCH();
}

#endif  // KERUTILS_IS_BUILD_ON_CUDA

}   // namespace kerutils
