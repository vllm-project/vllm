#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <type_traits>
#include <stdint.h>
#include <math.h>
#include <iostream>
#endif

#include "hip_float8_impl.h"

struct alignas(1) hip_fp8
{
    struct from_bits_t
    {
    };
    HIP_FP8_HOST_DEVICE static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    hip_fp8() = default;
    HIP_FP8_HOST_DEVICE constexpr hip_fp8(const hip_fp8&) = default;
    HIP_FP8_HOST_DEVICE constexpr hip_fp8(uint8_t v) = delete;
    explicit HIP_FP8_HOST_DEVICE constexpr hip_fp8(uint8_t v, from_bits_t)
        : data(v)
    {
    }

#ifdef __HIP__MI300__
    // NOTE: ON-DEVICE... always optimal bias
    explicit HIP_FP8_DEVICE hip_fp8(float v)
        : data(hip_fp8_impl::to_fp8_from_fp32(v))
    {
    }

    explicit HIP_FP8_DEVICE hip_fp8(_Float16 v)
        : hip_fp8(static_cast<float>(v))
    {
    }

    // Host only implementation using s/w simulation
    explicit HIP_FP8_HOST
#else  // __HIP__MI300__
    // both Host and DEVICE for non-MI300 using s/w simulation
    explicit HIP_FP8_HOST_DEVICE
#endif // __HIP__MI300__
    hip_fp8(float v)
    {
        data = hip_fp8_impl::to_float8<4, 3, float, true /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit HIP_FP8_HOST_DEVICE hip_fp8(double v)
        : hip_fp8(static_cast<float>(v))
    {
    }

#ifdef __HIP__MI300__
    // upcast using device specific intrinsic
    explicit inline HIP_FP8_DEVICE operator float() const
    {
        float fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }

    explicit inline HIP_FP8_HOST operator float() const
#else  // __HIP__MI300__
    explicit inline HIP_FP8_HOST_DEVICE operator float() const
#endif // __HIP__MI300__
    {
        return hip_fp8_impl::from_float8<4, 3, float, true /*negative_zero_nan*/>(data);
    }
};

namespace std
{
inline hip_fp8 sin(hip_fp8 a)
{
    return hip_fp8(sinf(float(a)));
}
inline hip_fp8 cos(hip_fp8 a)
{
    return hip_fp8(cosf(float(a)));
}
HIP_FP8_HOST_DEVICE constexpr hip_fp8 real(const hip_fp8& a)
{
    return a;
}
} // namespace std

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const hip_fp8& f8)
{
    return os << float(f8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline HIP_FP8_HOST_DEVICE float operator+(const float fa, hip_fp8 b)
{
    return (fa + float(b));
}

inline HIP_FP8_HOST_DEVICE float operator+(hip_fp8 a, const float fb)
{
    return (float(a) + fb);
}

inline HIP_FP8_HOST_DEVICE hip_fp8 operator+(hip_fp8 a, hip_fp8 b)
{
    return hip_fp8(float(a) + float(b));
}

inline HIP_FP8_HOST_DEVICE hip_fp8& operator+=(hip_fp8& a, hip_fp8 b)
{
    return a = hip_fp8(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline HIP_FP8_HOST_DEVICE float operator*(hip_fp8 a, hip_fp8 b)
{
    return float(a) * float(b);
}

inline HIP_FP8_HOST_DEVICE float operator*(float a, hip_fp8 b)
{
    return (a * float(b));
}

inline HIP_FP8_HOST_DEVICE float operator*(hip_fp8 a, float b)
{
    return (float(a) * b);
}

inline HIP_FP8_HOST_DEVICE float operator*(int32_t a, hip_fp8 b)
{
    return ((float)a * float(b));
}

inline HIP_FP8_HOST_DEVICE float operator*(double a, hip_fp8 b)
{
    return ((float)a * float(b));
}

// overloading for compare
inline HIP_FP8_HOST_DEVICE bool operator==(hip_fp8 a, hip_fp8 b)
{
    return (a.data == b.data);
}
inline HIP_FP8_HOST_DEVICE bool operator!=(hip_fp8 a, hip_fp8 b)
{
    return (a.data != b.data);
}

inline HIP_FP8_HOST_DEVICE bool operator>=(hip_fp8 a, hip_fp8 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline HIP_FP8_HOST_DEVICE bool operator>(hip_fp8 a, hip_fp8 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
