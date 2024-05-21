/*
Adapted from https://github.com/turboderp/exllamav2 and https://github.com/turboderp/exllama
*/

#ifndef _matrix_view_cuh
#define _matrix_view_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "qdq_util.cuh"

namespace vllm {
namespace gptq {

class MatrixView_half
{
public:
    const half* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_half(const half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ half item(int row, int column) const { return data[row * width + column]; }
    __device__ __forceinline__ half2 item_half2(int row, int column) const { return ((half2*)data)[(row * width + column) / 2]; }
    __device__ __forceinline__ half2 item_half2half2(int row, int column) const { return __half2half2(data[row * width + column]); }
    __device__ __forceinline__ const half* item_ptr(int row, int column) const { return &data[row * width + column]; }

    __device__ __forceinline__ void item4(half (&items)[4], int row, int column) const
    {
        half2* ptr = (half2*) item_ptr(row, column);
        half2 i01 = ptr[0];
        half2 i23 = ptr[1];
        items[0] = __low2half(i01);
        items[1] = __high2half(i01);
        items[2] = __low2half(i23);
        items[3] = __high2half(i23);
    }
    __device__ __forceinline__ void item4_f(float (&items)[4], int row, int column) const
    {
        half2* ptr = (half2*)item_ptr(row, column);
        half2 i01 = ptr[0];
        half2 i23 = ptr[1];
        items[0] = __half2float(__low2half(i01));
        items[1] = __half2float(__high2half(i01));
        items[2] = __half2float(__low2half(i23));
        items[3] = __half2float(__high2half(i23));
    }

    __device__ __forceinline__ void item4_h2(half2 (&items)[4], int row, int column) const
    {
        half2* ptr = (half2*)item_ptr(row, column);
        half2 i01 = ptr[0];
        half2 i23 = ptr[1];
        items[0] = __half2half2(__low2half(i01));
        items[1] = __half2half2(__high2half(i01));
        items[2] = __half2half2(__low2half(i23));
        items[3] = __half2half2(__high2half(i23));
    }
};

class MatrixView_half_rw
{
public:
    half* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_half_rw(half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ half item(int row, int column) const { return data[row * width + column]; }
    __device__ __forceinline__ half2 item_half2(int row, int column) const { return ((half2*)data)[(row * width + column) / 2]; }
    __device__ __forceinline__ half* item_ptr(int row, int column) { return &data[row * width + column]; }
    __device__ __forceinline__ void set(int row, int column, half value) { data[row * width + column] = value; }
    __device__ __forceinline__ void set_half2(int row, int column, half2 value) { ((half2*)data)[(row * width + column) / 2] = value; }

    __device__ __forceinline__ void set4(int row, int column, half v0, half v1, half v2, half v3)
    {
        half2 v01 = __halves2half2(v0, v1);
        half2 v23 = __halves2half2(v2, v3);
        half2* ptr = (half2*) item_ptr(row, column);
        ptr[0] = v01;
        ptr[1] = v23;
    }
};

class MatrixView_q4_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q4_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        return (data[row * width / 8 + column / 8] >> shift) & 0x0f;
    }

    __device__ __forceinline__ void item2(int (&items)[2], int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        uint32_t d = data[row * width / 8 + column / 8] >> shift;
        items[0] = d & 0x0f;
        items[1] = (d >> 4) & 0x0f;
    }

    __device__ __forceinline__ void item4(int (&items)[4], int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        uint32_t d = data[row * width / 8 + column / 8] >> shift;
        items[0] = d & 0x0f;
        items[1] = (d >> 4) & 0x0f;
        items[2] = (d >> 8) & 0x0f;
        items[3] = (d >> 12) & 0x0f;
    }
};

class MatrixView_q4_column
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q4_column(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (row & 0x07) * 4;
        return (data[row / 8 * width + column] >> shift) & 0x0f;
    }

    __device__ __forceinline__ uint32_t item_uint32_t(int row, int column) { return data[row / 8 * width + column]; }
    __device__ __forceinline__ const uint32_t* item_uint32_ptr(int row, int column) { return &data[row / 8 * width + column]; }
};

class MatrixView_q2_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q2_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (column & 0x0f) * 2;
        return (data[row * width / 16 + column / 16] >> shift) & 0x03;
    }

    __device__ __forceinline__ void item2(int (&items)[2], int row, int column) const
    {
        int shift = (column & 0x0f) * 2;
        uint32_t d = data[row * width / 16 + column / 16] >> shift;
        items[0] = d & 0x03;
        items[1] = (d >> 2) & 0x03;
    }

    __device__ __forceinline__ void item4(int (&items)[4], int row, int column) const
    {
        int shift = (column & 0x0f) * 2;
        uint32_t d = data[row * width / 16 + column / 16] >> shift;
        items[0] = d & 0x03;
        items[1] = (d >> 2) & 0x03;
        items[2] = (d >> 4) & 0x03;
        items[3] = (d >> 6) & 0x03;
    }
};

class MatrixView_q3_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q3_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int z_w = column * 3 / 32;
        int z_mod =  column & 0x1f;

        if (z_mod == 10) {
            return (data[row * width * 3 / 32 + z_w] >> 30) | ((data[row * width * 3 / 32 + (z_w + 1)] << 2) & 0x4);
        } else if (z_mod == 21) {
            return (data[row * width * 3 / 32 + z_w] >> 31) | ((data[row * width * 3 / 32 + (z_w + 1)] << 1) & 0x6);
        } else if (z_mod < 10) {
            return (data[row * width * 3 / 32 + z_w] >> (z_mod * 3)) & 0x07;
        } else if (z_mod < 21) {
            return (data[row * width * 3 / 32 + z_w] >> (z_mod * 3  - 32)) & 0x07;
        } else {
            return (data[row * width * 3 / 32 + z_w] >> (z_mod * 3  - 64)) & 0x07;
        }
    }

    __device__ __forceinline__ void item4(int (&items)[4], int row, int column) const
    {
        int shift = (column & 0x1f);
        uint32_t d;
        if (shift <= 4) {
            d = data[row * width / 32 * 3 + column * 3 / 32] >> (shift * 3);
        } else if (shift == 8) {
            d = (data[row * width / 32 * 3 + column * 3 / 32] >> 24) | ((data[row * width / 32 * 3 + column * 3 / 32 + 1] & 0x0f) << 8);
        } else if (shift <= 16) {
            d = data[row * width / 32 * 3 + column * 3 / 32] >> (shift * 3 - 32);
        } else if (shift == 20) {
            d = (data[row * width / 32 * 3 + column * 3 / 32] >> 28) | ((data[row * width / 32 * 3 + column * 3 / 32 + 1] & 0xff) << 4);
        } else {
            d = data[row * width / 32 * 3 + column * 3 / 32] >> (shift * 3 - 64);
        }
        items[0] = d & 0x07;
        items[1] = (d >> 3) & 0x07;
        items[2] = (d >> 6) & 0x07;
        items[3] = (d >> 9) & 0x07;
    }
};

class MatrixView_q8_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q8_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (column & 0x03) * 8;
        return (data[row * width / 4 + column / 4] >> shift) & 0xff;
    }

    __device__ __forceinline__ void item2(int (&items)[2], int row, int column) const
    {
        int shift = (column & 0x03) * 8;
        uint32_t d = data[row * width / 4 + column / 4] >> shift;
        items[0] = d & 0xff;
        items[1] = (d >> 8) & 0xff;
    }

    __device__ __forceinline__ void item4(int (&items)[4], int row, int column) const
    {
        int shift = (column & 0x03) * 2;
        uint32_t d = data[row * width / 4 + column / 4] >> shift;
        items[0] = d & 0xff;
        items[1] = (d >> 8) & 0xff;
        items[2] = (d >> 16) & 0xff;
        items[3] = (d >> 24) & 0xff;
    }
};

}  // namespace gptq
}  // namespace vllm
#endif
