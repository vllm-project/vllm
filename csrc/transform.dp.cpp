// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "custom_cuda_layers.h"

#define rows_trans 16
#define cols_trans 16

template <typename T>
void Transpose_Kernel(const T* inp, T* out, int row_width, int col_width,
                      const sycl::nd_item<3> &item_ct1, T *data_block)
{

    int r = item_ct1.get_local_id(2) / cols_trans;
    int c = item_ct1.get_local_id(2) % cols_trans;

    int m = row_width / cols_trans;

    int i = item_ct1.get_group(2) / m * rows_trans + r;
    int j = item_ct1.get_group(2) % m * cols_trans + c;

    int row_stride = rows_trans / ((rows_trans * cols_trans + THREADS - 1) / THREADS);

    for (int k = 0; k < rows_trans; k += row_stride)
        data_block[(k + r) * cols_trans + c] = inp[(i + k) * row_width + j];

    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    i = item_ct1.get_group(2) % m * rows_trans + r;
    j = item_ct1.get_group(2) / m * cols_trans + c;

    for (int k = 0; k < rows_trans; k += row_stride)
        out[(i + k) * col_width + j] = data_block[c * cols_trans + r + k];
}

template <>
void Transpose<__half>(const __half* inp_mat,
                       __half* out_mat,
                       int rows,
                       int cols,
                       cudaStream_t stream)
{
    int threads = THREADS;

    Transpose_Kernel<__half><<<(rows * cols + threads - 1) / threads, threads, 0, stream>>>(
        inp_mat, out_mat, cols, rows);
}

template <>
void Transpose<float>(const float* inp_mat, float* out_mat, int rows, int cols, cudaStream_t stream)
{
    int threads = THREADS;

    Transpose_Kernel<float><<<(rows * cols + threads - 1) / threads, threads, 0, stream>>>(
        inp_mat, out_mat, cols, rows);
}

template <typename T>
void transform_0213(T* output,
                               const T* vals,
                               int hidden_dim,
                               int seq_length,
                               int heads,
                               int head_ext);

template <>
void transform_0213<float>(float* output,
                                      const float* vals,
                                      int hidden_dim,
                                      int seq_length,
                                      int heads,
                                      int head_ext,
                                      const sycl::nd_item<3> &item_ct1)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = item_ct1.get_group(2);            // Batch
    int d1 = item_ct1.get_group(1) / head_ext; // Sequence ID (0-127)
    int d2 = item_ct1.get_local_id(1) + (item_ct1.get_group(1) % head_ext) *
                                            (heads / head_ext); // Head (0-11)
    int d3 = item_ct1.get_local_id(2); // Values (groups of 4)

    const sycl::float4 *vals_vec = reinterpret_cast<const sycl::float4 *>(vals);
    sycl::float4 *output_vec = reinterpret_cast<sycl::float4 *>(output);

    sycl::float4 inputs =
        vals_vec[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = inputs;
}

template <>
void transform_0213<__half>(__half* output,
                                       const __half* vals,
                                       int hidden_dim,
                                       int seq_length,
                                       int heads,
                                       int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y / head_ext;                                       // Sequence ID (0-127)
    int d2 = threadIdx.y + (blockIdx.y % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr[1];

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_arr[0] = vals_vec[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = vals_arr[0];
#endif
}

template <>
void launch_transform_0213<float>(float *output, const float *vals,
                                  int batch_size, int seq_length,
                                  int hidden_dim, int heads,
                                  dpct::queue_ptr stream)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    sycl::range<3> block_dim(1, (heads / head_ext), hidden_dim / heads);
    sycl::range<3> grid_dim(1, (seq_length * head_ext), batch_size);

    transform_0213<float>
        <<<grid_dim, block_dim, 0, stream>>>(output, vals, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_transform_0213<__half>(__half* output,
                                   const __half* vals,
                                   int batch_size,
                                   int seq_length,
                                   int hidden_dim,
                                   int heads,
                                   cudaStream_t stream)
{
    hidden_dim >>= 3;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, (seq_length * head_ext));
    transform_0213<__half>
        <<<grid_dim, block_dim, 0, stream>>>(output, vals, hidden_dim, seq_length, heads, head_ext);
}

// Bias add
template <typename T>
void bias_add_transform_0213(T* output,
                                        const T* vals,
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        int heads,
                                        int head_ext);

template <>
void bias_add_transform_0213<float>(float* output,
                                               const float* vals,
                                               const float* bias,
                                               int hidden_dim,
                                               int seq_length,
                                               int heads,
                                               int head_ext,
                                               const sycl::nd_item<3> &item_ct1)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = item_ct1.get_group(2);             // Batch
    int d1 = item_ct1.get_group(1);             // Sequence ID (0-127)
    int cnt = item_ct1.get_group(0) / head_ext; // Hidden count
    int d2 = item_ct1.get_local_id(1) + (item_ct1.get_group(0) % head_ext) *
                                            (heads / head_ext); // Head (0-11)
    int d3 = item_ct1.get_local_id(2); // Values (groups of 4)

    const sycl::float4 *vals_vec = reinterpret_cast<const sycl::float4 *>(vals);
    const sycl::float4 *bias_vec = reinterpret_cast<const sycl::float4 *>(bias);
    sycl::float4 *output_vec = reinterpret_cast<sycl::float4 *>(output);

    sycl::float4 inputs =
        vals_vec[d0 * d0_stride * (item_ct1.get_group_range(0) / head_ext) +
                 cnt * d1_stride +
                 d1 * d1_stride * (item_ct1.get_group_range(0) / head_ext) +
                 d2 * d2_stride + d3];
    sycl::float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3];

    sycl::float4 outputs;
    outputs.x() = inputs.x() + biases.x();
    outputs.y() = inputs.y() + biases.y();
    outputs.z() = inputs.z() + biases.z();
    outputs.w() = inputs.w() + biases.w();

    output_vec[cnt * d0_out_stride * item_ct1.get_group_range(2) +
               d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride +
               d3] = outputs;
}

#define ATTN_H 3
#define MAX_SEQ_LINE 10

template <>
void bias_add_transform_0213<__half>(__half* output,
                                                const __half* vals,
                                                const __half* bias,
                                                int hidden_dim,
                                                int seq_length,
                                                int heads,
                                                int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr;
    float4 bias_arr;
    float4 output_arr;
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(&bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    bias_vec += (cnt * d1_stride);
    bias_vec += (d2 * d2_stride);

    output_vec += (cnt * d0_stride * gridDim.x);
    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    bias_arr = bias_vec[d3];
    vals_arr = vals_vec[d3];

#if defined(__ACC_HALF__)
    output_half[0] = vals_half[0] + bias_half[0];
    output_half[1] = vals_half[1] + bias_half[1];
    output_half[2] = vals_half[2] + bias_half[2];
    output_half[3] = vals_half[3] + bias_half[3];
#else
    float2 bias_arr_f[4];
    float2 vals_arr_f[4];
#pragma unroll
    for (int l = 0; l < 4; l++) {
        bias_arr_f[l] = __half22float2(bias_half[l]);
        vals_arr_f[l] = __half22float2(vals_half[l]);
        vals_arr_f[l].x += bias_arr_f[l].x;
        vals_arr_f[l].y += bias_arr_f[l].y;
        output_half[l] = __float22half2_rn(vals_arr_f[l]);
    }
#endif
    output_vec[d3] = output_arr;

#endif
}

void bias_add_transform_0213_v2(__half* output,
                                           const __half* vals,
                                           const __half* bias,
                                           int hidden_dim,
                                           int seq_length,
                                           int heads)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;
    int iteration_stride = d1_stride * blockDim.z;  // Hidden * 3 / 8
    int batch_stride = d0_stride * blockDim.z;      // Hidden * S * 3 / 8

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;    // Batch
    int d1 = blockIdx.y;    // Sequence ID (0-127)
    int cnt = threadIdx.z;  // blockIdx.z; // Hidden count
    int d2 = threadIdx.y;   // Head (0-11)
    int d3 = threadIdx.x;   // Values (groups of 4)

    float4 vals_arr[1];
    float4 bias_arr[1];
    float4 output_arr[1];
    __half2* vals_half = reinterpret_cast<__half2*>(vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int iter_index = cnt * d1_stride + d2 * d2_stride + d3;
    int input_offset = d0 * batch_stride + d1 * (iteration_stride << 1);
    bias_arr[0] = bias_vec[iter_index];

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        vals_arr[0] = vals_vec[input_offset + iter_id];

        output_half[0] = vals_half[0] + bias_half[0];
        output_half[1] = vals_half[1] + bias_half[1];
        output_half[2] = vals_half[2] + bias_half[2];
        output_half[3] = vals_half[3] + bias_half[3];

        in_data[iter_id] = output_arr[0];
    }
    __syncthreads();

    iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_out_stride * gridDim.x);
    int head_count = (d2 >> 1) + cnt * (blockDim.y >> 1);

    int out_index = d0 * d0_out_stride + d1 * (d1_out_stride << 1) + d3 + (d2 % 2) * d2_stride;

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = (iter * iteration_stride) + head_count;
        int iter_offset =
            (iter_row % blockDim.y) * d2_out_stride + (iter_row / blockDim.y) * matrix_stride;
        output_vec[out_index + iter_offset] =
            in_data[iter_row * d2_stride + d3 + (d2 % 2) * (d1_stride * blockDim.z)];
    }
#endif
}

// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(float* output,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           int hidden_dim,
                                           int heads,
                                           cudaStream_t stream,
                                           int trans_count)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    bias_add_transform_0213<float><<<grid_dim, block_dim, 0, stream>>>(
        output, vals, bias, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_bias_add_transform_0213<__half>(__half* output,
                                            const __half* vals,
                                            const __half* bias,
                                            int batch_size,
                                            int seq_length,
                                            int hidden_dim,
                                            int heads,
                                            cudaStream_t stream,
                                            int trans_count)
{
    hidden_dim >>= 3;
    if (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 block_dim(hidden_dim / heads, (heads / head_ext));
        dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));
        bias_add_transform_0213<__half><<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads, head_ext);
    } else {
        dim3 block_dim(hidden_dim / heads, heads, trans_count);
        dim3 grid_dim(batch_size, seq_length / 2);
        bias_add_transform_0213_v2<<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads);
    }
}

template <typename T>
void transform4d_0213(T* out,
                                 const T* in,
                                 int heads,
                                 int seq_length,
                                 int hidden_dim,
                                 int head_ext);

template <>
void transform4d_0213<float>(float* out,
                                        const float* in,
                                        int heads,
                                        int seq_length,
                                        int hidden_dim,
                                        int head_ext,
                                        const sycl::nd_item<3> &item_ct1)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = d0_stride / heads;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = hidden_dim;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_group(1) /
             ((seq_length - 1) / item_ct1.get_local_range(1) + 1); // Head
    int d2 = (item_ct1.get_local_id(1) +
              item_ct1.get_local_range(1) * item_ct1.get_group(1)) %
             seq_length;
    int cnt = item_ct1.get_group(0);
    int d3 = item_ct1.get_local_id(2); // Values (groups of 8)

    if (d2 < seq_length) {
        const sycl::float4 *in_vec = reinterpret_cast<const sycl::float4 *>(in);
        sycl::float4 *out_vec = reinterpret_cast<sycl::float4 *>(out);

        sycl::float4 vals_vec =
            in_vec[cnt * d0_stride * item_ct1.get_group_range(2) +
                   d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
        out_vec[d0 * d0_out_stride * item_ct1.get_group_range(0) +
                cnt * d2_out_stride + d1 * d1_out_stride +
                d2 * d2_out_stride * item_ct1.get_group_range(0) + d3] =
            vals_vec;
    }
}

template <>
void transform4d_0213<__half>(__half* out,
                                         const __half* in,
                                         int heads,
                                         int seq_length,
                                         int hidden_dim,
                                         int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE

    int d0_stride = hidden_dim * (seq_length / head_ext);
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head
    int d2 = blockIdx.z / head_ext;                                       // Sequence
    int cnt = blockIdx.y;                                                 // Hidden count
    int d3 = threadIdx.x;                                                 // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    in_vec += (cnt * d0_stride * gridDim.x);
    in_vec += (d0 * d0_stride);
    in_vec += (d2 * d2_stride);
    in_vec += (d1 * d2_stride * seq_length);

    out_vec += (cnt * d1_stride);
    out_vec += (d1 * d2_stride);
    out_vec += (d0 * d0_stride * gridDim.y);
    out_vec += (d2 * d1_stride * gridDim.y);

    out_vec[d3] = in_vec[d3];

#endif
}

void transform4d_0213_v2(__half* out,
                                    const __half* in,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;    // Batch
    int d1 = threadIdx.y;   // Head
    int d2 = blockIdx.y;    // Sequence
    int cnt = threadIdx.z;  // Hidden count
    int d3 = threadIdx.x;   // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int input_offset = d0 * d0_stride + d2 * (d2_stride << 1) + d3 + (d1 % 2) * d2_stride;
    int head_count = (d1 >> 1) + cnt * (blockDim.y >> 1);
    int iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_stride * gridDim.x);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = iter * iteration_stride + head_count;
        int iter_offset = (iter_row % blockDim.y) * d2_stride;

        in_data[d3 + iter_offset + (iter_row / blockDim.y + (d1 % 2) * blockDim.z) * d1_stride] =
            in_vec[input_offset + iter_offset * seq_length +
                   (iter_row / blockDim.y) * matrix_stride];
    }
    __syncthreads();

    iteration_stride = d1_stride * blockDim.z;
    int iter_index = cnt * d1_stride + d1 * d2_stride + d3;
    int output_offset = d0 * d0_stride * blockDim.z + d2 * (iteration_stride << 1);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        out_vec[output_offset + iter_id] = in_data[iter_id];
    }
#endif
}

// 3 * [B A S N] - > [B S C*H]
template <>
void launch_transform4d_0213<float>(float *out, const float *in, int batch_size,
                                    int heads, int seq_length, int hidden_dim,
                                    dpct::queue_ptr stream, int trans_count)
{
    hidden_dim >>= 2;
    sycl::range<3> grid_dims(trans_count, heads * ((seq_length - 1) / 8 + 1),
                             batch_size);
    sycl::range<3> block_dims(1, 8, hidden_dim / heads);
    transform4d_0213<float>
        <<<grid_dims, block_dims, 0, stream>>>(out, in, heads, seq_length, hidden_dim, 1);
}

template <>
void launch_transform4d_0213<__half>(__half* out,
                                     const __half* in,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     int trans_count)
{
    hidden_dim >>= 3;
    if (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 grid_dims(batch_size, trans_count, (seq_length * head_ext));
        dim3 block_dims(hidden_dim / heads, (heads / head_ext));
        transform4d_0213<__half><<<grid_dims, block_dims, 0, stream>>>(
            out, in, heads, seq_length, hidden_dim, head_ext);
    } else {
        dim3 grid_dims(batch_size, seq_length / 2);
        dim3 block_dims(hidden_dim / heads, heads, trans_count);
        transform4d_0213_v2<<<grid_dims, block_dims, 0, stream>>>(
            out, in, heads, seq_length, hidden_dim);
    }
}
