#include "q_matrix.cuh"
#include "matrix_view.cuh"

#include "qdq_4.cuh"

#define BLOCK_KN_SIZE 128

#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// Shuffle quantized data on load

__global__ void shuffle_kernel
(
    uint32_t* __restrict__ b_q_weight,
    const int size_k,
    const int size_n
)
{
    int n = blockIdx.x * THREADS_X + threadIdx.x;
    if (n >= size_n) return;
    int k = 0;
    uint32_t* b_ptr = b_q_weight + n;
    while (k < size_k) { shuffle_4bit_8 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  8; }
}


// QMatrix constructor

QMatrix::QMatrix
(
    const int _device,
    const int _height,
    const int _width,
    const int _groups,

    uint32_t* _q_weight,
    uint16_t* _q_perm,
    uint16_t* _q_invperm,

    uint32_t* _gptq_qzeros,
    half* _gptq_scales,
    uint32_t* _gptq_g_idx,

    half* _temp_dq
) :
    device(_device),
    height(_height),
    width(_width),
    groups(_groups),
    temp_dq(_temp_dq)
{
    cudaSetDevice(device);

    cuda_q_weight = _q_weight;
    cuda_q_perm = _q_perm;
    cuda_q_invperm = _q_invperm;
    cuda_gptq_qzeros = _gptq_qzeros;
    cuda_gptq_scales = _gptq_scales;

    is_gptq = true;

    groupsize = 1;
    while (groupsize * groups < height) groupsize *= 2;

    if (_gptq_g_idx) make_sequential(_gptq_g_idx);

    // Shuffle quantized data

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = 1;

    shuffle_kernel<<<gridDim, blockDim>>>(cuda_q_weight, height, width);
}


// Reconstruct b[k,n] (GPTQ)

__global__ void reconstruct_gptq_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    const uint16_t* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b
)
{
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int offset_k = BLOCK_KN_SIZE * blockIdx.y;
    int offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

    // Preload remapping table

    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    int t = threadIdx.x;

    if (b_q_perm)
    {
        if (offset_k + t < size_k)
            perm[t] = b_q_perm[offset_k + t];
    }

    // Column

    int n = offset_n + t * 4;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    // Initial zeros/scale

    int zeros[4];
    half2 scales[4];
    half2 z1z16[4][2];
    half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);
    dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

    __syncthreads();

    int k = offset_k;
    int lk = 0;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
            dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
        }

        for (int p = 0; p < 4; p++)
        {
            half2 dq[4][4];
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n, false);
            dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n, false);
            dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n, false);
            dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n, false);

            b_ptr += size_n;
            //half* dqh = (half*)dq;
            if (b_q_perm)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
            else
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(offset_k + lk++, n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(offset_k + lk++, n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
        }
        k += 32;
    }
}

void QMatrix::reconstruct(half* out)
{
    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);
    gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);

    reconstruct_gptq_kernel<<<gridDim, blockDim>>>
    (
        cuda_q_weight,
        cuda_q_perm,
        cuda_gptq_qzeros,
        cuda_gptq_scales,
        height,
        width,
        groupsize,
        groups,
        out
    );
}

__global__ void make_sequential_kernel
(
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ w_new,
    const uint16_t* __restrict__ q_perm,
    const int w_height,
    const int w_width
)
{
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;

    int w2_column = THREADS_X * blockIdx.x + threadIdx.x;
    if (w2_column >= w2_stride) return;

    int w_new2_row = blockIdx.y;

    int q_perm_idx = w_new2_row << 3;

    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = q_perm[q_perm_idx++];

        int w2_row = source_row >> 3;
        int w2_subrow = source_row & 0x07;
        int w2_row_shift = w2_subrow << 2;
        int wnew2_row_shift = i << 2;

        uint64_t src = w2[w2_row * w2_stride + w2_column];
        src >>= w2_row_shift;
        src &= 0x0000000f0000000f;
        src <<= wnew2_row_shift;
        dst |= src;
    }

    w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

void QMatrix::make_sequential(const uint32_t* cpu_g_idx)
{
    uint32_t* cuda_new_qweight = NULL;
    cudaMalloc(&cuda_new_qweight, height / 8 * width * sizeof(uint32_t));

    uint32_t* cpu_g_idx_map = (uint32_t*) calloc(groups, sizeof(uint32_t));
    uint32_t* cpu_x_map = (uint32_t*) malloc(height * sizeof(uint32_t));
    uint32_t* cpu_x_map_inv = (uint32_t*) malloc(height * sizeof(uint32_t));

    // Group histogram

    for (int i = 0; i < height; i++) cpu_g_idx_map[cpu_g_idx[i]]++;

    // Group map

    for (int i = 0, acc = 0; i < groups; i++)
    {
        short tmp = cpu_g_idx_map[i];
        cpu_g_idx_map[i] = acc;
        acc += tmp;
    }

    // X map (inverse)

    for (int row = 0; row < height; row++)
    {
        uint32_t target_group = cpu_g_idx[row];
        uint32_t target_row = cpu_g_idx_map[target_group];
        cpu_g_idx_map[target_group]++;
        cpu_x_map_inv[row] = target_row;
    }

    // X map

    for (int row = 0; row < height; row++) cpu_x_map[cpu_x_map_inv[row]] = row;

    // Reduce to uint16_t

    uint16_t* cpu_x_map16 = (uint16_t*)cpu_x_map;
    uint16_t* cpu_x_map_inv16 = (uint16_t*)cpu_x_map_inv;
    for (int row = 0; row < height; row++) cpu_x_map16[row] = (uint16_t) cpu_x_map[row];
    for (int row = 0; row < height; row++) cpu_x_map_inv16[row] = (uint16_t) cpu_x_map_inv[row];

    // Move to CUDA

    cudaMemcpyAsync(cuda_q_perm, cpu_x_map16, height * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_q_invperm, cpu_x_map_inv16, height * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Rearrange rows in w

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = height / 8;

    make_sequential_kernel<<<gridDim, blockDim>>>
    (
        cuda_q_weight,
        cuda_new_qweight,
        cuda_q_perm,
        height / 8,
        width
    );

    // Replace qweights

    cudaMemcpyAsync(cuda_q_weight, cuda_new_qweight, height / 8 * width * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // Cleanup

    cudaDeviceSynchronize();

    cudaFree(cuda_new_qweight);
    free(cpu_g_idx_map);
    free(cpu_x_map);
    free(cpu_x_map_inv);
}
