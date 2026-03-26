/*
 * TurboQuant CUDA Kernels
 * 
 * Efficient bit-packing and bit-unpacking operations for KV cache quantization.
 * These kernels avoid Python loop overhead and leverage GPU parallelism for
 * high-performance quantization on the critical path.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace turboquant {
namespace cuda {

/*
 * CUDA kernel for packing low-bit integers into 32-bit words.
 * 
 * Each thread handles packing a single batch element's values.
 * Within a thread, we process all values for that batch element,
 * calculating bit placement and handling cross-boundary cases.
 */
__global__ void pack_lowbit_kernel(
    const uint32_t* __restrict__ values,  // [batch_size, length]
    uint32_t* __restrict__ packed,         // [batch_size, packed_width]
    int length,
    int bits,
    int packed_width,
    int batch_size
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Get pointers to this batch element's data
    const uint32_t* batch_values = values + batch_idx * length;
    uint32_t* batch_packed = packed + batch_idx * packed_width;
    
    // Initialize output to zero (cooperative effort across threads)
    for (int i = thread_id; i < packed_width; i += num_threads) {
        batch_packed[i] = 0;
    }
    __syncthreads();
    
    // Process values - each thread handles a subset
    uint32_t value_mask = (1u << bits) - 1;
    
    for (int idx = thread_id; idx < length; idx += num_threads) {
        uint32_t value = batch_values[idx] & value_mask;
        int bit_offset = idx * bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        
        // Place primary bits
        uint32_t primary = value << offset;
        atomicOr(batch_packed + word_idx, primary);
        
        // Handle spill to next word if necessary
        int spill = offset + bits - 32;
        if (spill > 0) {
            uint32_t spill_bits = value >> (bits - spill);
            atomicOr(batch_packed + word_idx + 1, spill_bits);
        }
    }
}


/*
 * CUDA kernel for unpacking low-bit integers from 32-bit words.
 * 
 * Each thread handles unpacking a single batch element.
 * Within a thread, we gather bits from potentially multiple words
 * and reconstruct the original values.
 */
__global__ void unpack_lowbit_kernel(
    const uint32_t* __restrict__ packed,   // [batch_size, packed_width]
    uint32_t* __restrict__ values,         // [batch_size, length]
    int length,
    int bits,
    int packed_width,
    int batch_size
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Get pointers to this batch element's data
    const uint32_t* batch_packed = packed + batch_idx * packed_width;
    uint32_t* batch_values = values + batch_idx * length;
    
    uint32_t value_mask = (1u << bits) - 1;
    
    // Process values - each thread handles a subset
    for (int idx = thread_id; idx < length; idx += num_threads) {
        int bit_offset = idx * bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        
        // Extract primary bits
        uint32_t value = (batch_packed[word_idx] >> offset) & value_mask;
        
        // Handle spill from next word if necessary
        int spill = offset + bits - 32;
        if (spill > 0) {
            uint32_t spill_bits = batch_packed[word_idx + 1] & ((1u << spill) - 1);
            value |= (spill_bits << (bits - spill));
        }
        
        batch_values[idx] = value;
    }
}

}  // namespace cuda
}  // namespace turboquant


/*
 * PyTorch Binding Functions
 */

at::Tensor pack_lowbit_cuda(
    at::Tensor values,
    int bits
) {
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(values.dtype() == at::kInt, "values must be int32");
    
    int64_t length = values.shape(-1);
    int packed_width = (length * bits + 31) / 32;
    
    // Flatten to 2D: [batch_size, length]
    auto values_flat = values.reshape({-1, length});
    int batch_size = values_flat.shape(0);
    
    auto packed = at::zeros(
        {batch_size, packed_width},
        at::TensorOptions().device(values.device()).dtype(at::kInt)
    );
    
    // Launch kernel
    dim3 blocks(batch_size);
    dim3 threads(256);  // 256 threads per block for good occupancy
    
    turboquant::cuda::pack_lowbit_kernel<<<blocks, threads>>>(
        (uint32_t*)values_flat.data_ptr(),
        (uint32_t*)packed.data_ptr(),
        length,
        bits,
        packed_width,
        batch_size
    );
    
    // Reshape back to original batch shape
    auto output_shape = values.shape().vec();
    output_shape[output_shape.size() - 1] = packed_width;
    return packed.reshape(output_shape);
}


at::Tensor unpack_lowbit_cuda(
    at::Tensor packed,
    int bits,
    int length
) {
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(packed.dtype() == at::kInt, "packed must be int32");
    
    // Flatten to 2D: [batch_size, packed_width]
    auto packed_flat = packed.reshape({-1, packed.shape(-1)});
    int batch_size = packed_flat.shape(0);
    int packed_width = packed_flat.shape(1);
    
    auto values = at::zeros(
        {batch_size, length},
        at::TensorOptions().device(packed.device()).dtype(at::kInt)
    );
    
    // Launch kernel
    dim3 blocks(batch_size);
    dim3 threads(256);  // 256 threads per block for good occupancy
    
    turboquant::cuda::unpack_lowbit_kernel<<<blocks, threads>>>(
        (uint32_t*)packed_flat.data_ptr(),
        (uint32_t*)values.data_ptr(),
        length,
        bits,
        packed_width,
        batch_size
    );
    
    // Reshape back to original batch shape
    auto output_shape = packed.shape().vec();
    output_shape[output_shape.size() - 1] = length;
    return values.reshape(output_shape);
}
