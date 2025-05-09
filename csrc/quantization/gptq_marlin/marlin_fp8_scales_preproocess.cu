
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>


__global__ void marlin_fp8_scales_preprocess_kernel(int4* __restrict__ in_ptr,
int4* __restrict__ out_ptr, int64_t s_size) {

    // convert subnormal fp8_e4m3 value to fp8_e5m3_val
    // #0bEEEEEMMM   // subnormal_e4m3_val = e5m3_val
    const uint8_t subnormal_val_map[9] = {
        0b00000000,  // 0 / 2 = 0
        0b00110000,  // 1 / 8 * (2 ** -6) = 1.00 * (2 ** (6 - 15))
        0b00111000,  // 2 / 8 * (2 ** -6) = 1.00 * (2 ** (7 - 15))
        0b00111100,  // 3 / 8 * (2 ** -6) = 1.50 * (2 ** (7 - 15))
        0b01000000,  // 4 / 8 * (2 ** -6) = 1.00 * (2 ** (8 - 15))
        0b01000010,  // 5 / 8 * (2 ** -6) = 1.25 * (2 ** (8 - 15))
        0b01000100,  // 6 / 8 * (2 ** -6) = 1.50 * (2 ** (8 - 15))
        0b01000110,  // 7 / 8 * (2 ** -6) = 1.75 * (2 ** (8 - 15))
    };

    int offset = blockIdx.x * blockDim.x;

    // Note that after the conversion,
    // the first bit of all values (except 0.0) is 1
    auto process_val = [&](uint8_t val) {
        if (val == 0) return 0;

        // normalized value case
        // (x | 0x80): set the top bit of exponent to 1
        //             so that we have less exponent bias with fp16/bf16
        // (x - 8): divide the fp8 value by 2
        //          to avoid the value become NaN after dequantization
        // when x = *reinterpret_cast<uint8*>(&fp8_val)
        // (x - 8 * y) means the exponent is decreased by y,
        // which corresponds to dividing the fp8 value by 2 ** y
        else if (val >= 8) return (val | 0x80) - 8;

        // subnormal value (all exponent bits is 0)
        // (x - 8 * 8): to match the exponent bias used by normalized numbers
        // (x - 8): same with normalized value case
        else return (subnormal_val_map[val] | 0x80) - 8 * (8 + 1);
    };

    for (int i = offset + threadIdx.x; i < s_size / 16; i += blockDim.x) {
        int4 val = in_ptr[i];
        uint8_t* vals = reinterpret_cast<uint8_t*>(&val);

  #pragma unroll
        for (int j = 0; j < 16; j++) vals[j] = process_val(vals[j]);

        out_ptr[i] = *reinterpret_cast<int4*>(vals);
    }
};


torch::Tensor marlin_fp8_scales_preprocess(torch::Tensor scales) {
    TORCH_CHECK(scales.device().is_cuda(), "scales is not on GPU");

    int dev = scales.get_device();
    torch::Tensor out_scales = torch::empty_like(scales);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
    marlin_fp8_scales_preprocess_kernel<<<256, 512, 0, stream>>>(
        reinterpret_cast<int4*>(scales.data_ptr()),
        reinterpret_cast<int4*>(out_scales.data_ptr()),
        scales.nbytes()
    );

    return out_scales;
}


TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_fp8_scales_preprocess", &marlin_fp8_scales_preprocess);
}
