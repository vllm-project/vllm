from vllm import quantization_ops

def awq_linear(input, qweight, scales, qzeros, bias=None, w_bit=4):
    pack_factor = 32 // w_bit
    out_shape = (input.shape[-2], qweight.shape[-1] * pack_factor)
    out = quantization_ops.gemm_forward_cuda(
        input.reshape(-1, input.shape[-1]),
        qweight,
        scales,
        qzeros,
        pack_factor)
    out = out + bias if bias is not None else out
    return out.reshape(out_shape)