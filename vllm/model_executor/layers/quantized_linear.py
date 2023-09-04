from vllm import quantization_ops


def awq_linear(input_, qweight, scales, qzeros, bias=None, pack_factor=8):
    out_shape = (input_.shape[-2], qweight.shape[-1] * pack_factor)
    out = quantization_ops.gemm_forward_cuda(
        input_.reshape(-1, input_.shape[-1]), qweight, scales, qzeros,
        pack_factor)
    out = out + bias if bias is not None else out
    return out.reshape(out_shape)
