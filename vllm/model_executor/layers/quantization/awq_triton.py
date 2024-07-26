import torch

import triton
import triton.language as tl


@triton.jit
def awq_dequantize_kernel(qweight_ptr,   # quantized matrix
                          scales_ptr,    # scales, per group
                          zeros_ptr,     # zeros, per group
                          split_k_iters, # Not used
                          thx,           # Not used
                          thy,           # Not used
                          group_size,    # Should always be 128
                          result_ptr,    # Output matrix
                          num_cols,      # input num cols in qweight
                          num_rows,      # input num rows in qweight
                          reverse_awq_order_ptr,
                          BLOCK_SIZE_X: tl.constexpr,
                          BLOCK_SIZE_Y: tl.constexpr):
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    reverse_awq_order_offsets = tl.arange(0, 8)
    reverse_awq_order_tensor =  tl.load(reverse_awq_order_ptr +
            reverse_awq_order_offsets)

    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    offsets = num_cols  * offsets_y[:, None] + offsets_x[None, :]

    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = 8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols 

    masks = masks_y[:, None] & masks_x[None, :]

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    iweights = tl.load(qweight_ptr + offsets, masks)

    iweights = iweights.reshape(BLOCK_SIZE_Y * BLOCK_SIZE_X, 8,
                                can_reorder = True)

    shifts = reverse_awq_order_tensor * 4#tl.arange(0, 8)  * 4
    shifts = shifts[None, :].broadcast_to(BLOCK_SIZE_Y * BLOCK_SIZE_X, 8)

    iweights = (iweights >> shifts) & 0xF

    iweights = iweights.reshape(BLOCK_SIZE_Y, BLOCK_SIZE_X * 8,
                                can_reorder = True)

    zero_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size
                      + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]
    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    zeros = zeros.reshape(BLOCK_SIZE_Y * BLOCK_SIZE_X, 8, can_reorder = True)
    zeros = zeros >> shifts & 0xF
    zeros = zeros.reshape(BLOCK_SIZE_Y, BLOCK_SIZE_X * 8, can_reorder = True)

    scale_offsets_y  = (pid_y * BLOCK_SIZE_Y // group_size
                       + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8
                        + tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
                    scale_offsets_x[None, :])
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    scales = tl.load(scales_ptr + scale_offsets, scale_masks)

    iweights = iweights.to(tl.float16)
    tl.store(result_ptr + result_offsets, iweights, masks)

# Example input: 
#   qweight.size=torch.Size([3584, 576]),
#   qweight.dtype = torch.int32,
#   scales.size=torch.Size([28, 4608]),
#   scales.dtype=torch.float16,
#   zeros.size=torch.Size([28, 576]),
#   zeros.dtype=torch.int32
#   split_k_iters=0
#   thx=0
#   thy=0
def awq_dequantize_triton(qweight: torch.Tensor,
                         scales: torch.Tensor,
                         zeros: torch.Tensor,
                         split_k_iters: int,
                         thx: int,
                         thy: int) -> torch.Tensor:
    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device = qweight.device,
                         dtype = torch.float16)

    reverse_awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7],
            dtype = torch.uint8, device = qweight.device)

    Y = qweight.shape[0] # num rows
    X = qweight.shape[1] # num cols
    group_size = 128
    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']), triton.cdiv(Y,
                    META['BLOCK_SIZE_Y']), )
    awq_dequantize_kernel[grid](qweight, scales, zeros, split_k_iters, 
            thx, thy, group_size, result, X, Y, reverse_awq_order,
            BLOCK_SIZE_X = 32, BLOCK_SIZE_Y = 64)

    return result


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t 


# qweightss [R     , C // 8], int32
# scales  - [R // G, C     ], float16
# zeros   - [R // G, C // 8], int32
def awq_dequantize_torch(qweight: torch.Tensor,
                         scales: torch.Tensor,
                         qzeros: torch.Tensor,
                         split_k_iters: int,
                         thx: int,
                         thy: int) -> torch.Tensor:
    print(f"awq_dequantize_torch:qweight.shape = {qweight.shape}"
          f", qzeros.shape={qzeros.shape}")
    bits = 4
    group_size = 128
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    
    iweights = torch.bitwise_right_shift(
        qweight[:, :, None],
        shifts[None, None, :]).to(torch.int8)

    iweights = iweights.view(iweights.shape[0], -1)

    iweights = reverse_awq_order(iweights)
    return (iweights & 0xF).to(torch.float16)

    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)

    zeros = zeros.view(qzeros.shape[0], -1)

    zeros = reverse_awq_order(zeros)
    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)


    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    print(f"awq_dequantize_torch:iweights.shape = {iweights.shape},"
          f"zeros.shape={zeros.shape}, "
          f"scales.shape={scales.shape}")

    return iweights.to(torch.float16)
    # return (iweights - zeros) * scales

def main():
    use_triton = True 
    use_torch = True

    qweight_rows = 3584
    qweight_cols = 576
    group_size = 128
    small_test_size = True
    if small_test_size:
        qweight_rows = 256
        qweight_cols = 128
    print(f"qweight_rows = {qweight_rows}, qweight_cols = {qweight_cols}")
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32
    split_k_iters=0
    thx=0
    thy=0
    device='cuda'
    torch.manual_seed(0)

    qweight = torch.randint(0,10000000, (qweight_rows,
                         qweight_cols),
                         dtype=qweight_dtype,
                         device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0, 10000000, (zeros_rows,
                          zeros_cols),
                          dtype=zeros_dtype,
                          device=device)
    print(f"zeros.shape = {zeros.shape}")
    print(f"qweight = {qweight}")
    if use_triton:
      iweights_triton = awq_dequantize_triton(
        qweight, scales, zeros, split_k_iters, thx, thy)
      print(f"Triton result:iweights_triton = {iweights_triton}")
      print(f"Any infs in triton result? -->"
            f"{not torch.all(False == torch.isinf(iweights_triton))}")

    if use_torch:
      iweights_torch = awq_dequantize_torch(
        qweight, scales, zeros, split_k_iters, thx, thy)
      print(f"Torch result:iweights_torch = {iweights_torch}")

    if use_torch and use_triton:
        diff = iweights_torch - iweights_triton
        error = torch.sum(torch.sqrt(diff * diff))
        print(f"error = {error}")

if __name__ == '__main__':
    main()
