import torch

import pytest
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_shrink import sgmv_shrink




def ref_torch_groupgemm(
    x_ptr,
    lora_ptr,
    batchs,
    lora_indices_tensor,
    seq_len_tensor,
) -> torch.Tensor:
    out_list = []

    current_offset = 0
    for lora_index, b_length in zip(range(batchs), seq_len_tensor):
        input_weight = x_ptr[current_offset : b_length + current_offset, :]
        current_offset += b_length
        lora_weight = lora_ptr[lora_indices_tensor[lora_index]]
        result = torch.nn.functional.linear(input_weight, lora_weight)
        out_list.append(result)
    out = torch.cat(out_list, dim=0)
    return out


@pytest.mark.parametrize("batchs", [i for i in range(0, 128, 8)])
@pytest.mark.parametrize("hidden_size", [128, 256, 512, 1024, 4096, 8192, 3424])
@pytest.mark.parametrize("lora_nums", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("max_rank", [1, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,torch.float32])
@torch.inference_mode()
def test_shrink_kernel(batchs, hidden_size, lora_nums, max_rank, dtype):
    SEED = [0xABCDABCD987]
    torch.manual_seed(SEED[0])
    if batchs == 0:
        batchs += 1

    seq_len_tensor = torch.randint(1, 1024, (batchs,)).cuda()
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).cuda()
    total_tokens = seq_len_tensor.sum()

    inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).cuda()
    lora_a_weights = torch.rand(
        (lora_nums, max_rank, hidden_size),  # col-major
        dtype=dtype,
    ).cuda()

    lora_indices_tensor = torch.randint(0, lora_nums - 1, (batchs,)).cuda()
    output_tensor = torch.zeros(
        total_tokens, max_rank, dtype=torch.float32
    ).cuda()

    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()

    sgmv_shrink(
        inputs_tensor,
        lora_a_weights,
        output_tensor,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batchs,
        max_seq_length,
    )
    torch.cuda.synchronize()
    torch_out_tensor = ref_torch_groupgemm(
        inputs_tensor,
        lora_a_weights,
        batchs,
        lora_indices_tensor,
        seq_len_tensor,
    )
    torch_out_tensor = torch_out_tensor.to(torch.float32)
    assert torch.allclose(torch_out_tensor, output_tensor, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("batchs", [i for i in range(0, 128, 8)])
@pytest.mark.parametrize("hidden_size", [128, 256, 512, 1024, 4096, 8192, 3424])
@pytest.mark.parametrize("lora_nums", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("max_rank", [1, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,torch.float32])
@torch.inference_mode()
def test_expand_kernel(batchs, hidden_size, lora_nums, max_rank, dtype):
    SEED = [0xABCDABCD987]
    torch.manual_seed(SEED[0])
    if batchs == 0:
        batchs += 1

    seq_len_tensor = torch.randint(1, 1024, (batchs,)).cuda()
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).cuda()
    total_tokens = seq_len_tensor.sum()

    inputs_tensor = torch.rand((total_tokens, max_rank), dtype=dtype).cuda()
    lora_b_weights = torch.rand(
        (lora_nums,hidden_size, max_rank),  # col-major
        dtype=dtype,
    ).cuda()

    lora_indices_tensor = torch.randint(0, lora_nums - 1, (batchs,)).cuda()
    output_tensor = torch.zeros(
        total_tokens, hidden_size, dtype=dtype
    ).cuda()

    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()

    sgmv_expand(
        inputs_tensor,
        lora_b_weights,
        output_tensor,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batchs,
        max_seq_length,
    )
    torch.cuda.synchronize()
    torch_out_tensor = ref_torch_groupgemm(
        inputs_tensor,
        lora_b_weights,
        batchs,
        lora_indices_tensor,
        seq_len_tensor,
    )
    assert torch.allclose(torch_out_tensor, output_tensor, atol=1e-2, rtol=1e-2)
