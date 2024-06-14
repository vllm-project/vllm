import pytest
import torch

import vllm._punica_C as punica_kernels
import vllm.lora.punica as punica
from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink

# The current punica kernel supports dimension and adds a dimension of 3424.
HIDDEN_SIZES = [
    128,
    256,
    512,
    1024,
    1152,
    1280,
    1536,
    2048,
    2304,
    2560,
    2752,
    3072,
    3424,
    3456,
    3584,
    4096,
    4608,
    5120,
    5504,
    5632,
    6144,
    6848,
    6912,
    7168,
    8192,
    9216,
    10240,
    11008,
    13824,
    14336,
    15360,
    22016,
    24576,
    27392,
    27648,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]

_BATCH_SIZE_ALIGNMENT = 8

# vllm support batch size
BATCHS = [1, 2, 4] + [_BATCH_SIZE_ALIGNMENT * i for i in range(1, 8)]

NUM_LORA = [1, 4, 8, 16, 32, 64, 128, 256]
DTYPES = [torch.float16, torch.bfloat16]
MAX_RANKS = [1, 4, 8, 16, 32, 64, 128]
SCALES = [0.5]
OP_TYPES = ["shrink", "expand"]
SEED = [0]
CUDA_DEVICES = [f"cuda:{0}"]
NSLICES = [2, 3]


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@torch.inference_mode()
def _punica_bgmv(out_tensor, inputs, lora_weights, indices, scaling):
    layer_idx = 0
    punica.bgmv(out_tensor, inputs, lora_weights, indices, layer_idx, scaling)
    return


def _torch_groupgemm(
    out_tensor,
    inputs,
    lora_weights,
    lora_indices_tensor,
    seq_len_tensor,
    batchs,
    scaling,
    op_type,
) -> torch.Tensor:
    out_list = []
    current_offset = 0
    for lora_index, b_length in zip(range(batchs), seq_len_tensor):
        input_weight = inputs[current_offset:b_length + current_offset, :]
        current_offset += b_length
        lora_weight = lora_weights[lora_indices_tensor[lora_index]]
        result = torch.nn.functional.linear(input_weight, lora_weight)
        result *= scaling
        out_list.append(result)
    cat_result = torch.cat(out_list, dim=0)
    if op_type == "expand":
        out_tensor += cat_result
    else:
        out_tensor.copy_(cat_result)
    return


def _generate_data(batchs, hidden_size, lora_nums, max_rank, max_length, dtype,
                   op_type, device):
    if max_length == 1:
        max_length += 1
    seq_len_tensor = torch.randint(1, max_length, (batchs, )).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    if op_type == "shrink":
        inputs_tensor = torch.rand((total_tokens, hidden_size),
                                   dtype=dtype).to(device)
        lora_weights = torch.rand(
            (lora_nums, max_rank, hidden_size),  # col-major
            dtype=dtype,
        ).to(device)
        # shrink op need atomic_add, so output is initinized by 0
        ref_out_tensor = torch.zeros((total_tokens, max_rank),
                                     dtype=dtype,
                                     device=inputs_tensor.device)
        # NOTE  shrink kernel using torch.float32 as output type
        our_out_tensor = torch.zeros(
            (total_tokens, max_rank),
            dtype=torch.float32,
            device=inputs_tensor.device,
        )
    else:
        inputs_tensor = torch.rand(
            (total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        lora_weights = torch.rand(
            (lora_nums, hidden_size, max_rank),  # col-major
            dtype=dtype,
        ).to(device)
        # expand op needs to complete y+=a@lora_b, so output is
        # initinized randomly
        ref_out_tensor = torch.rand(
            (total_tokens, hidden_size),
            dtype=dtype,
            device=inputs_tensor.device,
        )
        # Ensure the same input.
        our_out_tensor = ref_out_tensor.clone()

    lora_indices_tensor = torch.randint(0,
                                        lora_nums - 1 if lora_nums > 1 else 1,
                                        (batchs, )).to(device)
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batchs):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset +
                seq_len_tensor[b_id]] = lora_index.item()
        current_offset += seq_len_tensor[b_id].item()
    return (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def _generate_data_expand_nslices(batchs, hidden_size, lora_nums, max_rank,
                                  max_length, dtype, nslices, device):
    if max_length == 1:
        max_length += 1
    seq_len_tensor = torch.randint(1, max_length, (batchs, )).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()

    inputs_tensor = torch.rand(
        (total_tokens, max_rank),
        dtype=dtype,
    ).to(device)
    lora_weights_lst = []
    for _ in range(nslices):
        lora_weights_lst.append(
            torch.rand(
                (lora_nums, hidden_size, max_rank),  # col-major
                dtype=dtype,
            ).to(device))
    # expand op needs to complete y+=a@lora_b, so output is
    # initinized randomly
    ref_out_tensor = torch.rand(
        (total_tokens, hidden_size * nslices),
        dtype=dtype,
        device=inputs_tensor.device,
    )
    # Ensure the same input.
    our_out_tensor = ref_out_tensor.clone()

    lora_indices_tensor = torch.randint(0,
                                        lora_nums - 1 if lora_nums > 1 else 1,
                                        (batchs, )).to(device)
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batchs):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset +
                seq_len_tensor[b_id]] = lora_index.item()
        current_offset += seq_len_tensor[b_id].item()
    return (
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


# @pytest.mark.parametrize("batchs", BATCHS)
# @pytest.mark.parametrize("num_loras", NUM_LORA)
# @pytest.mark.parametrize("rank", MAX_RANKS)
# @pytest.mark.parametrize("scaling", SCALES)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("op_type", OP_TYPES)
# @pytest.mark.parametrize("seed", SEED)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# def test_sgmv_torch(
#     batchs: int,
#     num_loras: int,
#     rank: int,
#     scaling: float,
#     dtype: torch.dtype,
#     op_type: str,
#     seed: int,
#     device: str,
# ):
#     torch.manual_seed(seed)
#     torch.set_default_device(device)
#     if batchs == 0:
#         batchs += 1
#     hidden_size_index = random.randint(0, len(HIDDEN_SIZES) - 1)
#     hidden_size = HIDDEN_SIZES[hidden_size_index]
#     if hidden_size > 100000:
#         hidden_size = hidden_size // 4  # avoid OOM
#     (
#         inputs_tensor,
#         lora_weights,
#         our_out_tensor,
#         ref_out_tensor,
#         b_seq_start_loc,
#         lora_indices_tensor,
#         seq_len_tensor,
#         indices,
#     ) = _generate_data(
#         batchs, hidden_size, num_loras, rank, 1024, dtype, op_type, device
#     )  # The sequence length is restricted to the range [1, 1024].
#     max_seq_length = seq_len_tensor.max()
#     if isinstance(max_seq_length, tuple):
#         max_seq_length = max_seq_length[0].item()
#     else:
#         max_seq_length = max_seq_length.item()
#     if op_type == "shrink":
#         sgmv_shrink(
#             inputs_tensor,
#             lora_weights,
#             our_out_tensor,
#             b_seq_start_loc,
#             seq_len_tensor,
#             lora_indices_tensor,
#             batchs,
#             max_seq_length,
#             scaling,
#         )
#     else:
#         sgmv_expand(
#             inputs_tensor,
#             lora_weights,
#             our_out_tensor,
#             b_seq_start_loc,
#             seq_len_tensor,
#             lora_indices_tensor,
#             batchs,
#             max_seq_length,
#             add_inputs=True,
#         )
#     _torch_groupgemm(
#         ref_out_tensor,
#         inputs_tensor,
#         lora_weights,
#         lora_indices_tensor,
#         seq_len_tensor,
#         batchs,
#         scaling if op_type == "shrink" else 1.0,
#         op_type,
#     )
#     if op_type == "shrink":
#         ref_out_tensor = ref_out_tensor.to(torch.float32)
#     assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.skip("stop")
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", OP_TYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_triton_sgmv_punica_bgmv(
    hidden_size,
    scaling: float,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    # avoid `No suitable kernel. h_in=xx h_out=xxxx ` error
    if dtype == torch.float32 or hidden_size == 3424:
        return
    torch.manual_seed(seed)
    torch.set_default_device(device)
    batchs = 4  # Arbitrary values for testing
    rank = 16  # Arbitrary values for testing
    seq_len = 128  # Arbitrary values for testing
    num_loras = 8  # Arbitrary values for testing
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = _generate_data(batchs, hidden_size, num_loras, rank, seq_len, dtype,
                       op_type, device)

    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    if op_type == "shrink":
        sgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batchs,
            max_seq_length,
            scaling,
        )
    else:
        sgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batchs,
            max_seq_length,
            add_inputs=True,
        )
    lora_weights_4d = lora_weights.unsqueeze(dim=1)
    _punica_bgmv(
        ref_out_tensor,
        inputs_tensor,
        lora_weights_4d,
        indices,
        scaling if op_type == "shrink" else 1.0,
    )
    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.parametrize("batchs", BATCHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", OP_TYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_triton_bgmv_punica_bgmv(
    batchs: int,
    hidden_size: int,
    scaling: float,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    # avoid `No suitable kernel. h_in=xx h_out=xxxx ` error
    if dtype == torch.float32 or hidden_size == 3424:
        return
    torch.manual_seed(seed)
    torch.set_default_device(device)
    if batchs == 0:
        batchs += 1
    rank = 16
    seq_len = 1  #
    num_loras = 8  # Arbitrary values for testing
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = _generate_data(batchs, hidden_size, num_loras, rank, seq_len, dtype,
                       op_type, device)

    if op_type == "shrink":
        bgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            lora_indices_tensor,
            scaling,
        )
    else:
        bgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            lora_indices_tensor,
            add_inputs=True,
        )
    lora_weights_4d = lora_weights.unsqueeze(dim=1)
    _punica_bgmv(
        ref_out_tensor,
        inputs_tensor,
        lora_weights_4d,
        indices,
        scaling if op_type == "shrink" else 1.0,
    )
    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.skip("stop")
@pytest.mark.parametrize("batchs", BATCHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", NSLICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sgmv_expand_slice(
    batchs: int,
    hidden_size: int,
    nslices: int,
    dtype: str,
    seed: int,
    device: str,
):
    # avoid `No suitable kernel. h_in=xx h_out=xxxx ` error
    if dtype == torch.float32 or hidden_size == 3424:
        return
    torch.manual_seed(seed)
    torch.set_default_device(device)
    max_rank = 16
    lora_nums = 4
    max_length = 128
    (
        inputs_tensor,
        lora_weights_lst,
        our_outputs,
        ref_outputs,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = _generate_data_expand_nslices(
        batchs,
        hidden_size,
        lora_nums,
        max_rank,
        max_length,
        dtype,
        nslices,
        device,
    )
    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    slice_offset = 0
    for index in range(nslices):
        lora_weights = lora_weights_lst[index]
        sgmv_expand_slice(
            inputs_tensor,
            lora_weights,
            our_outputs,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batchs,
            max_seq_length,
            slice_offset,
            hidden_size,
            add_inputs=True,
        )
        lora_weights_4d = lora_weights.unsqueeze(dim=1)
        punica_kernels.dispatch_bgmv_low_level(
            ref_outputs,
            inputs_tensor,
            lora_weights_4d,
            indices,
            0,
            1.0,
            inputs_tensor.size(1),
            hidden_size,
            slice_offset,
        )
        slice_offset += hidden_size
    assert_close(our_outputs, ref_outputs)


@pytest.mark.skip("stop")
@pytest.mark.parametrize("batchs", BATCHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", NSLICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_bgmv_expand_slice(
    batchs: int,
    hidden_size: int,
    nslices: int,
    dtype: str,
    seed: int,
    device: str,
):
    # avoid `No suitable kernel. h_in=xx h_out=xxxx ` error
    if dtype == torch.float32 or hidden_size == 3424:
        return
    torch.manual_seed(seed)
    torch.set_default_device(device)
    max_rank = 64
    lora_nums = 8
    (
        inputs_tensor,
        lora_weights_lst,
        our_outputs,
        ref_outputs,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = _generate_data_expand_nslices(
        batchs,
        hidden_size,
        lora_nums,
        max_rank,
        1,
        dtype,
        nslices,
        device,
    )
    slice_offset = 0
    for index in range(nslices):
        lora_weights = lora_weights_lst[index]
        bgmv_expand_slice(
            inputs_tensor,
            lora_weights,
            our_outputs,
            lora_indices_tensor,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=True,
        )
        lora_weights_4d = lora_weights.unsqueeze(dim=1)
        punica_kernels.dispatch_bgmv_low_level(
            ref_outputs,
            inputs_tensor,
            lora_weights_4d,
            lora_indices_tensor,
            0,
            1.0,
            inputs_tensor.size(1),
            hidden_size,
            slice_offset,
        )
        slice_offset += hidden_size
    assert_close(our_outputs, ref_outputs)


if __name__ == "__main__":
    test_bgmv_expand_slice(
        batchs=32,
        hidden_size=128,
        nslices=2,
        dtype=torch.bfloat16,
        seed=0,
        device="cuda:0",
    )
