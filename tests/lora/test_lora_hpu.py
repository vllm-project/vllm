import pytest
import torch
from vllm_hpu_extension.ops import LoraMask

from vllm.lora.punica_wrapper.punica_hpu import PunicaWrapperHPU

from .utils import DummyLoRAManager

TENSOR_SIZES = [128, 1024, 2048, 4096, 8192, 11008, 11008 // 2, 11008 // 4]
QKV_TENSOR_SIZES = [
    (8192, 1024, 1024),
    (8192 // 8, 1024 // 8, 1024 // 8),
    (4096, 4096, 4096),
    (4096 // 2, 4096 // 2, 4096 // 2),
]
BATCH_SIZES = [8, 32, 256]
RANKS = [8]
DTYPES = [torch.bfloat16]
TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}


def createLoraMask(indices, batch_size, seq_len, max_loras, max_lora_rank,
                   lora_dtype):
    indices = indices.view(-1, 1)
    mask = torch.arange(max_loras * max_lora_rank, device=indices.device)
    mask = mask.view(1, -1)
    mask = ((mask >= ((indices) * max_lora_rank)) *
            (mask < ((indices + 1) * max_lora_rank))).to(dtype=lora_dtype)
    mask = mask.view(batch_size, 1,
                     -1).expand(batch_size, seq_len,
                                -1).reshape(batch_size * seq_len, -1)
    return mask


@pytest.mark.parametrize("m", TENSOR_SIZES)
@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("k", BATCH_SIZES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_lora(m, n, k, rank, dtype) -> None:
    manager = DummyLoRAManager(device="hpu")

    module_name = "module"
    weight = torch.rand([m, n], device="hpu", dtype=dtype)

    manager.init_random_lora(module_name, weight, rank=rank)
    lora = manager.get_module_lora(module_name)

    input = torch.rand(k, n, device="hpu", dtype=dtype)
    expected = input @ lora.lora_a @ lora.lora_b * lora.scaling

    lora_a_stack = [
        torch.zeros(8,
                    1,
                    lora.lora_a.shape[1],
                    lora.lora_a.shape[0],
                    device="hpu",
                    dtype=dtype)
    ]
    lora_b_stack = [
        torch.zeros(8,
                    1,
                    lora.lora_b.shape[1],
                    lora.lora_b.shape[0],
                    device="hpu",
                    dtype=dtype)
    ]
    for i in range(lora_a_stack[0].shape[0]):
        lora_a_stack[0][i][0] = lora.lora_a.T
        lora_b_stack[0][i][0] = (lora.lora_b * lora.scaling).T

    output = torch.zeros(k, m, device="hpu", dtype=dtype)
    indices = torch.randint(0,
                            lora_a_stack[0].shape[0], (len(input), ),
                            device="hpu")
    mask = createLoraMask(indices, k, 1, 8, rank, dtype)
    LoraMask.setLoraMask(mask)
    punica_wrapper = PunicaWrapperHPU(4096, max_batches=256, device="hpu")

    lora_bias_stacked = None
    output_slices = (lora_b_stack[0].shape[2], )
    punica_wrapper.add_lora_linear(output, input, lora_a_stack, lora_b_stack,
                                   lora_bias_stacked, 1.0, output_slices)

    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(expected, output, rtol=rtol, atol=atol)

    output[:] = 0
    indices = torch.full((len(input), ), -1, device="hpu")
    mask = createLoraMask(indices, k, 1, 8, rank, dtype)
    LoraMask.setLoraMask(mask)

    punica_wrapper.add_lora_linear(output, input, lora_a_stack, lora_b_stack,
                                   lora_bias_stacked, 1.0, output_slices)
    assert torch.allclose(torch.zeros_like(output), output)

    manager.reset_lora()


@pytest.mark.parametrize("m", TENSOR_SIZES)
@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("k", BATCH_SIZES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_lora_packed_2slice(m, n, k, rank, dtype) -> None:
    if m % 2 != 0:
        pytest.skip("m must be divisible by 2")
    if m // 2 not in TENSOR_SIZES:
        pytest.skip("m//2 must be in TENSOR_SIZES")

    manager = DummyLoRAManager(device="hpu")

    module_name = "module"
    weight = torch.rand([m // 2, n], device="hpu", dtype=dtype)

    manager.init_random_lora(module_name + "1", weight, rank=rank)
    lora_1 = manager.get_module_lora(module_name + "1")
    manager.init_random_lora(module_name + "2", weight, rank=rank)
    lora_2 = manager.get_module_lora(module_name + "2")

    input = torch.rand(k, n, device="hpu", dtype=dtype)
    expected = torch.cat([
        input @ lora_1.lora_a @ lora_1.lora_b * lora_1.scaling,
        input @ lora_2.lora_a @ lora_2.lora_b * lora_2.scaling
    ],
                         dim=1)

    lora_a_stacks = [
        torch.zeros(8,
                    1,
                    lora_1.lora_a.shape[1],
                    lora_1.lora_a.shape[0],
                    device="hpu",
                    dtype=dtype) for i in range(2)
    ]
    lora_b_stacks = [
        torch.zeros(8,
                    1,
                    lora_1.lora_b.shape[1],
                    lora_1.lora_b.shape[0],
                    device="hpu",
                    dtype=dtype) for i in range(2)
    ]
    for i in range(lora_a_stacks[0].shape[0]):
        lora_a_stacks[0][i][0] = lora_1.lora_a.T
        lora_b_stacks[0][i][0] = (lora_1.lora_b * lora_1.scaling).T
        lora_a_stacks[1][i][0] = lora_2.lora_a.T
        lora_b_stacks[1][i][0] = (lora_2.lora_b * lora_2.scaling).T

    output = torch.zeros(k, m, device="hpu", dtype=dtype)
    indices = torch.randint(0,
                            lora_a_stacks[0].shape[0], (len(input), ),
                            device="hpu")
    mask = createLoraMask(indices, k, 1, 8, rank, dtype)
    LoraMask.setLoraMask(mask)

    lora_bias_stacked = None
    punica_wrapper = PunicaWrapperHPU(4096, max_batches=256, device="hpu")
    punica_wrapper.add_lora_linear(output, input, lora_a_stacks, lora_b_stacks,
                                   lora_bias_stacked, 1.0, (m // 2, m // 2))

    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(expected, output, rtol=rtol, atol=atol)

    output[:] = 0
    indices = torch.full((len(input), ), -1, device="hpu")
    mask = createLoraMask(indices, k, 1, 8, rank, dtype)
    LoraMask.setLoraMask(mask)

    punica_wrapper.add_lora_linear(output, input, lora_a_stacks, lora_b_stacks,
                                   lora_bias_stacked, 1.0, (m // 2, m // 2))
    assert torch.allclose(torch.zeros_like(output), output)

    manager.reset_lora()


@pytest.mark.parametrize("qkv", QKV_TENSOR_SIZES)
@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("k", BATCH_SIZES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_lora_packed_3slice(qkv, n, k, rank, dtype) -> None:
    manager = DummyLoRAManager(device="hpu")

    module_name = "module"
    weight_q = torch.empty(qkv[0], n, device="hpu", dtype=dtype)
    weight_kv = torch.empty(qkv[1], n, device="hpu", dtype=dtype)

    manager.init_random_lora(module_name + "q", weight_q, rank=rank)
    lora_q = manager.get_module_lora(module_name + "q")
    manager.init_random_lora(module_name + "k", weight_kv, rank=rank)
    lora_k = manager.get_module_lora(module_name + "k")
    manager.init_random_lora(module_name + "v", weight_kv, rank=rank)
    lora_v = manager.get_module_lora(module_name + "v")

    input = torch.rand(k, n, device="hpu", dtype=dtype)
    expected = torch.cat([
        input @ lora_q.lora_a @ lora_q.lora_b * lora_q.scaling,
        input @ lora_k.lora_a @ lora_k.lora_b * lora_k.scaling,
        input @ lora_v.lora_a @ lora_v.lora_b * lora_v.scaling
    ],
                         dim=1)

    lora_a_stacks = [
        torch.zeros(8,
                    1,
                    lora_q.lora_a.shape[1],
                    lora_q.lora_a.shape[0],
                    device="hpu",
                    dtype=dtype)
    ] + [
        torch.zeros(8,
                    1,
                    lora_k.lora_a.shape[1],
                    lora_k.lora_a.shape[0],
                    device="hpu",
                    dtype=dtype) for i in range(2)
    ]
    lora_b_stacks = [
        torch.zeros(8,
                    1,
                    lora_q.lora_b.shape[1],
                    lora_q.lora_b.shape[0],
                    device="hpu",
                    dtype=dtype)
    ] + [
        torch.zeros(8,
                    1,
                    lora_k.lora_b.shape[1],
                    lora_k.lora_b.shape[0],
                    device="hpu",
                    dtype=dtype) for i in range(2)
    ]
    for i in range(lora_a_stacks[0].shape[0]):
        lora_a_stacks[0][i][0] = lora_q.lora_a.T
        lora_b_stacks[0][i][0] = (lora_q.lora_b * lora_q.scaling).T
        lora_a_stacks[1][i][0] = lora_k.lora_a.T
        lora_b_stacks[1][i][0] = (lora_k.lora_b * lora_k.scaling).T
        lora_a_stacks[2][i][0] = lora_v.lora_a.T
        lora_b_stacks[2][i][0] = (lora_v.lora_b * lora_v.scaling).T

    output = torch.zeros(k, sum(qkv), device="hpu", dtype=dtype)
    indices = torch.randint(0,
                            lora_a_stacks[0].shape[0], (len(input), ),
                            device="hpu")
    mask = createLoraMask(indices, k, 1, 8, rank, dtype)
    LoraMask.setLoraMask(mask)

    lora_bias_stacked = None
    punica_wrapper = PunicaWrapperHPU(4096, max_batches=256, device="hpu")
    qkvs = (qkv[0], qkv[1], qkv[2])
    punica_wrapper.add_lora_linear(output, input, lora_a_stacks, lora_b_stacks,
                                   lora_bias_stacked, 1.0, qkvs)

    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(expected, output, rtol=rtol, atol=atol)

    output[:] = 0
    indices = torch.full((len(input), ), -1, device="hpu")
    mask = createLoraMask(indices, k, 1, 8, rank, dtype)
    LoraMask.setLoraMask(mask)
    qkvs = (qkv[0], qkv[1], qkv[2])
    punica_wrapper.add_lora_linear(output, input, lora_a_stacks, lora_b_stacks,
                                   lora_bias_stacked, 1.0, qkvs)
    assert torch.allclose(torch.zeros_like(output), output)

    manager.reset_lora()
