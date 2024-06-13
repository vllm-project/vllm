# Based on code from https://github.com/punica-ai/punica

import pytest
import torch

import vllm.lora.punica as punica


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
        torch.float32: (None, None),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def _lora_ref_impl(
    y_final: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    y_stage_1 = torch.empty(
        (x.size(0), wa_T_all.size(-2)),
        dtype=torch.float32,
        device=x.device,
    )
    bs = x.shape[0]
    s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    for i, lora_idx in zip(range(bs), indicies.cpu().tolist()):
        xi = x[i].unsqueeze(0).to(torch.float32)
        wa = wa_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)
        if wb_T_all is not None:
            wb = wb_T_all[lora_idx, layer_idx].transpose(-1,
                                                         -2).to(torch.float32)

        tmp = xi @ wa
        y_stage_1[i] = tmp.squeeze(0)
        y_final[i] += ((tmp @ wb).squeeze(0) *
                       s if wb_T_all is not None else y_stage_1[i])
    return y_final, y_stage_1


H1 = H2 = [
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
    3328,
    3456,
    3584,
    4096,
    4608,
    5120,
    5504,
    5632,
    6144,
    6400,
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
H2 = [64] + H2
R = [1, 2, 4]
SEED = [0xabcdabcd987]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("h1", H1)
@pytest.mark.parametrize("r", R)
@pytest.mark.parametrize("seed", SEED)
@torch.inference_mode()
def test_lora_a_extra_shapes(dtype_str, h1, r, seed):
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    bs = 32
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda")

    wa_T_all = torch.randn(num_loras,
                           num_layers,
                           r,
                           h1,
                           dtype=dtype,
                           device=device)
    indices = torch.randint(num_loras, (bs, ), dtype=torch.long, device=device)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, r, dtype=dtype, device=device)

        y_ref = y.clone()
        _lora_ref_impl(
            y_ref,
            x,
            wa_T_all,
            None,
            indices,
            layer_idx,
            1.0,
        )

        y_our = y.clone()
        punica.bgmv(y_our, x, wa_T_all, indices, layer_idx, 1.0)

        assert_close(y_ref, y_our)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("h1", H1)
@pytest.mark.parametrize("h2", H2)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_lora_correctness(dtype_str, h1, h2, seed, device):
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    torch.set_default_device(device)

    wa_T_all = torch.randn(num_loras, num_layers, r, h1, dtype=dtype)
    wb_T_all = torch.randn(num_loras, num_layers, h2, r, dtype=dtype)
    indices = torch.randint(num_loras, (bs, ), dtype=torch.long)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype)
        y = torch.randn(bs, h2, dtype=dtype)

        y_ref = y.clone()
        _lora_ref_impl(y_ref, x, wa_T_all, wb_T_all, indices, layer_idx, scale)

        y_our = y.clone()
        punica.add_lora(y_our, x, wa_T_all, wb_T_all, indices, layer_idx,
                        scale)

        assert_close(y_ref, y_our)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("h1", H1)
@pytest.mark.parametrize("h2", H2)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_lora_correctness_slice(dtype_str, h1, h2, seed, device):
    if h2 % 3 != 0 or h2 // 3 not in H1:
        pytest.skip("h2 must be divisible by 3 and in supported shapes")
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    torch.set_default_device(device)

    wa_T_all_0 = torch.randn(num_loras, num_layers, r, h1, dtype=dtype)
    wa_T_all_1 = torch.randn(num_loras, num_layers, r, h1, dtype=dtype)
    wa_T_all_2 = torch.randn(num_loras, num_layers, r, h1, dtype=dtype)
    wb_T_all_0 = torch.randn(num_loras, num_layers, h2 // 3, r, dtype=dtype)
    wb_T_all_1 = torch.randn(num_loras, num_layers, h2 // 3, r, dtype=dtype)
    wb_T_all_2 = torch.randn(num_loras, num_layers, h2 // 3, r, dtype=dtype)

    indices = torch.randint(num_loras, (bs, ), dtype=torch.long)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype)
        y = torch.randn(bs, h2, dtype=dtype)
        s = h2 // 3

        y_ref = y.clone()
        _lora_ref_impl(y_ref[:, :s], x, wa_T_all_0, wb_T_all_0, indices,
                       layer_idx, scale)
        _lora_ref_impl(y_ref[:, s:s * 2], x, wa_T_all_1, wb_T_all_1, indices,
                       layer_idx, scale)
        _lora_ref_impl(y_ref[:, s * 2:], x, wa_T_all_2, wb_T_all_2, indices,
                       layer_idx, scale)

        y_our = y.clone()
        punica.add_lora_slice(y_our, x, wa_T_all_0, wb_T_all_0, indices,
                              layer_idx, scale, 0, s)
        punica.add_lora_slice(y_our, x, wa_T_all_1, wb_T_all_1, indices,
                              layer_idx, scale, s, s)
        punica.add_lora_slice(y_our, x, wa_T_all_2, wb_T_all_2, indices,
                              layer_idx, scale, s * 2, s)

        assert_close(y_ref[:, :s], y_our[:, :s])
        assert_close(y_ref[:, s:s * 2], y_our[:, s:s * 2])
        assert_close(y_ref[:, s * 2:], y_our[:, s * 2:])
