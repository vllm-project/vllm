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
        wb = wb_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)

        tmp = xi @ wa
        y_stage_1[i] = tmp.squeeze(0)
        y_final[i] += (tmp @ wb).squeeze(0) * s
    return y_final, y_stage_1


H1 = H2 = [
    128, 256, 512, 1024, 1280, 2048, 2560, 2752, 3072, 3456, 3584, 4096, 5120,
    5504, 5632, 6912, 7168, 8192, 9216, 10240, 11008, 13824, 14336, 32000,
    32256, 32512, 32768, 33024
]
SEED = [0xabcdabcd987]


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("h1", H1)
@pytest.mark.parametrize("h2", H2)
@pytest.mark.parametrize("seed", SEED)
@torch.inference_mode()
def test_lora_correctness(dtype_str, h1, h2, seed):
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda")

    wa_T_all = torch.randn(num_loras,
                           num_layers,
                           r,
                           h1,
                           dtype=dtype,
                           device=device)
    wb_T_all = torch.randn(num_loras,
                           num_layers,
                           h2,
                           r,
                           dtype=dtype,
                           device=device)
    indices = torch.randint(num_loras, (bs, ), dtype=torch.long, device=device)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, h2, dtype=dtype, device=device)

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
@torch.inference_mode()
def test_lora_correctness_slice(dtype_str, h1, h2, seed):
    if h2 % 3 != 0 or h2 // 3 not in H1:
        pytest.skip("h2 must be divisible by 3 and in supported shapes")
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda")

    wa_T_all_0 = torch.randn(num_loras,
                             num_layers,
                             r,
                             h1,
                             dtype=dtype,
                             device=device)
    wa_T_all_1 = torch.randn(num_loras,
                             num_layers,
                             r,
                             h1,
                             dtype=dtype,
                             device=device)
    wa_T_all_2 = torch.randn(num_loras,
                             num_layers,
                             r,
                             h1,
                             dtype=dtype,
                             device=device)
    wb_T_all_0 = torch.randn(num_loras,
                             num_layers,
                             h2 // 3,
                             r,
                             dtype=dtype,
                             device=device)
    wb_T_all_1 = torch.randn(num_loras,
                             num_layers,
                             h2 // 3,
                             r,
                             dtype=dtype,
                             device=device)
    wb_T_all_2 = torch.randn(num_loras,
                             num_layers,
                             h2 // 3,
                             r,
                             dtype=dtype,
                             device=device)

    indices = torch.randint(num_loras, (bs, ), dtype=torch.long, device=device)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, h2, dtype=dtype, device=device)
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
