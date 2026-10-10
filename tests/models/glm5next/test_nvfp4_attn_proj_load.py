# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading ModelOpt NVFP4 MLA projections into GLM-5.x's BF16 fused_qkv_a_proj."""

import pytest
import torch

from vllm.models.glm5next.common.model import _try_load_nvfp4_attn_proj

E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
PREFIX = "model.layers.3.self_attn"


class _RecordingParam:
    def __init__(self):
        self.calls: list = []

    def weight_loader(self, param, weight, shard_id=None):
        self.calls.append((weight, shard_id))


def _nvfp4(out_dim: int, in_dim: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0, 256, (out_dim, in_dim // 2), dtype=torch.uint8, generator=g
    )
    scale = (torch.rand(out_dim, in_dim // 16, generator=g) * 4 + 0.25).to(
        torch.float8_e4m3fn
    )
    gscale = torch.tensor(0.0123, dtype=torch.float32)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(out_dim, -1).long()
    vals = E2M1[codes].view(out_dim, -1, 16)
    ref = (vals * (scale.float() * gscale).unsqueeze(-1)).reshape(out_dim, -1)
    return packed, scale, gscale, ref.to(torch.bfloat16)


def _feed(proj, packed, scale, gscale, params, buf, loaded, pad=0):
    names = [
        (f"{PREFIX}.{proj}.input_scale", torch.tensor(0.25)),
        (f"{PREFIX}.{proj}.weight_scale", scale),
        (f"{PREFIX}.{proj}.weight", packed),
        (f"{PREFIX}.{proj}.weight_scale_2", gscale),
        (f"{PREFIX}.{proj}.input_scale", torch.tensor(0.25)),
    ]
    return [_try_load_nvfp4_attn_proj(n, t, buf, params, loaded, pad) for n, t in names]


def test_nvfp4_q_a_and_kv_a_dequantize_into_bf16_fused_shards():
    fused = _RecordingParam()
    params = {f"{PREFIX}.fused_qkv_a_proj.weight": fused}
    buf: dict = {}
    loaded: set[str] = set()

    q_packed, q_scale, q_gs, q_ref = _nvfp4(64, 128, seed=0)
    kv_packed, kv_scale, kv_gs, kv_ref = _nvfp4(32, 128, seed=1)
    assert all(_feed("q_a_proj", q_packed, q_scale, q_gs, params, buf, loaded, 8))
    assert all(
        _feed("kv_a_proj_with_mqa", kv_packed, kv_scale, kv_gs, params, buf, loaded, 8)
    )

    (q_w, q_shard), (kv_w, kv_shard) = fused.calls
    assert (q_shard, kv_shard) == (0, 1)
    assert q_w.dtype == torch.bfloat16
    torch.testing.assert_close(q_w, q_ref)
    # NoPE models pad the kv_a rope rows with zeros; q_a is not padded.
    torch.testing.assert_close(kv_w[:32], kv_ref)
    assert kv_w.shape == (40, 128) and not kv_w[32:].any()
    assert loaded == {f"{PREFIX}.fused_qkv_a_proj.weight"}


@pytest.mark.parametrize("proj", ["q_b_proj", "o_proj"])
def test_nvfp4_bf16_projection_ignores_input_scale(proj):
    param = _RecordingParam()
    target = f"{PREFIX}.{proj}.weight"
    params = {target: param}
    buf: dict = {}
    loaded: set[str] = set()
    packed, scale, gscale, ref = _nvfp4(64, 128, seed=3)

    assert all(_feed(proj, packed, scale, gscale, params, buf, loaded))
    ((weight, shard_id),) = param.calls
    torch.testing.assert_close(weight, ref)
    assert shard_id is None
    assert loaded == {target}
    assert not any(buf.values())


def test_nvfp4_projection_with_quantized_target_uses_normal_path():
    params = {
        f"{PREFIX}.q_b_proj.weight": _RecordingParam(),
        f"{PREFIX}.q_b_proj.weight_scale": _RecordingParam(),
    }
    packed, scale, gscale, _ = _nvfp4(64, 128, seed=2)
    assert not any(_feed("q_b_proj", packed, scale, gscale, params, {}, set()))


def test_registered_input_scale_uses_normal_path():
    name = f"{PREFIX}.q_b_proj.input_scale"
    params = {
        f"{PREFIX}.q_b_proj.weight": _RecordingParam(),
        name: _RecordingParam(),
    }
    assert not _try_load_nvfp4_attn_proj(name, torch.tensor(0.25), {}, params, set(), 0)


def test_fp8_and_bf16_tensors_are_left_alone():
    params = {f"{PREFIX}.fused_qkv_a_proj.weight": _RecordingParam()}
    fp8 = torch.zeros(64, 128, dtype=torch.float8_e4m3fn)
    bf16 = torch.zeros(64, 128, dtype=torch.bfloat16)
    for t in (fp8, bf16):
        assert not _try_load_nvfp4_attn_proj(
            f"{PREFIX}.q_a_proj.weight", t, {}, params, set(), 0
        )
