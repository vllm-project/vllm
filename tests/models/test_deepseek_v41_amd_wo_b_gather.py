# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which steps of the ROCm DeepSeek V4.1 attention gather wo_b's input, and
that the gathered rows give the output the all-reduce gives."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

rocm = pytest.importorskip("vllm.models.deepseek_v41.amd.rocm")

from vllm.config import VllmConfig  # noqa: E402
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    _mxfp8_e4m3_quantize_torch,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    is_layer_skipped,
    kMxfp8Dynamic,
)

TP = 2
K_LOCAL = 64
HIDDEN = 16


def _dequant(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    blocks = data.to(torch.float32).view(data.shape[0], -1, 32)
    return (blocks * torch.exp2(scale.to(torch.float32) - 127.0)[..., None]).flatten(1)


class _FullWoB(nn.Module):
    """Stands in for the replicated wo_b: a float GEMM on the gathered rows."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.weight = torch.randn(HIDDEN, TP * K_LOCAL, generator=gen)
        self.rows: list[int] = []

    def forward(self, x):
        assert x.quant_key == kMxfp8Dynamic
        assert x.scale.shape == (x.data.shape[0], TP * K_LOCAL // 32)
        self.rows.append(x.data.shape[0])
        return _dequant(x.data, x.scale) @ self.weight.T


class _OtherRank:
    """The all-gather as rank 0 sees it: its rows, then rank 1's."""

    def __init__(self, z: torch.Tensor):
        self.data, self.scale = _mxfp8_e4m3_quantize_torch(z)
        self.start = 0

    def __call__(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == -1 and x.dtype == torch.bfloat16
        rows = slice(self.start, self.start + x.shape[0])
        if x.shape[1] * 2 == K_LOCAL:
            theirs = self.data[rows]
        else:
            # The scales follow the data of the same rows.
            theirs = self.scale[rows]
            self.start += x.shape[0]
        return torch.cat([x, theirs.contiguous().view(torch.bfloat16)], dim=-1)


def _attention(monkeypatch, min_rows: int, piece_rows: int, z_other=None):
    attn = object.__new__(rocm.DeepseekV41ROCMAiterMLAAttention)
    nn.Module.__init__(attn)
    attn._wo_b_scale = None
    attn._wo_b_gather_min_rows = min_rows
    attn.wo_b_full = _FullWoB()
    attn.wo_b = lambda zf: torch.full((zf.shape[0], HIDDEN), -1.0)
    monkeypatch.setattr(
        rocm.AiterCustomAllreduce,
        "effective_max_size",
        classmethod(lambda cls: piece_rows * TP * K_LOCAL),
    )
    monkeypatch.setattr(rocm, "mxfp8_e4m3_quantize", _mxfp8_e4m3_quantize_torch)
    if z_other is not None:
        monkeypatch.setattr(
            rocm, "tensor_model_parallel_all_gather", _OtherRank(z_other)
        )
    return attn


@pytest.mark.parametrize(
    ("num_rows", "pieces"), [(8, [8]), (20, [8, 8, 4]), (16, [8, 8])]
)
def test_gathered_rows_give_the_all_reduced_output(monkeypatch, num_rows, pieces):
    gen = torch.Generator().manual_seed(1)
    z = [torch.randn(num_rows, K_LOCAL, generator=gen) * 8**rank for rank in range(TP)]
    attn = _attention(monkeypatch, min_rows=8, piece_rows=8, z_other=z[1])
    assert attn.wo_b_gathers(num_rows)
    out = attn._wo_b_after_wo_a(z[0])
    assert attn.wo_b_full.rows == pieces
    # What the ranks' slices of wo_b sum to.
    weight = attn.wo_b_full.weight
    expected = sum(
        _dequant(*_mxfp8_e4m3_quantize_torch(z[rank]))
        @ weight[:, rank * K_LOCAL : (rank + 1) * K_LOCAL].T
        for rank in range(TP)
    )
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)


def test_steps_below_the_threshold_keep_the_all_reduce(monkeypatch):
    attn = _attention(monkeypatch, min_rows=8, piece_rows=8)
    zf = torch.randn(7, K_LOCAL)
    assert not attn.wo_b_gathers(7)
    assert torch.equal(attn._wo_b_after_wo_a(zf), torch.full((7, HIDDEN), -1.0))
    assert attn.wo_b_full.rows == []


def test_no_full_copy_means_no_gather(monkeypatch):
    attn = _attention(monkeypatch, min_rows=8, piece_rows=8)
    attn.wo_b_full = None
    assert not attn.wo_b_gathers(1 << 20)
    assert torch.equal(
        attn._wo_b_after_wo_a(torch.randn(64, K_LOCAL)),
        torch.full((64, HIDDEN), -1.0),
    )


def test_a_gathering_step_never_returns_the_rank_partial(monkeypatch):
    # The layer skips its all-reduce whenever wo_b_gathers says so.
    z = torch.randn(8, K_LOCAL)
    attn = _attention(monkeypatch, min_rows=8, piece_rows=8, z_other=z)
    attn._wo_b_scale = torch.ones(1)
    attn._bpre_attn_gemm = lambda *args: pytest.fail("ran the rank's slice")
    attn._wo_b_after_wo_a(z)
    assert attn.wo_b_full.rows == [8]


class _Norm(nn.Identity):
    variance_epsilon = 1e-6


class _FfnOnlyMono:
    """A mono layer of the FFN launch that declines every step."""

    ffn_only = True

    def __init__(self):
        self.parts: list[torch.Tensor] = []

    def __call__(self, *args):
        return None

    def ffn(self, layer, part, *args):
        self.parts.append(part)
        return None


def _layer(monkeypatch, attn):
    """A decoder layer whose seams, norms and FFN pass their input on, around
    `attn` from wo_b on (the layer's input is wo_b's); returns it with its
    mono and the tensors it all-reduced."""
    model = pytest.importorskip("vllm.models.deepseek_v41.amd.model")
    layer = object.__new__(model.DeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    for name in (
        "mhc_overlap",
        "mhc_seam",
        "engram",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_attn_fn_packed",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "hc_ffn_fn_packed",
        "rms_norm_eps",
        "hc_eps",
        "hc_post_alpha",
        "hc_sinkhorn_iters",
    ):
        setattr(layer, name, None)
    layer.fuse_seam_norm = False
    layer.attn_seam_mxfp8 = layer.ffn_seam_mxfp8 = False
    layer.use_sequence_parallel = False
    layer.mono = _FfnOnlyMono()
    layer.mhc_pre_delayed = lambda residual, *args, sublayer_out, **kwargs: (
        residual,
        None,
        None,
        sublayer_out,
        None,
    )
    layer.attn_norm = _Norm()
    attn.forward = lambda positions, x, _: attn._wo_b_after_wo_a(x)
    layer.attn = attn
    layer.ffn_norm = _Norm()
    layer.ffn = lambda x, input_ids: x
    reduced: list[torch.Tensor] = []

    def all_reduce(x):
        reduced.append(x)
        return x * TP

    monkeypatch.setattr(model, "tensor_model_parallel_all_reduce", all_reduce)
    return layer, layer.mono, reduced


def test_the_layer_does_not_reduce_a_step_that_gathered(monkeypatch):
    z = torch.randn(8, K_LOCAL)
    attn = _attention(monkeypatch, min_rows=8, piece_rows=8, z_other=z)
    layer, mono, reduced = _layer(monkeypatch, attn)
    out = layer(z, torch.arange(8), None, residual=z)[0]
    assert attn.wo_b_full.rows == [8]
    # Neither the FFN launch nor the all-reduce takes the reduced output for
    # a rank's partial.
    assert mono.parts == [] and reduced == []
    z_q = _dequant(*_mxfp8_e4m3_quantize_torch(z))
    expected = torch.cat([z_q, z_q], dim=-1) @ attn.wo_b_full.weight.T
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)


def test_the_layer_reduces_a_step_that_did_not_gather(monkeypatch):
    z = torch.randn(7, K_LOCAL)
    attn = _attention(monkeypatch, min_rows=8, piece_rows=8)
    layer, mono, reduced = _layer(monkeypatch, attn)
    out = layer(z, torch.arange(7), None, residual=z)[0]
    assert attn.wo_b_full.rows == []
    part = torch.full((7, HIDDEN), -1.0)
    # The FFN launch is offered the rank's partial first.
    assert len(mono.parts) == 1 and torch.equal(mono.parts[0], part)
    assert len(reduced) == 1 and torch.equal(reduced[0], part)
    assert torch.equal(out, part * TP)


def _linear(in_features: int, loader=None, names=("weight",)) -> nn.Module:
    linear = nn.Module()
    for name in names:
        param = nn.Parameter(torch.zeros(HIDDEN, in_features), requires_grad=False)
        if loader is not None:
            param.weight_loader = loader
        setattr(linear, name, param)
    return linear


def _load(layers, weights):
    """DeepseekV4Model.load_weights on a model of `layers`' attentions."""
    model = pytest.importorskip("vllm.models.deepseek_v41.amd.model")
    target = nn.Module()
    target.layers = nn.ModuleList()
    for wo_b, wo_b_full in layers:
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.wo_b = wo_b
        layer.attn.wo_b_full = wo_b_full
        target.layers.append(layer)
    target.quant_config = None
    target.use_sequence_parallel = False
    target.get_expert_mapping = lambda: []
    return model.DeepseekV4Model.load_weights(target, weights)


def _rank_slice(param, loaded_weight):
    param.data.copy_(loaded_weight[:, :K_LOCAL])


@pytest.mark.parametrize("loader", [None, lambda param, w: param.data.copy_(w)])
def test_the_copy_loads_the_whole_of_what_wo_b_loads(loader):
    # Without a loader of its own the copy takes the default one, as wo_b does.
    names = ("weight", "weight_scale")
    wo_b = _linear(K_LOCAL, _rank_slice, names)
    wo_b_full = _linear(TP * K_LOCAL, loader, names)
    weights = {
        f"layers.0.attn.wo_b.{name}": torch.randn(HIDDEN, TP * K_LOCAL)
        for name in names
    }
    loaded = _load([(wo_b, wo_b_full)], weights.items())
    assert loaded == set(weights) | {
        f"layers.0.attn.wo_b_full.{name}" for name in names
    }
    for name in names:
        whole = weights[f"layers.0.attn.wo_b.{name}"]
        assert torch.equal(getattr(wo_b_full, name), whole)
        assert torch.equal(getattr(wo_b, name), whole[:, :K_LOCAL])


def test_a_call_may_carry_only_some_of_wo_b_s_tensors():
    # Weight updates call load_weights with part of the checkpoint (a load
    # plan's fallbacks, a patch): the copy follows what the call brings.
    names = ("weight", "weight_scale")
    layers = [
        (_linear(K_LOCAL, _rank_slice, names), _linear(TP * K_LOCAL, names=names))
        for _ in range(2)
    ]
    whole = torch.randn(HIDDEN, TP * K_LOCAL)
    assert _load(layers, [("layers.1.attn.wo_b.weight", whole)]) == {
        "layers.1.attn.wo_b.weight",
        "layers.1.attn.wo_b_full.weight",
    }
    assert torch.equal(layers[1][1].weight, whole)
    assert not layers[1][1].weight_scale.any() and not layers[0][1].weight.any()


def test_thresholds():
    assert rocm._wo_b_gather_min_rows(2) == 256
    assert rocm._wo_b_gather_min_rows(4) == 1024
    assert rocm._wo_b_gather_min_rows(8) == 1024


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_a_full_piece_is_one_aiter_all_gather(tp_size):
    k_full, k_local = 8192, 8192 // tp_size
    max_size = rocm.AiterCustomAllreduce.MAX_SIZE
    piece = rocm.AiterCustomAllreduce.effective_max_size() // k_full
    # AITER takes max_size / (2 * world_size) bytes a rank, in 16-byte rows.
    assert piece * k_local <= max_size // (2 * tp_size)
    assert k_local % 16 == 0 and (k_local // 32) % 16 == 0


class _OtherKernel:
    """A linear kernel that is not the MXFP8 32x32 one."""


def _copy_builder(monkeypatch, kernel_type=None):
    kernel = object.__new__(kernel_type or rocm.RocmDotScaledMxfp8LinearKernel)
    copy = SimpleNamespace(
        quant_method=SimpleNamespace(kernel=kernel), _input_quant_key=kMxfp8Dynamic
    )
    attn = SimpleNamespace(
        use_pcp=False,
        _o_proj_block_size=32,
        n_groups=8,
        o_lora_rank=1024,
        hidden_size=5120,
        prefix="layers.0.attn",
    )
    comm = SimpleNamespace(
        use_aiter_tp_all_gather=True,
        pynccl_comm=SimpleNamespace(disabled=False),
        aiter_ar_comm=SimpleNamespace(disabled=False),
    )
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(rocm, "_ON_GFX950", True)
    monkeypatch.setattr(rocm, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        rocm, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )

    def replicated_linear(*args, **kwargs):
        copy.built_with = kwargs
        return copy

    monkeypatch.setattr(rocm, "ReplicatedLinear", replicated_linear)
    config = VllmConfig()
    build = rocm.DeepseekV41ROCMAiterMLAAttention._make_wo_b_full
    return lambda: build(attn, config), attn, comm, config, copy


@pytest.mark.parametrize(("tp_size", "threshold"), [(2, 256), (4, 1024)])
def test_the_copy_is_built_for_tensor_parallel_mxfp8(monkeypatch, tp_size, threshold):
    build, attn, _, config, copy = _copy_builder(monkeypatch)
    monkeypatch.setattr(rocm, "get_tensor_model_parallel_world_size", lambda: tp_size)
    config.scheduler_config.max_num_batched_tokens = threshold
    assert build() is copy
    # The copy is quantized as wo_b is: a config that excludes wo_b by name
    # excludes the copy, which then fails the kernel check.
    assert copy.built_with["prefix"] == "layers.0.attn.wo_b"
    assert copy.built_with["quant_config"] is config.quant_config
    assert attn._wo_b_gather_min_rows == threshold


@pytest.mark.parametrize(
    "case",
    [
        "tp1",
        "tp8",
        "dbo",
        "ubatching",
        "small_budget",
        "no_pynccl",
        "disabled_pynccl",
        "no_aiter",
        "disabled_aiter",
        "gfx942",
        "batch_invariant",
        "lora",
        "pcp",
        "block_128",
        "rccl",
        "kernel",
    ],
)
def test_the_copy_is_not_built(monkeypatch, case):
    kernel_type = _OtherKernel if case == "kernel" else None
    build, attn, comm, config, _ = _copy_builder(monkeypatch, kernel_type)
    if case == "tp1":
        monkeypatch.setattr(rocm, "get_tensor_model_parallel_world_size", lambda: 1)
    elif case == "tp8":
        monkeypatch.setattr(rocm, "get_tensor_model_parallel_world_size", lambda: 8)
    elif case == "dbo":
        config.parallel_config.enable_dbo = True
    elif case == "ubatching":
        config.parallel_config.ubatch_size = 2
    elif case == "small_budget":
        config.scheduler_config.max_num_batched_tokens = 255
    elif case == "no_pynccl":
        comm.pynccl_comm = None
    elif case == "disabled_pynccl":
        comm.pynccl_comm.disabled = True
    elif case == "no_aiter":
        comm.aiter_ar_comm = None
    elif case == "disabled_aiter":
        comm.aiter_ar_comm.disabled = True
    elif case == "gfx942":
        monkeypatch.setattr(rocm, "_ON_GFX950", False)
    elif case == "batch_invariant":
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    elif case == "lora":
        config.lora_config = object()
    elif case == "pcp":
        attn.use_pcp = True
    elif case == "block_128":
        attn._o_proj_block_size = 128
    elif case == "rccl":
        comm.use_aiter_tp_all_gather = False
    assert build() is None


@pytest.mark.parametrize(
    ("match_mode", "wo_b_name", "other_name"),
    [
        ("exact", "layers.0.attn.wo_b", "layers.1.attn.wo_b"),
        ("suffix", "wo_b", "wo_a"),
    ],
)
def test_a_config_that_leaves_wo_b_out_gets_no_copy(
    monkeypatch, match_mode, wo_b_name, other_name
):
    # ignored_layers resolves by layer name: the copy has to answer to wo_b's
    # name, or it would be MXFP8 beside an unquantized wo_b, with scales the
    # checkpoint does not have.
    build, _, _, _, copy = _copy_builder(monkeypatch)
    ignored = [wo_b_name]

    def replicated_linear(*args, prefix, **kwargs):
        if is_layer_skipped(prefix, ignored, match_mode=match_mode):
            return SimpleNamespace(quant_method=SimpleNamespace())
        return copy

    monkeypatch.setattr(rocm, "ReplicatedLinear", replicated_linear)
    assert build() is None
    ignored[:] = [other_name]
    assert build() is copy
