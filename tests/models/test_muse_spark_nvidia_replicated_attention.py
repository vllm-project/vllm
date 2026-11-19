# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark attention replicated across TP ranks (VLLM_MUSE_SPARK_REPLICATE_ATTN).

Under a fake TP=2 rank 1, a replicated layer must hold every head, load whole
checkpoint tensors and never all-reduce; `replicates_attention` must refuse the
layouts that shard heads, tokens or KV on their own.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
import vllm.model_executor.layers.rotary_embedding as rotary_embedding
import vllm.models.muse_spark.nvidia.model as nvidia_model
from vllm.config import CompilationMode
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.triton_utils import tp_synced_autotune

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, TP_SIZE = 128, 4, 2, 64, 2
Q_ROWS, KV_ROWS = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM


def _forbidden_all_reduce(*args, **kwargs):
    raise AssertionError("all-reduce in a replicated attention layer")


def _fake_tp_rank_1(monkeypatch) -> None:
    """Pretend to be rank 1 of a TP=2 group whose all-reduce must never run."""
    for module in (linear, parameter, nvidia_model):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
        )
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        linear, "tensor_model_parallel_all_reduce", _forbidden_all_reduce
    )


def _build_attention(monkeypatch, replicate: bool) -> nn.Module:
    # Keep the CPU rope table out of the get_rope cache later GPU tests use.
    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    monkeypatch.setattr(
        nvidia_model, "Attention", lambda *args, **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(nvidia_model, "supports_fused_kv_insert", lambda attn: False)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        layer_types=["full_attention"],
        layer_rope_theta=[5e5],
        sliding_window=None,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-5,
        qk_scale_factor=1.0,
    )
    return nvidia_model.MuseSparkAttention(
        config=config,
        max_position_embeddings=4096,
        prefix="model.layers.0.self_attn",
        replicate=replicate,
    )


def _checkpoint(rows: int, cols: int, shard_id: str) -> torch.Tensor:
    # Distinct values per row, so a rank-1 slice can't pass for the whole.
    weight = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    weight.shard_id = shard_id
    return weight


def _load_qkv(attn: nn.Module) -> dict[str, torch.Tensor]:
    ckpt = {
        "q": _checkpoint(Q_ROWS, HIDDEN, "q"),
        "k": _checkpoint(KV_ROWS, HIDDEN, "k"),
        "v": _checkpoint(KV_ROWS, HIDDEN, "v"),
        "o_gate": _checkpoint(Q_ROWS, HIDDEN, "o_gate"),
    }
    loaded = list(attn.qkv_proj.load_weights(("weight", w) for w in ckpt.values()))
    assert loaded == ["weight"] * 4
    return ckpt


def test_replicated_attention_holds_and_loads_every_head(
    monkeypatch, default_vllm_config
):
    _fake_tp_rank_1(monkeypatch)
    attn = _build_attention(monkeypatch, replicate=True)

    assert (attn.num_heads, attn.num_kv_heads) == (HEADS, KV_HEADS)
    assert (attn.attn.num_kv_heads, attn.attn.prefix) == (
        KV_HEADS,
        "model.layers.0.self_attn.attn",
    )
    qkv = attn.qkv_proj
    assert (qkv.tp_rank, qkv.tp_size) == (0, 1)
    assert qkv.weight.shape == (2 * (Q_ROWS + KV_ROWS), HIDDEN)

    ckpt = _load_qkv(attn)
    q, k, v, o_gate = qkv.weight.data.split([Q_ROWS, KV_ROWS, KV_ROWS, Q_ROWS])
    for got, name in ((q, "q"), (k, "k"), (v, "v"), (o_gate, "o_gate")):
        torch.testing.assert_close(got, ckpt[name], rtol=0, atol=0)


def test_sharded_attention_still_loads_its_rank_shard(monkeypatch, default_vllm_config):
    # Control: without replicate the same fake world gives rank 1 its half.
    _fake_tp_rank_1(monkeypatch)
    attn = _build_attention(monkeypatch, replicate=False)

    assert (attn.num_heads, attn.num_kv_heads) == (HEADS // 2, KV_HEADS // 2)
    assert (attn.qkv_proj.tp_rank, attn.o_proj.tp_size) == (1, TP_SIZE)
    ckpt = _load_qkv(attn)
    q = attn.qkv_proj.weight.data[: Q_ROWS // 2]
    torch.testing.assert_close(q, ckpt["q"][Q_ROWS // 2 :], rtol=0, atol=0)


def test_replicated_o_proj_is_whole_and_never_reduces(monkeypatch, default_vllm_config):
    _fake_tp_rank_1(monkeypatch)
    o_proj = _build_attention(monkeypatch, replicate=True).o_proj

    assert (o_proj.tp_rank, o_proj.tp_size) == (0, 1)
    assert o_proj.weight.shape == (HIDDEN, Q_ROWS)
    full = torch.randn(HIDDEN, Q_ROWS)
    o_proj.weight.weight_loader(o_proj.weight, full)
    torch.testing.assert_close(o_proj.weight.data, full, rtol=0, atol=0)

    # A forward reaching the all-reduce would raise; the output is complete.
    x = torch.randn(3, Q_ROWS)
    out, _ = o_proj(x)
    torch.testing.assert_close(out, x @ full.T)


def _vllm_config(**overrides) -> SimpleNamespace:
    parallel = dict(
        enable_expert_parallel=False,
        use_sequence_parallel_moe=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=TP_SIZE,
    )
    parallel.update(overrides.pop("parallel", {}))
    config = dict(
        parallel_config=SimpleNamespace(**parallel),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
        kv_transfer_config=None,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.fixture
def installs(monkeypatch) -> list[bool]:
    calls: list[bool] = []
    monkeypatch.setattr(tp_synced_autotune, "install", lambda: calls.append(True))
    return calls


def test_replicates_attention_follows_the_flag(monkeypatch, installs):
    monkeypatch.setattr(envs, "VLLM_MUSE_SPARK_REPLICATE_ATTN", False)
    # Off: refusable layouts don't matter and nothing is installed.
    off = _vllm_config(parallel=dict(enable_expert_parallel=True))
    assert nvidia_model.replicates_attention(off) is False
    assert installs == []

    monkeypatch.setattr(envs, "VLLM_MUSE_SPARK_REPLICATE_ATTN", True)
    assert nvidia_model.replicates_attention(_vllm_config()) is True
    assert installs == [True]


@pytest.mark.parametrize(
    ("overrides", "what"),
    [
        (dict(parallel=dict(use_sequence_parallel_moe=True)), "sequence-parallel"),
        (
            dict(
                parallel=dict(enable_expert_parallel=True),
                kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
            ),
            "sequence-parallel",
        ),
        (dict(parallel=dict(decode_context_parallel_size=2)), "context parallel"),
        (dict(parallel=dict(prefill_context_parallel_size=2)), "context parallel"),
        (dict(parallel=dict(enable_expert_parallel=True)), "expert parallelism"),
        (
            dict(compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)),
            "torch.compile",
        ),
        (
            dict(cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=["0"])),
            "KV skip layers",
        ),
        (dict(kv_transfer_config=object()), "KV transfer"),
    ],
)
def test_replicates_attention_refuses(monkeypatch, installs, overrides, what):
    monkeypatch.setattr(envs, "VLLM_MUSE_SPARK_REPLICATE_ATTN", True)
    with pytest.raises(NotImplementedError, match=what):
        nvidia_model.replicates_attention(_vllm_config(**overrides))
    assert installs == []
