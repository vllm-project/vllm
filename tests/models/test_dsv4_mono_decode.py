# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4-Pro mono decode layer runs only where its kernels are exact:
decode-only steps of at most 64 rows with causal SWA windows, eager or in a
FULL graph. Every backbone layer takes the whole layer: a CSA layer through
its top-k (its compressor and indexer run between the launches), an HCA layer
through every compressed row (vLLM's c128a table), a hash-routed layer with
its tokens' experts. Deployments the kernels cannot run are an error."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.models.deepseek_v4.amd import mono_decode as md
from vllm.models.deepseek_v4.amd.mono.dims import HIDDEN, MAX_TOKENS, WINDOW

M = 6
HCA_WIDTH = 512
DEFAULT = object()


class _Runner:
    def __init__(self):
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("x", "residual", "post_mix", "res_mix", "extra")


def _cache(block):
    return torch.zeros(4, block * md.RECORD, dtype=torch.uint8)


def _layer(ratio=md.CSA, tid2eid=None):
    attn = SimpleNamespace(
        prefix="comp",
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=_cache(64)),
        kv_cache=_cache(64),
        compress_ratio=ratio,
        topk_indices_buffer=torch.zeros(64, 1024, dtype=torch.int32),
    )
    ffn = SimpleNamespace(gate=SimpleNamespace(tid2eid=tid2eid))
    return SimpleNamespace(self_attn=attn, mlp=ffn)


def _swa(**overrides):
    fields = dict(
        num_prefills=0,
        num_decodes=1,
        num_decode_tokens=M,
        decode_swa_width=WINDOW,
        decode_swa_indices=torch.zeros(64, 1, WINDOW, dtype=torch.int32),
        decode_swa_lens=torch.zeros(64, dtype=torch.int32),
        token_to_req_indices=torch.zeros(64, dtype=torch.int32),
        slot_mapping=torch.zeros(64, dtype=torch.int64),
        block_size=64,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _c128a(rows=M, width=128):
    """VLLM's c128a decode table: its step's rows of a 64-row buffer as wide as
    the longest context, cut to the step's active width."""
    buffer = torch.arange(64 * HCA_WIDTH, dtype=torch.int32).view(64, HCA_WIDTH)
    return buffer[:rows, :width].view(rows, 1, -1)


def _comp(rows=M, **overrides):
    fields = dict(
        block_size=256,
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        c128a_global_decode_topk_indices=_c128a(rows),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _inputs(rows=M):
    return dict(
        x=torch.zeros(rows, 4, HIDDEN, dtype=torch.bfloat16),
        positions=torch.zeros(rows, dtype=torch.int64),
        input_ids=torch.arange(rows),
        residual=torch.zeros(rows, 4, HIDDEN, dtype=torch.bfloat16),
        post_mix=torch.zeros(rows, 4, 1),
        res_mix=torch.zeros(rows, 4, 4),
    )


@pytest.fixture
def run(monkeypatch):
    """Call the mono path for one layer under a forward context built from
    ``mode`` and ``metadata``; returns (result, runner calls). ``run.built``
    records the runner's build arguments."""
    built: list = []

    def _run(mode=CUDAGraphMode.FULL, metadata=DEFAULT, layer=None, **inputs):
        if metadata is DEFAULT:
            metadata = {"swa": _swa(), "comp": _comp()}
        fc = SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata=metadata)
        runner = _Runner()
        monkeypatch.setattr(md, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(md, "get_forward_context", lambda: fc)

        def build(*args):
            built.append(args)
            return runner

        monkeypatch.setattr(md, "_mono_runner", build)
        monkeypatch.setattr(md.MonoDecodeLayer, "weights", lambda self, layer: None)
        mono = md.MonoDecodeLayer(HCA_WIDTH)
        return mono(layer or _layer(), **{**_inputs(), **inputs}), runner.calls

    _run.built = built  # type: ignore[attr-defined]
    return _run


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_csa_step_runs_mono_with_its_topk(run, mode):
    layer = _layer()
    out, calls = run(mode, layer=layer)
    assert out == ("x", "residual", "post_mix", "res_mix") and len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["topk_indices"] is layer.self_attn.topk_indices_buffer
    # the CSA cache: a 256-token page holds 64 compressed records
    assert kwargs["comp_cache"].shape == (4, 64, md.RECORD)
    assert kwargs["fold"] and kwargs["route_ids"] is None
    assert "global_keys" not in kwargs


@pytest.mark.parametrize(
    "ratio, indexed, expected",
    [
        (md.CSA, True, ["compressor", "indexer", "topk"]),
        # a short context's top-k is every row, already filled
        (md.CSA, False, ["compressor", "indexer"]),
        (md.HCA, False, ["compressor"]),
    ],
    ids=["csa", "csa-short-context", "hca"],
)
def test_select_runs_the_compressor_then_the_indexer(run, ratio, indexed, expected):
    """Between K1 and K2 the layer's compressor writes its row and a CSA
    layer's indexer selects the top-k from K1's normed input and qr."""
    log = []
    normed = torch.empty(M, HIDDEN, device="meta")
    qr = torch.empty(M, 1536, device="meta")
    q = torch.empty(M, 4, device="meta") if indexed else None

    class Compressor:
        fused_wkv_wgate = SimpleNamespace(weight=torch.empty(8, HIDDEN, device="meta"))

        def __call__(self, score, positions, rope):
            log.append("compressor")

    class Indexer:
        compressor = Compressor()

        def weights_proj(self, x):
            return x[:, :4], None

        def __call__(self, x, q_rank, score, weights, positions, rope):
            assert x is normed and q_rank is qr
            log.append("indexer")
            return q, None, weights

        def indexer_op(self, x, index_q, _, weights):
            assert x is normed and index_q is q
            log.append("topk")

    layer = _layer(ratio)
    layer.self_attn.compressor = Compressor()
    layer.self_attn.indexer = Indexer() if ratio == md.CSA else None
    layer.self_attn.rotary_emb = layer.self_attn.indexer_rotary_emb = None
    _, calls = run(layer=layer)
    calls[0][1]["select"](normed, qr)
    assert log == expected


@pytest.mark.parametrize("decode_tokens", [M, M - 2], ids=["exact", "padded"])
def test_hca_step_reads_every_compressed_row(run, decode_tokens):
    """An HCA layer's keys are its c128a table's rows: the runner is built as
    wide as vLLM's table, and the kernels read the table's buffer at its row
    stride. A FULL graph padded past the step's decode tokens reads its pad
    rows from the same buffer (their slots are -1)."""
    comp = _comp(rows=decode_tokens)
    metadata = {"swa": _swa(num_decode_tokens=decode_tokens), "comp": comp}
    out, calls = run(metadata=metadata, layer=_layer(ratio=md.HCA))
    assert out is not None and len(calls) == 1
    assert [args[1] for args in run.built] == [HCA_WIDTH]
    kwargs = calls[0][1]
    assert kwargs["global_keys"] and kwargs["topk"] == HCA_WIDTH
    keys = kwargs["topk_indices"]
    table = comp.c128a_global_decode_topk_indices
    assert keys.data_ptr() == table.data_ptr() and keys.stride(0) == HCA_WIDTH
    assert keys.untyped_storage().nbytes() >= M * HCA_WIDTH * keys.element_size()
    assert kwargs["comp_cache"].shape == (4, 2, md.RECORD)


def test_hash_layer_routes_its_tokens(run):
    tid2eid = torch.arange(100 * 6).view(100, 6)
    ids = torch.tensor([3, 1, 4, 1, 5, 9])
    out, calls = run(layer=_layer(tid2eid=tid2eid), input_ids=ids)
    assert out is not None
    assert torch.equal(calls[0][1]["route_ids"], tid2eid[ids])


def test_first_layer_runs_mono_from_the_embedding(run):
    """The first layer's residual is the embedding broadcast to the four
    streams, which K1 takes as it is (no owed post to fold)."""
    out, calls = run(residual=None, post_mix=None, res_mix=None)
    assert out is not None and len(calls) == 1
    args, kwargs = calls[0]
    assert args[2] is args[1] and not kwargs["fold"]


@pytest.mark.parametrize(
    "case",
    [
        dict(mode=CUDAGraphMode.PIECEWISE),
        dict(metadata={"swa": _swa(num_prefills=1), "comp": _comp()}),
        dict(metadata=None),  # a profile / dummy run
        dict(metadata={"swa": _swa()}),
        dict(
            metadata={
                "swa": _swa(),
                "comp": _comp(c128a_global_decode_topk_indices=None),
            },
            layer=_layer(ratio=md.HCA),
        ),
        dict(**_inputs(MAX_TOKENS + 1)),
        dict(residual=torch.zeros(M, HIDDEN, 4, dtype=torch.bfloat16).mT),
        dict(layer=_layer(tid2eid=torch.zeros(100, 6)), input_ids=None),
    ],
    ids=[
        "piecewise",
        "prefill",
        "no-metadata",
        "no-comp-metadata",
        "no-c128a-metadata",
        "rows",
        "non-contiguous",
        "hash-without-ids",
    ],
)
def test_other_steps_take_the_original_path(run, case):
    out, calls = run(**case)
    assert out is None and not calls


def _decoder_layer(layer_id=5, **attn):
    attn = (
        dict(
            prefix=f"model.layers.{layer_id}.self_attn",
            compress_ratio=md.CSA,
            indexer=object(),
            kv_cache_dtype="fp8_ds_mla",
        )
        | attn
    )
    quant_method = SimpleNamespace(mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_BF16)
    ffn = SimpleNamespace(
        experts=SimpleNamespace(
            routed_experts=SimpleNamespace(quant_method=quant_method)
        ),
        shared_experts=object(),
        use_mega_moe=False,
        fuse_heterogeneous_shared_expert=False,
        is_fused_shared_expert_enabled=False,
        gate=SimpleNamespace(tid2eid=None, bias_vl=None),
        n_routed_experts=384,
        n_activated_experts=6,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        renormalize=True,
    )
    return SimpleNamespace(
        self_attn=SimpleNamespace(**attn),
        mlp=ffn,
        use_fused_mhc=True,
        fuse_mhc_rmsnorm=True,
    )


def _config(monokernel=True, max_model_len=8192):
    parallel = dict(
        enable_expert_parallel=False, enable_eplb=False, data_parallel_size=1
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=61),
            max_model_len=max_model_len,
        ),
        parallel_config=SimpleNamespace(**parallel),
        kernel_config=SimpleNamespace(monokernel=monokernel),
    )


def _deployment(monkeypatch, tp=8, aiter_linear=True):
    """A deployment the kernels take (the runner module stubbed: no FlyDSL)."""
    import vllm.platforms.rocm as rocm
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", raising=False)
    monkeypatch.delenv("VLLM_ROCM_FP8_PADDING", raising=False)
    monkeypatch.setattr(rocm, "get_cdna_version", lambda: 4)
    monkeypatch.setattr(md, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(md, "_compute_units", lambda: 256)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_linear_fp8_enabled", classmethod(lambda cls: aiter_linear)
    )
    name = "vllm.models.deepseek_v4.amd.mono.runner"
    runner = ModuleType(name)
    runner.BLOCKS = 256  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, runner)


@pytest.mark.parametrize(
    "layer, eligible",
    [
        (_decoder_layer(), True),
        (_decoder_layer(compress_ratio=md.HCA, indexer=None), True),
        (_decoder_layer(layer_id=61), False),
    ],
    ids=["csa", "hca", "draft"],
)
def test_backbone_layers_take_the_whole_layer(monkeypatch, layer, eligible):
    _deployment(monkeypatch)
    mono = md.MonoDecodeLayer.create(layer, _config())
    assert (mono is not None) == eligible


def test_off_without_the_flag(monkeypatch):
    """No --monokernel: no mono layer, whatever the deployment."""
    _deployment(monkeypatch, tp=4)
    assert (
        md.MonoDecodeLayer.create(_decoder_layer(), _config(monokernel=False)) is None
    )


@pytest.mark.parametrize(
    "case, match",
    [
        (dict(tp=4), "tensor parallel size 8"),
        (dict(aiter_linear=False), "VLLM_ROCM_USE_AITER_LINEAR"),
        (dict(padding="0"), "VLLM_ROCM_FP8_PADDING"),
    ],
    ids=["tp", "aiter-linear", "fp8-padding"],
)
def test_opt_in_outside_the_kernels_raises(monkeypatch, case, match):
    """The kernels read TP8 shards of vLLM's preshuffled, padded aiter FP8
    weights: any other deployment is an error, not garbage."""
    padding = case.get("padding")
    _deployment(monkeypatch, **{k: v for k, v in case.items() if k != "padding"})
    if padding is not None:
        monkeypatch.setenv("VLLM_ROCM_FP8_PADDING", padding)
    with pytest.raises(ValueError, match=match):
        md.MonoDecodeLayer.create(_decoder_layer(), _config())


def test_hca_key_table_is_vllms_c128a_width(monkeypatch):
    """The runner's HCA key table is as wide as the c128a table vLLM's metadata
    builder allocates: every compressed row of the longest context, in 128-row
    multiples (sparse_mla)."""
    _deployment(monkeypatch)
    layer = _decoder_layer(compress_ratio=md.HCA, indexer=None)
    mono = md.MonoDecodeLayer.create(layer, _config(max_model_len=1_000_000))
    assert mono is not None and mono.hca_width == 7936
