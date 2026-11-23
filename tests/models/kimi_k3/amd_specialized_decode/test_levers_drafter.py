# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers and the draft defaults for both Kimi-K3.1 DSpark drafts (CPU).

- The preview (Inferact/Kimi-K3.1-Pro-DSpark-preview): K3DSparkForCausalLM, a
  dense MLA draft with its own fp8 cache on ROCM_AITER_MLA.
- The release (Inferact/Kimi-K3.1-Pro-DSpark): a speculators checkpoint served
  as Qwen3DSparkForCausalLM, a GQA draft (96 / 16 heads, head dim 64, a 2048
  sliding window) with its own model-dtype cache on TRITON_ATTN on ROCm.

``VLLM_K31_BEAT_MARKOV`` (``markov_sample.why_not``) takes both, greedy or
probabilistic, with or without the confidence head. ``levers.dspark`` computes
the confidence head from the drafted tokens as vLLM's loop does.
``VLLM_K31_BEAT_DRAFTMETA`` tells an MLA draft (the bs1 MLA planner) from a GQA
one (nothing to plan). ``configure_dense_draft`` gives a GQA draft on ROCm a
backend that runs there and fits the DSA indexer's BLHNC layout.

The fused launches themselves are compared with vLLM's loop on the GPU in
tests/kernels/models/amd/test_k31_fused_markov.py."""

from types import SimpleNamespace

import pytest
import torch

VOCAB, RANK, N_SPEC = 512, 128, 7


def _draft(cls_name: str, d2t=None):
    if cls_name == "release":
        from vllm.model_executor.models.qwen3_dspark import (
            Qwen3DSparkForCausalLM as cls,
        )
    else:
        from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM as cls

    g = torch.Generator().manual_seed(0)
    head = torch.nn.Module()
    head.markov_w1 = torch.nn.Embedding(
        VOCAB, RANK, _weight=torch.randn(VOCAB, RANK, generator=g).bfloat16()
    )
    head.markov_w2 = torch.nn.Module()
    head.markov_w2.weight = torch.nn.Parameter(
        (torch.randn(VOCAB, RANK, generator=g) * 0.05).bfloat16(), requires_grad=False
    )
    base = (torch.randn(8 * N_SPEC, VOCAB, generator=g) * 3).bfloat16()

    class _Draft(cls):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model = SimpleNamespace(markov_head=head)
            self.draft_id_to_target_id = d2t
            self.logits_processor = SimpleNamespace(
                scale=1.0, soft_cap=None, head_dtype=torch.bfloat16
            )

        def compute_draft_logits(self, hidden):
            return base[: hidden.shape[0]].clone()

        def markov_embed(self, ids):
            return self.model.markov_head.markov_w1(ids)

        def markov_bias(self, embed):
            return embed @ self.model.markov_head.markov_w2.weight.t()

        def map_draft_to_target(self, ids):
            return ids

        def compute_confidence(self, hidden, embed):
            return torch.sigmoid(
                hidden.float().sum(-1) * 1e-2 + embed.float().sum(-1) * 1e-3
            )

    return _Draft()


def _spec(draft: str, num_reqs: int = 1, greedy: bool = True, d2t=None):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    g = torch.Generator().manual_seed(1)
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = _draft(draft, d2t)
    s.num_speculative_steps = N_SPEC
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._d2t_scatter_index = None
    s.use_fp64_gumbel = False
    s.draft_logits = (
        None
        if greedy
        else SimpleNamespace(values=torch.zeros(8, N_SPEC, VOCAB, dtype=torch.bfloat16))
    )
    s.sample_indices = torch.arange(8 * N_SPEC)
    s.sample_idx_mapping = torch.arange(8).repeat_interleave(N_SPEC)
    s.sample_pos = torch.arange(8 * N_SPEC) + 100
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(0, VOCAB, (8 * N_SPEC,), generator=g)
    )
    s._anchor_idx = torch.arange(8) * N_SPEC
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32)
    s.draft_tokens = torch.zeros(8, N_SPEC, dtype=torch.int64)
    s.draft_token_confidence_probs = torch.zeros(8, N_SPEC)
    return s


@pytest.mark.parametrize("draft", ["preview", "release"])
@pytest.mark.parametrize("greedy", [True, False])
def test_markov_takes_both_drafts(draft, greedy):
    from vllm.models.kimi_k3.amd.ops import markov_sample

    s = _spec(draft, greedy=greedy)
    assert markov_sample.why_not(s, 1) is None
    s.use_confidence_head = True
    assert markov_sample.why_not(s, 1) is None
    assert markov_sample.why_not(s, 2) is not None  # MAX_ROWS


def test_markov_refusals():
    from vllm.models.kimi_k3.amd.ops import markov_sample

    s = _spec("release", d2t=torch.zeros(VOCAB, dtype=torch.long))
    assert "reduced draft vocabulary" in markov_sample.why_not(s, 1)
    s = _spec("release")
    s.model = torch.nn.Module()  # another DSpark draft
    assert "not a Kimi-K3.1 DSpark draft" in markov_sample.why_not(s, 1)
    s = _spec("release")
    s.acceptance_estimator = object()
    assert markov_sample.why_not(s, 1) is not None
    s = _spec("release", greedy=False)
    s.draft_logits.values = s.draft_logits.values.float()
    assert "draft logits cache" in markov_sample.why_not(s, 1)


def _stock(s, num_reqs, hidden):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    s.draft_tokens.zero_()
    s.draft_token_confidence_probs.zero_()
    DSparkSpeculator._sample_sequential(s, num_reqs, hidden)
    return s.draft_tokens.clone(), s.draft_token_confidence_probs.clone()


@pytest.mark.parametrize("draft", ["preview", "release"])
@pytest.mark.parametrize("fused", [False, True])
def test_sample_sequential_confidence(monkeypatch, draft, fused):
    """``levers.dspark.sample_sequential`` against vLLM's loop (greedy drafts,
    confidence head on). ``fused``: the launches stand in as a fake that writes
    vLLM's tokens, so the lever must rebuild the confidence head's Markov
    embeddings from the drafted tokens; else they decline and the lever's own
    loop runs."""
    from vllm.models.kimi_k3.amd.ops import markov_sample
    from vllm.models.kimi_k3.amd.specialized_decode.levers import dspark

    num_reqs = 3
    s = _spec(draft)
    s.use_confidence_head = True
    hidden = torch.randn(8 * N_SPEC, 16, generator=torch.Generator().manual_seed(2))
    ref_tok, ref_conf = _stock(s, num_reqs, hidden)
    assert bool(ref_conf[:num_reqs].all())

    def fake_run(spec, n, base_logits):
        if not fused:
            return False
        spec.draft_tokens[:n] = ref_tok[:n]
        return True

    monkeypatch.setattr(markov_sample, "run", fake_run)
    s.draft_tokens.zero_()
    s.draft_token_confidence_probs.zero_()
    dspark.sample_sequential(s, num_reqs, hidden)
    assert torch.equal(s.draft_tokens, ref_tok)
    assert torch.equal(s.draft_token_confidence_probs, ref_conf)


def test_draftmeta_tells_mla_from_gqa_drafts(monkeypatch):
    from vllm.models.kimi_k3.amd.specialized_decode import levers
    from vllm.models.kimi_k3.amd.specialized_decode.levers import dspark
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowSpec

    def group(backend, spec_cls, layers):
        return SimpleNamespace(
            backend=SimpleNamespace(get_name=lambda: backend),
            kv_cache_spec=object.__new__(spec_cls),
            layer_names=layers,
        )

    draft = {f"draft.{i}" for i in range(5)}
    gqa = SimpleNamespace(
        draft_attn_layer_names=draft,
        attn_groups=[
            [],
            [group("TRITON_ATTN", SlidingWindowSpec, sorted(draft))],
        ],
    )
    mla = SimpleNamespace(
        draft_attn_layer_names=draft,
        attn_groups=[[group("ROCM_AITER_MLA", MLAAttentionSpec, sorted(draft))]],
    )
    assert dspark.draft_attn_backends(gqa) == {"TRITON_ATTN": False}
    assert dspark.draft_attn_backends(mla) == {"ROCM_AITER_MLA": True}

    logged = []
    monkeypatch.setattr(
        dspark.logger, "info_once", lambda msg, *args: logged.append(msg % args)
    )
    monkeypatch.setattr(levers, "_active", True)
    monkeypatch.setenv(levers.DRAFTMETA, "1")
    dspark.log_draft_attn(gqa)
    dspark.log_draft_attn(mla)
    assert "TRITON_ATTN" in logged[0] and "not used" in logged[0]
    assert "ROCM_AITER_MLA" in logged[1] and "the bs1 MLA planner" in logged[1]
    monkeypatch.setenv(levers.DRAFTMETA, "0")
    dspark.log_draft_attn(gqa)
    assert len(logged) == 2


def _dense_draft(monkeypatch, rocm: bool, mla: bool, **spec_kwargs):
    from vllm.models.kimi_k3.nvidia import dense_draft

    monkeypatch.setattr(dense_draft.current_platform, "is_rocm", lambda: rocm)
    draft_text = SimpleNamespace(kv_lora_rank=512) if mla else SimpleNamespace()
    spec = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_text_config=draft_text),
        attention_backend=None,
        kv_cache_dtype=None,
    )
    vars(spec).update(spec_kwargs)
    dense_draft.configure_dense_draft(
        spec, SimpleNamespace(linear_attn_config={"num_heads": 96})
    )
    return spec


def test_dense_draft_backends(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B

    s = _dense_draft(monkeypatch, rocm=True, mla=False)
    assert (s.attention_backend, s.kv_cache_dtype) == (
        B.KIMI_K3_FLYDSL_SWA_GQA,
        "auto",
    )
    monkeypatch.setenv("VLLM_K31_DRAFT_FLYDSL_ATTN", "0")
    s = _dense_draft(monkeypatch, rocm=True, mla=False)
    assert (s.attention_backend, s.kv_cache_dtype) == (B.TRITON_ATTN, "auto")
    monkeypatch.delenv("VLLM_K31_DRAFT_FLYDSL_ATTN")
    s = _dense_draft(monkeypatch, rocm=False, mla=False)
    assert (s.attention_backend, s.kv_cache_dtype) == (B.FLASH_ATTN, "auto")
    s = _dense_draft(monkeypatch, rocm=True, mla=True)
    assert (s.attention_backend, s.kv_cache_dtype) == (B.ROCM_AITER_MLA, "fp8")
    # The speculative config's own choices win.
    s = _dense_draft(
        monkeypatch,
        rocm=True,
        mla=False,
        attention_backend=B.ROCM_ATTN,
        kv_cache_dtype="fp8",
    )
    assert (s.attention_backend, s.kv_cache_dtype) == (B.ROCM_ATTN, "fp8")
    # An fp8 draft cache alone (InferenceX #5119 c16-c48 without its
    # TRITON_ATTN pin) keeps the FlyDSL backend.
    s = _dense_draft(monkeypatch, rocm=True, mla=False, kv_cache_dtype="fp8")
    assert (s.attention_backend, s.kv_cache_dtype) == (
        B.KIMI_K3_FLYDSL_SWA_GQA,
        "fp8",
    )


def test_rocm_gqa_draft_backend_fits_the_dsa_layout():
    """The DSA indexer pins the worker to BLHNC: the ROCm GQA draft's default
    backend must allow it (the worker intersects every backend's layouts)."""
    from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerBackend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B

    indexer = KimiK3IndexerBackend.supported_kv_cache_layouts()
    for backend in (B.KIMI_K3_FLYDSL_SWA_GQA, B.TRITON_ATTN):
        draft = backend.get_class().supported_kv_cache_layouts()
        assert draft is None or set(indexer) <= set(draft)
