# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Path-hit checks for the opt-in unified FlashInfer paged prefill.

``VLLM_FLASHINFER_PAGED_PREFILL=1`` routes prefill through
``flashinfer.prefill.PagedAttention`` (native FA2) and leaves the default path
untouched when it is off.

Wiring evidence is produced by the real ``FlashInferMetadataBuilder`` driven
with the repo's offline metadata helpers: with the opt-in off the prefill
metadata is the native ``FIPrefill``, with it on it is a ``PagedPrefill`` whose
API object reports the backend it planned on. The opt-in's one-time
preparation is placed the same way: it is recorded and logged while the
builder is constructed, and the first ``build()`` adds none of it.
Disabled-path evidence also includes a clean interpreter that sees neither the
opt-in state nor the experimental implementation package. Nothing here
instruments the attention path, so the check pollutes no timing.

Run on the L20 node (SM89) with the pinned FlashInfer build:
    pytest -q tests/v1/attention/test_flashinfer_paged_prefill_optin.py
"""

import os
import subprocess
import sys
import types

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.attention.backends.flashinfer import (
    PAGED_PREFILL_BACKEND,
    FIDecode,
    FIPrefill,
    FlashInferMetadataBuilder,
    PagedPrefill,
    PagedPrefillAdapter,
    _check_paged_prefill_api_surface,
)
from vllm.v1.attention.backends.utils import get_flashinfer_layout_string
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

SM89 = DeviceCapability(8, 9)
BLOCK_SIZE = 16
requires_sm89 = pytest.mark.skipif(
    current_platform.get_device_capability() != SM89,
    reason="the opt-in is validated on SM89 (L20) only",
)

# flashinfer.prefill itself imports the thin experimental entry module, so
# `flashinfer._paged_attention` is in sys.modules either way; what the opt-in
# alone controls is the implementation package behind the lazy accessors.
EXPERIMENTAL_MODULES = (
    "flashinfer.experimental",
    "flashinfer.experimental.paged_attention",
)

_IMPORT_PROBE = """
import sys

import vllm.v1.attention.backends.flashinfer  # noqa: F401

print(",".join(name for name in %r if name in sys.modules))
""" % (EXPERIMENTAL_MODULES,)


def _run_probe(code: str, **env_overrides: str | None) -> subprocess.CompletedProcess:
    """Run `code` in a clean interpreter; a None override means "unset"."""
    env = {k: v for k, v in os.environ.items() if k not in env_overrides}
    env.update({k: v for k, v in env_overrides.items() if v is not None})
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("VLLM_FLASHINFER_PAGED_PREFILL", "0", "False"),
        ("VLLM_FLASHINFER_PAGED_PREFILL", "1", "True"),
        # Defaults: opt-in off, its init-time warmup on.
        ("VLLM_FLASHINFER_PAGED_PREFILL", None, "False"),
        ("VLLM_FLASHINFER_PAGED_PREFILL_WARMUP", "0", "False"),
        ("VLLM_FLASHINFER_PAGED_PREFILL_WARMUP", "1", "True"),
        ("VLLM_FLASHINFER_PAGED_PREFILL_WARMUP", None, "True"),
    ],
)
def test_optin_flags_are_registered_and_parsed_like_other_flags(
    name: str, value: str | None, expected: str
):
    probe = _run_probe(f"from vllm import envs; print(envs.{name})", **{name: value})
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == expected


def test_optin_off_loads_no_experimental_path():
    """Default-off: the experimental implementation is never imported."""
    probe = _run_probe(_IMPORT_PROBE, VLLM_FLASHINFER_PAGED_PREFILL="0")
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "", f"experimental modules loaded: {probe.stdout}"


def _stub_builder(**overrides):
    """Stand-in for the builder fields the adapter reads.

    ``FlashInferMetadataBuilder`` fills these from the KV-cache spec, the
    global hyperparameters and the platform (see its ``__init__``); the adapter
    only consumes them, so the gate is exercised without a model.
    """
    fields = {
        "device": torch.device("cuda", torch.cuda.current_device()),
        "page_size": BLOCK_SIZE,
        # LBNHC is the first default preference and the layout
        # tests/v1/attention/utils.py hands to builders.
        "kv_cache_layout": KVCacheLayout.LBNHC,
        # Unresolved here, so this stub exercises the deferred pin; the real
        # builder below pins at construction.
        "cache_config": types.SimpleNamespace(kv_cache_layout=None),
        "num_qo_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 128,
        "q_data_type_prefill": torch.bfloat16,
        "cache_dtype": "auto",
        "use_dcp": False,
        "has_sinks": False,
        "logits_soft_cap": None,
        "window_left": -1,
        "reorder_batch_threshold": 1,
        "model_config": types.SimpleNamespace(dtype=torch.bfloat16),
        "attention_config": types.SimpleNamespace(use_non_causal=False),
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_api_surface_probe_rejects_a_drifted_signature():
    """A FlashInfer whose API drifted from the pin fails before any batch."""

    def resolve(
        *,
        device,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        q_dtype,
        page_size,
        kv_layout,
        causal,
        need_lse,
        window_left,
        backend,
    ):
        raise AssertionError("probe must reject before calling")

    def dense(
        qo_indptr,
        kv_seq_lens,
        block_tables,
        *,
        page_size,
        max_q_len,
        max_kv_len,
        qo_indptr_cpu=None,
        kv_seq_lens_cpu=None,
    ):
        raise AssertionError("probe must reject before calling")

    def plan(
        self,
        metadata,
        *,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        q_dtype,
        kv_layout,
        causal,
        window_left,
        lse_mode,
        backend,
    ):
        raise AssertionError("probe must reject before calling")

    def run(self, q, kv_cache, *, out=None, lse=None):
        # `sm_scale` missing: the adapter passes it per run.
        raise AssertionError("probe must reject before calling")

    drifted = types.SimpleNamespace(dense=staticmethod(dense), plan=plan, run=run)
    with pytest.raises(ImportError, match="sm_scale"):
        _check_paged_prefill_api_surface(resolve, drifted, drifted)


@requires_sm89
@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"cache_dtype": "fp8"}, "unquantized KV cache"),
        ({"use_dcp": True}, "decode context parallelism"),
        ({"has_sinks": True}, "attention sinks"),
        ({"logits_soft_cap": 50.0}, "logits soft cap"),
        ({"window_left": 4095}, "window_left=4095"),
        ({"reorder_batch_threshold": 2}, "speculative"),
        ({"attention_config": types.SimpleNamespace(use_non_causal=True)}, "causal"),
        (
            {"model_config": types.SimpleNamespace(dtype=torch.float8_e4m3fn)},
            "fp16/bf16",
        ),
    ],
)
def test_unsupported_configurations_raise_at_start_up(mutation, match):
    """Every prerequisite the pinned contract cannot serve fails loudly."""
    if envs.VLLM_BATCH_INVARIANT:
        pytest.skip("batch-invariant mode is rejected by the opt-in by design")
    with pytest.raises(ValueError, match=match):
        PagedPrefillAdapter(_stub_builder(**mutation))


@requires_sm89
def test_enabled_path_plans_on_fa2_and_writes_the_caller_slice():
    """The enabled path plans on FA2 and fills the output vLLM preallocated."""
    builder = _stub_builder()
    adapter = PagedPrefillAdapter(builder)
    device = builder.device
    page_size = builder.page_size
    num_qo_heads = builder.num_qo_heads
    num_kv_heads = builder.num_kv_heads
    head_dim = builder.head_dim

    torch.manual_seed(0)
    num_decodes = 1
    q_lens = torch.tensor([1, 30, 17], dtype=torch.int32)
    kv_lens = torch.tensor([40, 30, 200], dtype=torch.int32)
    qo_indptr_cpu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(q_lens, 0, dtype=torch.int32),
        ]
    )
    width = int(((kv_lens + page_size - 1) // page_size).max())
    num_reqs = int(q_lens.numel())
    pool_pages = num_reqs * width + 5
    block_table = (
        torch.randperm(pool_pages, dtype=torch.int32)[: num_reqs * width]
        .reshape(num_reqs, width)
        .to(device)
    )

    layout = get_flashinfer_layout_string(builder.kv_cache_layout)
    pool_shape = (
        (pool_pages, num_kv_heads, page_size, head_dim)
        if layout == "HND"
        else (pool_pages, page_size, num_kv_heads, head_dim)
    )
    k_pool = torch.randn(*pool_shape, dtype=torch.bfloat16, device=device)
    v_pool = torch.randn_like(k_pool)

    prefill = adapter.build(
        qo_indptr=qo_indptr_cpu.to(device),
        qo_indptr_cpu=qo_indptr_cpu,
        seq_lens=kv_lens.to(device),
        seq_lens_cpu=kv_lens,
        block_table_tensor=block_table,
        prefill_start=num_decodes,
        kv_layout=layout,
    )

    # Path-hit evidence: the backend the API published for this plan.
    assert prefill.attn.backend == PAGED_PREFILL_BACKEND
    assert f"chosen: {PAGED_PREFILL_BACKEND}" in prefill.attn.explain()

    num_prefill_tokens = int(qo_indptr_cpu[-1] - qo_indptr_cpu[num_decodes])
    q = torch.randn(
        num_prefill_tokens,
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    out = torch.empty_like(q)
    returned, lse = prefill.attn.run(
        q, (k_pool, v_pool), out=out, sm_scale=head_dim**-0.5
    )

    assert lse is None, "prefill plans lse_mode='none'; decode owns LSE in vLLM"
    assert returned.data_ptr() == out.data_ptr(), "must write the caller's slice"
    assert bool(torch.isfinite(returned).all())


@requires_sm89
def _offline_builder_inputs(monkeypatch, device):
    """Config and KV-cache spec for a real builder, with no model built.

    The builder reads per-layer hyperparameters from the model's attention
    layers; this harness builds no model, so register the one layer it is
    asked for, as tests/v1/attention/test_flashinfer_dcp_spec_reorder.py does.
    """
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE, max_model_len=1024, dtype=torch.bfloat16
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )
    return vllm_config, kv_cache_spec


@requires_sm89
def test_builder_routes_prefill_only_when_opted_in(monkeypatch):
    """The real builder: off keeps FIPrefill, on produces a PagedPrefill."""
    device = torch.device("cuda", torch.cuda.current_device())
    vllm_config, kv_cache_spec = _offline_builder_inputs(monkeypatch, device)
    # One decode followed by two prefills, the mixed batch the adapter rebases.
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[40, 30, 200], query_lens=[1, 30, 17]),
        BLOCK_SIZE,
        device,
    )

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_PAGED_PREFILL", False)
    with set_current_vllm_config(vllm_config):
        default_builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, device
        )
        assert default_builder.paged_prefill is None
        default_metadata = default_builder.build(
            common_prefix_len=0, common_attn_metadata=common
        )
    assert isinstance(default_metadata.prefill, FIPrefill)

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_PAGED_PREFILL", True)
    with set_current_vllm_config(vllm_config):
        unified_builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, device
        )
        assert unified_builder.paged_prefill is not None
        unified_metadata = unified_builder.build(
            common_prefix_len=0, common_attn_metadata=common
        )
    assert isinstance(unified_metadata.prefill, PagedPrefill)
    assert unified_metadata.prefill.attn.backend == PAGED_PREFILL_BACKEND
    assert unified_metadata.num_prefills == 2
    assert unified_metadata.num_prefill_tokens == 47
    # Decode is not migrated: it keeps the native FlashInfer wrapper.
    assert isinstance(unified_metadata.decode, FIDecode)


@requires_sm89
def test_warmup_runs_at_construction_and_nowhere_else(monkeypatch):
    """The one-time preparation is paid before the first build(), or never.

    Path evidence, not timing: the adapter records the warmup and logs it, so
    (a) with the opt-in off no adapter exists and nothing is prepared, (b) with
    it on, the preparation is already done and logged while the builder is
    being constructed — before any build() or forward() can run — and the
    first build() adds no second preparation. The control knob leaves the
    cold start in place with the opt-in itself still on.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    vllm_config, kv_cache_spec = _offline_builder_inputs(monkeypatch, device)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[40, 30, 200], query_lens=[1, 30, 17]),
        BLOCK_SIZE,
        device,
    )
    # The vllm logger does not propagate, so listen on the module's own logger.
    lines: list[str] = []
    monkeypatch.setattr(
        flashinfer_backend.logger,
        "handle",
        lambda record: lines.append(record.getMessage()),
    )

    def warmups() -> list[str]:
        return [line for line in lines if "paged prefill warmup" in line]

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_PAGED_PREFILL", False)
    with set_current_vllm_config(vllm_config):
        off_builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, device
        )
    assert off_builder.paged_prefill is None
    assert warmups() == []

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_PAGED_PREFILL", True)
    monkeypatch.setattr(envs, "VLLM_FLASHINFER_PAGED_PREFILL_WARMUP", False)
    with set_current_vllm_config(vllm_config):
        cold_builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, device
        )
    cold_adapter = cold_builder.paged_prefill
    assert cold_adapter is not None
    assert cold_adapter.warmup_s is None
    assert warmups() == []

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_PAGED_PREFILL_WARMUP", True)
    with set_current_vllm_config(vllm_config):
        warm_builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, device
        )
        warm_adapter = warm_builder.paged_prefill
        assert warm_adapter is not None
        warmup_s = warm_adapter.warmup_s
        assert warmup_s is not None
        [warmup] = warmups()
        assert f"backend={PAGED_PREFILL_BACKEND}" in warmup
        first = warm_builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert isinstance(first.prefill, PagedPrefill)
    # The first hot-path batch reused the preparation; it prepared nothing.
    assert warmups() == [warmup]
    assert warm_adapter.warmup_s == warmup_s
