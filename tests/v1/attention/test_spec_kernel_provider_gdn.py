# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A spec kernel provider's GDN verify kernel: the registry builds it from the
speculator config and binds it to the layer, which hands it its spec rows."""

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer_module
from tests.v1.attention.test_linear_attn_tree_parents import (
    NUM_TOKENS,
    _Calls,
    _layer,
    _spec_metadata,
    _vllm_config,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.worker.gpu.spec_decode import speculator_config as speculator_config_module
from vllm.v1.worker.gpu.spec_decode.kernels import (
    GDNSpecRows,
    KernelRegistry,
    SpecKernel,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    register_speculative_method,
)

NAME = "unit_test_spec_kernel_method"
FACTORY = f"{__name__}.RecordingProvider"
GDN_VERIFY = ("mamba:gdn@slots", "verify")
created: list = []


class RecordingKernel(SpecKernel):
    """Writes ``fill`` to every spec row's output and records what it got."""

    def __init__(self, fill: float = 3.0):
        self.fill = fill
        self.rows: list = []
        self.bound: list[SpecLayerSpec] = []

    def supports(self, layer):
        return frozenset({"chain", "tree"})

    def bind(self, ctx, layers):
        self.bound = list(layers)

    def verify(self, rows):
        self.rows.append(rows)
        rows.out.fill_(self.fill)


class RecordingProvider:
    """Provides ``keys`` (the GDN verify by default) with one RecordingKernel."""

    name = "recording"
    keys: tuple = (GDN_VERIFY,)

    def __init__(self, vllm_config=None):
        self.vllm_config = vllm_config
        self.kernel = RecordingKernel()
        created.append(self)

    def kernels(self):
        return {key: self.kernel for key in self.keys}


@pytest.fixture
def registered(monkeypatch):
    created.clear()
    monkeypatch.setattr(speculator_config_module, "_CUSTOM_CONFIGS", {})

    def _register(providers: tuple[str, ...] = (FACTORY,)):
        config_cls = type(
            "ProviderSpeculatorConfig",
            (SpeculatorConfig,),
            {
                "speculator_class": f"{__name__}.RecordingKernel",
                "spec_kernel_providers": providers,
            },
        )
        register_speculative_method(NAME, config_cls)

    return _register


# --- the registry builds the provider ----------------------------------------


def test_registry_creates_the_named_provider(registered):
    registered()
    vllm_config = _vllm_config(NAME)
    registry = KernelRegistry.from_config(vllm_config)
    (provider,) = created
    assert provider.vllm_config is vllm_config
    assert registry.get(GDN_VERIFY) is provider.kernel


# --- the layer hands the spec-decode rows to the kernel ---------------------


def _run_layer(monkeypatch, kernel, padded: int = 0, parents=None):
    """A spec-only batch of NUM_TOKENS rows; ``b`` / ``a`` are strided views of one
    projection and the buffers have ``padded`` rows past the batch."""
    metadata = _spec_metadata(parents)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    layer.spec_verify_kernel = kernel
    out = torch.zeros(NUM_TOKENS + padded, 2, 3)
    b, a = torch.zeros(NUM_TOKENS + padded, 4).chunk(2, dim=-1)
    layer._forward_core(
        mixed_qkv=torch.zeros(NUM_TOKENS + padded, 5), b=b, a=a, core_attn_out=out
    )
    return calls, out, a, metadata


def test_layer_runs_the_in_tree_kernels_without_a_bound_kernel(monkeypatch):
    calls, out, _, _ = _run_layer(monkeypatch, None)
    assert (calls.conv, calls.recurrent) == (1, 1)
    assert torch.all(out == -1.0)


def test_the_bound_kernel_replaces_the_spec_kernels(monkeypatch):
    kernel = RecordingKernel()
    parents = torch.tensor([-1, 0, 0, 1], dtype=torch.int32)
    calls, out, a, metadata = _run_layer(monkeypatch, kernel, padded=2, parents=parents)
    assert (calls.conv, calls.recurrent) == (0, 0)
    assert torch.all(out[:NUM_TOKENS] == 3.0) and torch.all(out[NUM_TOKENS:] == 0)
    (rows,) = kernel.rows
    assert isinstance(rows, GDNSpecRows)
    # The projections and the output buffer arrive as the views they are, cut to
    # the batch: no copies.
    assert rows.a.data_ptr() == a.data_ptr() and not rows.a.is_contiguous()
    assert rows.mixed_qkv.shape[0] == NUM_TOKENS and rows.out.shape[0] == NUM_TOKENS
    assert rows.layer_name == "layer"
    assert (rows.num_k_heads, rows.num_v_heads, rows.head_k_dim, rows.head_v_dim) == (
        1,
        2,
        2,
        3,
    )
    assert rows.state_indices is metadata.spec_state_indices_tensor
    assert rows.num_accepted_tokens is metadata.num_accepted_tokens
    assert rows.query_start_loc is metadata.spec_query_start_loc
    assert rows.parents is parents


def test_the_fused_norm_entry_passes_the_gate_views_through(monkeypatch):
    kernel = RecordingKernel()
    layer = _layer(monkeypatch, _Calls(), _spec_metadata())
    layer.spec_verify_kernel = kernel
    monkeypatch.setattr(layer, "_rms_norm_gated_cuda", lambda *args: None)
    b, a = torch.zeros(NUM_TOKENS, 4).chunk(2, dim=-1)
    out = torch.zeros(NUM_TOKENS, 2, 3)
    layer._forward_core_fused_norm(
        torch.zeros(NUM_TOKENS, 5), b, a, torch.zeros_like(out), out
    )
    (rows,) = kernel.rows
    assert rows.a.data_ptr() == a.data_ptr() and not rows.a.is_contiguous()


def test_the_kernel_gets_the_spec_rows_of_a_mixed_batch(monkeypatch):
    """Spec rows next to plain decode rows: the kernel gets the spec rows
    only, the decode rows run the in-tree kernels, and both outputs land in
    their rows."""
    kernel = RecordingKernel(fill=7.0)
    metadata = _spec_metadata()
    metadata.num_decodes = metadata.num_decode_tokens = 1
    metadata.num_actual_tokens = NUM_TOKENS + 1
    metadata.spec_token_indx = torch.arange(1, NUM_TOKENS + 1)
    metadata.non_spec_token_indx = torch.tensor([0])
    metadata.non_spec_state_indices_tensor = torch.tensor([1], dtype=torch.int32)
    metadata.non_spec_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    layer.spec_verify_kernel = kernel
    out = torch.zeros(NUM_TOKENS + 1, 2, 3)
    layer._forward_core(
        mixed_qkv=torch.zeros(NUM_TOKENS + 1, 5),
        b=torch.zeros(NUM_TOKENS + 1, 2),
        a=torch.zeros(NUM_TOKENS + 1, 2),
        core_attn_out=out,
    )
    (rows,) = kernel.rows
    assert rows.mixed_qkv.shape[0] == NUM_TOKENS
    assert (calls.conv, calls.recurrent) == (1, 1)  # the decode row
    assert torch.all(out[0] == -1.0) and torch.all(out[1:] == 7.0)


def test_fused_mtp_decode_is_skipped_when_a_kernel_is_bound(monkeypatch):
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    state_dtype = next(iter(gdn_layer_module.FUSED_GDN_STATE_DTYPES))
    layer.kv_cache = (torch.zeros(1), torch.zeros(1, dtype=state_dtype))
    layer.gdn_decode_kernel = "cuda"
    layer.num_v_heads = layer.num_k_heads = 2
    monkeypatch.setattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp", object(), raising=False
    )
    # Without a bound kernel the fused MTP kernel is eligible ...
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata()) is True
    # ... with one, the layer must go through the path that hands it the rows.
    layer.spec_verify_kernel = RecordingKernel()
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata()) is False
