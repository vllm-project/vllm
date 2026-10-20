# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The linear-attention verify extension of a plugin-registered speculative method."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer_module
from tests.v1.attention.test_gdn_metadata_builder import BLOCK_SIZE, DEVICE
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.verify_extension import (
    GDNSpecRows,
    LinearAttnVerifyExtension,
    create_linear_attn_verify_extension,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.spec_decode import registry
from vllm.v1.spec_decode.registry import (
    SpeculativeMethodSpec,
    register_speculative_method,
)

NAME = "unit_test_linear_attn_method"
FACTORY = f"{__name__}:make_extension"
created: list = []


class RecordingExtension(LinearAttnVerifyExtension):
    """Writes ``fill`` to every spec row's output and records what it got."""

    def __init__(self, vllm_config=None, fill: float = 3.0):
        self.vllm_config = vllm_config
        self.fill = fill
        self.rows: list[GDNSpecRows] = []

    def forward_spec_rows(self, rows):
        self.rows.append(rows)
        rows.out.fill_(self.fill)


def make_extension(vllm_config):
    extension = RecordingExtension(vllm_config)
    created.append(extension)
    return extension


@pytest.fixture
def registered(monkeypatch):
    created.clear()
    monkeypatch.setattr(registry, "_REGISTRY", {})

    def _register(factory: str | None = FACTORY):
        register_speculative_method(
            NAME,
            SpeculativeMethodSpec(
                "tests.v1.spec_decode.test_method_registry:DummySpeculator",
                linear_attn_verify_extension=factory,
            ),
        )

    return _register


def _vllm_config(method: str, num_speculative_tokens: int = 3):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method=method, num_speculative_tokens=num_speculative_tokens
    )
    return vllm_config


def _builder(vllm_config) -> GDNAttentionMetadataBuilder:
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(builder: GDNAttentionMetadataBuilder):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[40, 50], query_lens=[4, 4]), BLOCK_SIZE, DEVICE
    )
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )
    return common, metadata


# --- creation ---------------------------------------------------------------


def test_no_extension_for_builtin_methods():
    vllm_config = _vllm_config("ngram")
    assert create_linear_attn_verify_extension(vllm_config) is None
    builder = _builder(vllm_config)
    assert builder.verify_extension is None
    _, metadata = _build(builder)
    assert metadata.verify_extension is None


def test_builder_creates_and_attaches_the_registered_extension(registered):
    registered()
    vllm_config = _vllm_config(NAME)
    builder = _builder(vllm_config)
    assert len(created) == 1
    assert created[0].vllm_config is vllm_config
    assert builder.verify_extension is created[0]
    _, metadata = _build(builder)
    assert metadata.verify_extension is created[0]
    assert metadata.num_spec_decodes == 2


def test_extension_survives_update_block_table(registered, monkeypatch):
    registered()
    builder = _builder(_vllm_config(NAME))
    monkeypatch.setattr(builder.vllm_config.cache_config, "mamba_cache_mode", "none")
    _, metadata = _build(builder)
    block_table = torch.zeros(2, 8, dtype=torch.int32)
    updated = builder.update_block_table(metadata, block_table, torch.zeros(8))
    assert updated.verify_extension is created[0]


# --- the layer hands the spec-decode rows to the extension ------------------

NUM_TOKENS = 4


class _Calls:
    recurrent_a: torch.Tensor  # the gates the fake recurrence received

    def __init__(self):
        self.conv = 0
        self.recurrent = 0


def _layer(monkeypatch, calls: _Calls) -> QwenGatedDeltaNetAttention:
    def fake_conv(mixed_qkv_spec, *args, **kwargs):
        calls.conv += 1
        return mixed_qkv_spec + 100

    def fake_recurrent(**kwargs):
        calls.recurrent += 1
        calls.recurrent_a = kwargs["a"]
        return torch.full((1, kwargs["q"].shape[0], 2, 3), -1.0), None

    monkeypatch.setattr(gdn_layer_module, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(
        gdn_layer_module, "fused_sigmoid_gating_delta_rule_update", fake_recurrent
    )
    monkeypatch.setattr(gdn_layer_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        gdn_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": METADATA}),
    )
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.enable_packed_recurrent_decode = False
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(weight=torch.zeros(5, 1, 4), bias=None)
    layer.kv_cache = (torch.zeros(2, 5, 3), torch.zeros(2, 2, 3, 3))
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.ones(2)
    layer.tp_size = 1
    layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim = (
        1,
        2,
        2,
        3,
    )
    layer.rearrange_mixed_qkv = lambda x: (None, None, None) if x is None else (x, x, x)
    return layer


def _spec_metadata(extension) -> GDNAttentionMetadata:
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=NUM_TOKENS,
        num_actual_tokens=NUM_TOKENS,
        spec_query_start_loc=torch.tensor([0, NUM_TOKENS], dtype=torch.int32),
        spec_state_indices_tensor=torch.zeros(1, 4, dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        verify_extension=extension,
    )


METADATA: GDNAttentionMetadata


def _run_layer(monkeypatch, extension, padded: int = 0):
    """A spec-only batch of NUM_TOKENS rows; ``b`` / ``a`` are strided views of one
    projection and the buffers have ``padded`` rows past the batch."""
    global METADATA
    METADATA = _spec_metadata(extension)
    calls = _Calls()
    layer = _layer(monkeypatch, calls)
    out = torch.zeros(NUM_TOKENS + padded, 2, 3)
    b, a = torch.zeros(NUM_TOKENS + padded, 4).chunk(2, dim=-1)
    layer._forward_core(
        mixed_qkv=torch.zeros(NUM_TOKENS + padded, 5), b=b, a=a, core_attn_out=out
    )
    return calls, out, a


def test_layer_runs_the_in_tree_kernels_without_an_extension(monkeypatch):
    calls, out, _ = _run_layer(monkeypatch, None)
    assert (calls.conv, calls.recurrent) == (1, 1)
    assert torch.all(out == -1.0)
    assert calls.recurrent_a.is_contiguous()  # the chain kernels get contiguous gates


def test_the_extension_replaces_the_spec_kernels(monkeypatch):
    extension = RecordingExtension()
    calls, out, a = _run_layer(monkeypatch, extension, padded=2)
    assert (calls.conv, calls.recurrent) == (0, 0)
    assert torch.all(out[:NUM_TOKENS] == 3.0) and torch.all(out[NUM_TOKENS:] == 0)
    (rows,) = extension.rows
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
    assert rows.state_indices is METADATA.spec_state_indices_tensor
    assert rows.num_accepted_tokens is METADATA.num_accepted_tokens
    assert rows.query_start_loc is METADATA.spec_query_start_loc


def test_the_fused_norm_entry_passes_the_gate_views_through(monkeypatch):
    global METADATA
    extension = RecordingExtension()
    METADATA = _spec_metadata(extension)
    layer = _layer(monkeypatch, _Calls())
    monkeypatch.setattr(layer, "_rms_norm_gated_cuda", lambda *args: None)
    b, a = torch.zeros(NUM_TOKENS, 4).chunk(2, dim=-1)
    out = torch.zeros(NUM_TOKENS, 2, 3)
    layer._forward_core_fused_norm(
        torch.zeros(NUM_TOKENS, 5), b, a, torch.zeros_like(out), out
    )
    (rows,) = extension.rows
    assert rows.a.data_ptr() == a.data_ptr() and not rows.a.is_contiguous()


def test_the_extension_gets_the_spec_rows_of_a_mixed_batch(monkeypatch):
    """Spec rows next to plain decode rows: the extension gets the spec rows
    only, the decode rows run the in-tree kernels, and both outputs land in
    their rows."""
    global METADATA
    extension = RecordingExtension(fill=7.0)
    METADATA = _spec_metadata(extension)
    calls = _Calls()
    layer = _layer(monkeypatch, calls)
    METADATA.num_decodes = METADATA.num_decode_tokens = 1
    METADATA.num_actual_tokens = NUM_TOKENS + 1
    METADATA.spec_token_indx = torch.arange(1, NUM_TOKENS + 1)
    METADATA.non_spec_token_indx = torch.tensor([0])
    METADATA.non_spec_state_indices_tensor = torch.tensor([1], dtype=torch.int32)
    METADATA.non_spec_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    out = torch.zeros(NUM_TOKENS + 1, 2, 3)
    layer._forward_core(
        mixed_qkv=torch.zeros(NUM_TOKENS + 1, 5),
        b=torch.zeros(NUM_TOKENS + 1, 2),
        a=torch.zeros(NUM_TOKENS + 1, 2),
        core_attn_out=out,
    )
    (rows,) = extension.rows
    assert rows.mixed_qkv.shape[0] == NUM_TOKENS
    assert (calls.conv, calls.recurrent) == (1, 1)  # the decode row
    assert torch.all(out[0] == -1.0) and torch.all(out[1:] == 7.0)


def test_fused_mtp_decode_is_skipped_when_an_extension_is_present(monkeypatch):
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    state_dtype = next(iter(gdn_layer_module.FUSED_GDN_STATE_DTYPES))
    layer.kv_cache = (torch.zeros(1), torch.zeros(1, dtype=state_dtype))
    layer.gdn_decode_kernel = "cuda"
    layer.num_v_heads = layer.num_k_heads = 2
    monkeypatch.setattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp", object(), raising=False
    )
    # Without an extension the fused MTP kernel is eligible ...
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata(None)) is True
    # ... with one, the layer must go through the path that hands it the rows.
    with_extension = _spec_metadata(RecordingExtension())
    assert layer._can_use_fused_gdn_mtp_decode(with_extension) is False
