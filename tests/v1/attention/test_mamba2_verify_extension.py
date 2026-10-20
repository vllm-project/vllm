# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The linear-attention verify extension on Mamba2 layers."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.mamba_mixer2 as mixer_module
from tests.v1.attention.test_gdn_verify_extension import (
    NAME,
    RecordingExtension,
    created,
    registered,  # noqa: F401 (fixture)
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.verify_extension import (
    Mamba2SpecRows,
)
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadata,
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


class RecordingMamba2Extension(RecordingExtension):
    """Also takes Mamba2 rows: fills their output with ``fill``."""

    def forward_mamba2_spec_rows(self, rows):
        self.rows.append(rows)
        rows.out.fill_(self.fill)


def _vllm_config(method: str, use_replayssm: bool = False):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.model_config.hf_text_config.mamba_chunk_size = 256
    vllm_config.speculative_config = SpeculativeConfig(
        method=method, num_speculative_tokens=3
    )
    vllm_config.cache_config.use_replayssm = use_replayssm
    return vllm_config


def _builder(vllm_config) -> Mamba2AttentionMetadataBuilder:
    # ReplaySSM pages hold (conv, ssm, x, dt, B) states; the builder reads B's.
    num_states = 5 if vllm_config.cache_config.use_replayssm else 1
    return Mamba2AttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((1, 16, 64),) * num_states,
            dtypes=(torch.float16,) * num_states,
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(builder: Mamba2AttentionMetadataBuilder) -> Mamba2AttentionMetadata:
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[40, 50], query_lens=[4, 4]), BLOCK_SIZE, DEVICE
    )
    common = common.replace(is_prefilling=torch.zeros(2, dtype=torch.bool))
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )


# --- creation ---------------------------------------------------------------


def test_no_extension_for_builtin_methods():
    builder = _builder(_vllm_config("ngram"))
    assert builder.verify_extension is None
    assert _build(builder).verify_extension is None


def test_builder_creates_and_attaches_the_registered_extension(registered):  # noqa: F811
    registered()
    vllm_config = _vllm_config(NAME)
    builder = _builder(vllm_config)
    assert len(created) == 1
    assert created[0].vllm_config is vllm_config
    assert builder.verify_extension is created[0]
    metadata = _build(builder)
    assert metadata.verify_extension is created[0]
    assert metadata.num_decodes == 2


def test_extension_survives_update_block_table(registered, monkeypatch):  # noqa: F811
    registered()
    builder = _builder(_vllm_config(NAME))
    monkeypatch.setattr(builder.vllm_config.cache_config, "mamba_cache_mode", "none")
    metadata = _build(builder)
    updated = builder.update_block_table(
        metadata, torch.zeros(2, 8, dtype=torch.int32), torch.zeros(8)
    )
    assert updated.verify_extension is created[0]


def test_replayssm_is_refused_with_an_extension(registered):  # noqa: F811
    registered()
    with pytest.raises(ValueError, match="ReplaySSM"):
        _builder(_vllm_config(NAME, use_replayssm=True))


# --- the layer hands the decode rows to the extension -----------------------

HEADS, HEAD_DIM, GROUPS, D_STATE = 2, 3, 1, 4
INTERMEDIATE = HEADS * HEAD_DIM
CONV_DIM = INTERMEDIATE + 2 * GROUPS * D_STATE
NUM_TOKENS = 4  # one request verifying 4 rows


class _Calls:
    def __init__(self):
        self.conv = self.ssu = self.prefill_conv = self.prefill_scan = 0


METADATA: Mamba2AttentionMetadata


def _layer(monkeypatch, calls: _Calls) -> MambaMixer2:
    def fake_conv(x, *args, **kwargs):
        calls.conv += 1
        return x

    def fake_ssu(*args, out, **kwargs):
        calls.ssu += 1
        out.fill_(-1.0)

    def fake_prefill_conv(x, *args, **kwargs):
        calls.prefill_conv += 1
        return x

    def fake_prefill_scan(x, *args, out, **kwargs):
        calls.prefill_scan += 1
        out.fill_(-2.0)
        return torch.zeros(1, HEADS, HEAD_DIM, D_STATE)

    monkeypatch.setattr(mixer_module, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(mixer_module, "selective_state_update", fake_ssu)
    monkeypatch.setattr(mixer_module, "causal_conv1d_fn", fake_prefill_conv)
    monkeypatch.setattr(
        mixer_module, "mamba_chunk_scan_combined_varlen", fake_prefill_scan
    )
    monkeypatch.setattr(mixer_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        mixer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": METADATA}),
    )
    layer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.cache_config = SimpleNamespace()
    layer.num_spec = NUM_TOKENS - 1
    layer.use_replayssm = False
    layer.tp_size = 1
    layer.num_heads, layer.head_dim = HEADS, HEAD_DIM
    layer.n_groups, layer.ssm_state_size = GROUPS, D_STATE
    layer.tped_intermediate_size = INTERMEDIATE
    layer.tped_conv_size, layer.tped_dt_size = CONV_DIM, HEADS
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(bias=None)
    layer.conv_weights = torch.zeros(CONV_DIM, 4)
    layer.kv_cache = (
        torch.zeros(3, CONV_DIM, 3 + NUM_TOKENS - 1),
        torch.zeros(3, HEADS, HEAD_DIM, D_STATE),
    )
    layer.A, layer.D, layer.dt_bias = (
        -torch.ones(HEADS),
        torch.ones(HEADS),
        torch.ones(HEADS),
    )
    layer.is_blackwell = False
    layer.split_hidden_states_B_C_fn = lambda t: torch.split(
        t, [INTERMEDIATE, GROUPS * D_STATE, GROUPS * D_STATE], dim=-1
    )
    return layer


def _metadata(extension, num_prefill_tokens: int = 0) -> Mamba2AttentionMetadata:
    has_prefill = num_prefill_tokens > 0
    return Mamba2AttentionMetadata(
        num_prefills=int(has_prefill),
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=1,
        num_decode_tokens=NUM_TOKENS,
        num_reqs=1 + int(has_prefill),
        has_initial_states_p=None,
        query_start_loc_p=(
            torch.tensor([0, num_prefill_tokens], dtype=torch.int32)
            if has_prefill
            else None
        ),
        state_indices_tensor_p=(
            torch.tensor([2], dtype=torch.int32) if has_prefill else None
        ),
        state_indices_tensor_d=torch.tensor([[1, 1, 1, 1]], dtype=torch.int32),
        query_start_loc_d=torch.tensor([0, NUM_TOKENS], dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        seq_lens=torch.tensor([40], dtype=torch.int32),
        verify_extension=extension,
    )


def _run(monkeypatch, extension, num_prefill_tokens: int = 0, padded: int = 2):
    global METADATA
    METADATA = _metadata(extension, num_prefill_tokens)
    calls = _Calls()
    layer = _layer(monkeypatch, calls)
    tokens = NUM_TOKENS + num_prefill_tokens + padded
    projected = torch.zeros(tokens, INTERMEDIATE + CONV_DIM + HEADS)
    out = torch.zeros(tokens, INTERMEDIATE)
    layer.conv_ssm_forward(projected, out)
    return calls, out, projected


def test_layer_runs_the_in_tree_kernels_without_an_extension(monkeypatch):
    calls, out, _ = _run(monkeypatch, None)
    assert (calls.conv, calls.ssu) == (1, 1)
    assert torch.all(out[:NUM_TOKENS] == -1.0)


def test_the_extension_replaces_the_decode_kernels(monkeypatch):
    extension = RecordingMamba2Extension()
    calls, out, projected = _run(monkeypatch, extension)
    assert (calls.conv, calls.ssu) == (0, 0)
    assert torch.all(out[:NUM_TOKENS] == 3.0) and torch.all(out[NUM_TOKENS:] == 0)
    (rows,) = extension.rows
    assert isinstance(rows, Mamba2SpecRows)
    # The projections and the output arrive as views cut to the decode rows.
    assert rows.hidden_states_B_C.shape == (NUM_TOKENS, CONV_DIM)
    assert rows.hidden_states_B_C.data_ptr() == projected[:, INTERMEDIATE:].data_ptr()
    assert rows.dt.shape == (NUM_TOKENS, HEADS) and not rows.dt.is_contiguous()
    assert rows.out.shape == (NUM_TOKENS, INTERMEDIATE)
    assert rows.out.data_ptr() == out.data_ptr()
    assert (rows.num_heads, rows.head_dim, rows.n_groups, rows.ssm_state_size) == (
        HEADS,
        HEAD_DIM,
        GROUPS,
        D_STATE,
    )
    assert rows.layer_name == "layer"
    assert rows.state_indices is METADATA.state_indices_tensor_d
    assert rows.num_accepted_tokens is METADATA.num_accepted_tokens
    assert rows.query_start_loc is METADATA.query_start_loc_d


def test_the_extension_gets_only_the_decode_rows_of_a_mixed_batch(monkeypatch):
    extension = RecordingMamba2Extension(fill=7.0)
    calls, out, _ = _run(monkeypatch, extension, num_prefill_tokens=5, padded=0)
    (rows,) = extension.rows
    assert isinstance(rows, Mamba2SpecRows)
    assert rows.hidden_states_B_C.shape[0] == NUM_TOKENS
    assert (calls.conv, calls.ssu) == (0, 0)
    assert (calls.prefill_conv, calls.prefill_scan) == (1, 1)
    assert torch.all(out[:NUM_TOKENS] == 7.0) and torch.all(out[NUM_TOKENS:] == -2.0)
