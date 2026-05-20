# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ExampleConnector on the CPU attention backend.

Covered scenarios:

- start_load_kv must inject KV even when forward_context.attn_metadata is
  None (the kv_connector_no_forward path, where all tokens are externally
  available and no local forward pass runs).
- inject_kv_into_layer / extract_kv_from_layer must handle the CPU 5D KV
  layout [2, num_blocks, num_kv_heads, block_size, head_size] correctly —
  the 4D fallback reshape would conflate blocks with heads and corrupt
  the data.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import safetensors.torch
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
    ReqMeta,
)
from vllm.utils.hashing import safe_hash
from vllm.v1.kv_cache_interface import KVCacheConfig

from .utils import create_vllm_config

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 64
LAYER_NAME = "model.layers.0.self_attn"


@dataclass
class FakeForwardContext:
    """Minimal stand-in for vllm.forward_context.ForwardContext."""

    attn_metadata: Any = None
    no_compile_layers: dict[str, Any] = field(default_factory=dict)


def _make_connector(tmp_path) -> ExampleConnector:
    vllm_config = create_vllm_config(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        block_size=BLOCK_SIZE,
        kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    return ExampleConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)


def _filename_for(
    tmp_path, token_ids: torch.Tensor, mm_hashes: list[str], layer_name: str
):
    """Mirror ExampleConnector._generate_filename_debug for test setup."""
    token_bytes = token_ids.numpy().tobytes()
    if mm_hashes:
        token_bytes += "-".join(mm_hashes).encode("utf-8")
    folder = tmp_path / safe_hash(token_bytes, usedforsecurity=False).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{layer_name}.safetensors"


def _layer_with_kv(kv_cache: torch.Tensor):
    layer = MagicMock()
    layer.kv_cache = kv_cache
    return layer


# ---------------------------------------------------------------------------
# start_load_kv must run when forward_context.attn_metadata is None
# ---------------------------------------------------------------------------


def test_start_load_kv_runs_when_attn_metadata_is_none(tmp_path):
    """KV must be injected even when forward_context.attn_metadata is None.

    This is the kv_connector_no_forward path: all tokens are externally
    available so no local forward pass runs and the model runner sets
    attn_metadata to None. The connector must still populate the paged
    KV buffer; otherwise the worker generates from an empty context.
    """
    connector = _make_connector(tmp_path)

    num_tokens = BLOCK_SIZE
    token_ids = torch.arange(num_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    mm_hashes: list[str] = []

    # Standard 4D layout so this test isolates the None-metadata path from
    # the CPU 5D layout handling exercised by the next test.
    head_dim = NUM_KV_HEADS * HEAD_SIZE
    dst_kv = torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, head_dim)
    src_kv = torch.randn(2, num_tokens, head_dim)

    filename = _filename_for(tmp_path, token_ids, mm_hashes, LAYER_NAME)
    safetensors.torch.save_file({"kv_cache": src_kv}, str(filename))

    meta = ExampleConnectorMetadata()
    meta.requests.append(
        ReqMeta(
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_store=False,
            mm_hashes=mm_hashes,
        )
    )
    connector.bind_connector_metadata(meta)

    forward_context = FakeForwardContext(
        attn_metadata=None,  # the kv_connector_no_forward path
        no_compile_layers={LAYER_NAME: _layer_with_kv(dst_kv)},
    )

    connector.start_load_kv(forward_context)

    # With the fix: data was injected via the standard 4D fallback branch.
    flat = dst_kv.reshape(2, NUM_BLOCKS * BLOCK_SIZE, -1)
    assert torch.equal(flat[:, slot_mapping, :], src_kv), (
        "KV not injected when attn_metadata is None"
    )


# ---------------------------------------------------------------------------
# inject_kv_into_layer must handle the CPU 5D KV cache layout
# ---------------------------------------------------------------------------


def test_inject_kv_into_layer_handles_cpu_5d_layout(tmp_path, monkeypatch):
    """KV is correctly injected into a CPU 5D paged buffer.

    The CPU attention backend uses [2, num_blocks, num_kv_heads, block_size,
    head_size]. The 4D fallback would reshape to
    `(2, num_blocks * num_kv_heads, ...)`, conflating blocks with heads and
    landing the data in the wrong rows.

    Distinct values per (k/v, token, head, dim) make any misplacement
    detectable instead of being hidden by accidental broadcast equalities.
    """
    connector = _make_connector(tmp_path)
    # Force the CPU layout flag so this test runs the CPU branch regardless
    # of whether torch was built with CUDA (i.e. on any test runner).
    monkeypatch.setattr(connector, "_uses_cpu_kv_layout", True)

    # Two full blocks worth of tokens.
    num_tokens = BLOCK_SIZE * 2
    token_ids = torch.arange(num_tokens, dtype=torch.int64)
    block_ids = torch.tensor([0, 1], dtype=torch.int64)
    slot_mapping = (
        block_ids.reshape(-1, 1) * BLOCK_SIZE + torch.arange(BLOCK_SIZE).reshape(1, -1)
    ).flatten()
    mm_hashes: list[str] = []

    # CPU 5D destination buffer.
    dst_kv = torch.zeros(2, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    # Source KV: shape [2, num_tokens, num_kv_heads, head_size] with values
    # encoding (k_or_v, token, head, dim) so wrong placements are detectable.
    src_kv = torch.arange(
        2 * num_tokens * NUM_KV_HEADS * HEAD_SIZE, dtype=torch.float32
    ).reshape(2, num_tokens, NUM_KV_HEADS, HEAD_SIZE)

    filename = _filename_for(tmp_path, token_ids, mm_hashes, LAYER_NAME)
    safetensors.torch.save_file({"kv_cache": src_kv}, str(filename))

    meta = ExampleConnectorMetadata()
    meta.requests.append(
        ReqMeta(
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_store=False,
            mm_hashes=mm_hashes,
        )
    )
    connector.bind_connector_metadata(meta)

    forward_context = FakeForwardContext(
        attn_metadata=None,
        no_compile_layers={LAYER_NAME: _layer_with_kv(dst_kv)},
    )

    connector.start_load_kv(forward_context)

    # With the fix: src_kv has been written at dst_kv[:, b, :, o, :] for each
    # slot. Read back via the same advanced indexing — note that non-adjacent
    # advanced indexing in PyTorch returns shape [N, 2, num_kv_heads,
    # head_size], so we transpose the result back to canonical [2, N, ...]
    # before comparing with src_kv.
    block_idxs = slot_mapping // BLOCK_SIZE
    offsets = slot_mapping % BLOCK_SIZE
    actual = dst_kv[:, block_idxs, :, offsets, :].transpose(0, 1)
    assert torch.equal(actual, src_kv), "KV incorrectly injected for CPU 5D layout"


# ---------------------------------------------------------------------------
# extract_kv_from_layer must handle the CPU 5D KV cache layout
# ---------------------------------------------------------------------------


def test_extract_kv_from_layer_handles_cpu_5d_layout(tmp_path, monkeypatch):
    """KV is correctly extracted from a CPU 5D paged buffer.

    Symmetric to the inject side: the 4D fallback would reshape the CPU
    layout to `(2, num_blocks * num_kv_heads, block_size * head_size)` and
    index by slot_mapping, saving data from the wrong heads.
    """
    connector = _make_connector(tmp_path)
    # Force the CPU layout flag — see the inject test for context.
    monkeypatch.setattr(connector, "_uses_cpu_kv_layout", True)

    num_tokens = BLOCK_SIZE * 2
    token_ids = torch.arange(num_tokens, dtype=torch.int64)
    block_ids = torch.tensor([0, 1], dtype=torch.int64)
    slot_mapping = (
        block_ids.reshape(-1, 1) * BLOCK_SIZE + torch.arange(BLOCK_SIZE).reshape(1, -1)
    ).flatten()
    mm_hashes = ["save_5d"]

    # CPU 5D source buffer with distinct values per (k/v, block, head, off, dim).
    src_kv = torch.arange(
        2 * NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE, dtype=torch.float32
    ).reshape(2, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)

    # Expected extraction at the slot positions in canonical [2, N, H * D]
    # shape (the connector flattens the last two dims so the saved tensor
    # matches the convention of the 4D fallback branch).
    block_idxs = slot_mapping // BLOCK_SIZE
    offsets = slot_mapping % BLOCK_SIZE
    expected = (
        src_kv[:, block_idxs, :, offsets, :]
        .transpose(0, 1)
        .reshape(2, num_tokens, -1)
        .clone()
    )

    meta = ExampleConnectorMetadata()
    meta.requests.append(
        ReqMeta(
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_store=True,
            mm_hashes=mm_hashes,
        )
    )
    connector.bind_connector_metadata(meta)

    # save_kv_layer takes (layer_name, kv_layer, attn_metadata, **kwargs).
    # Pass attn_metadata=None so neither MLA nor Triton branches match and
    # the 5D-aware fallback is exercised.
    connector.save_kv_layer(LAYER_NAME, src_kv, attn_metadata=None)

    filename = _filename_for(tmp_path, token_ids, mm_hashes, LAYER_NAME)
    assert filename.exists(), f"KV file not saved: {filename}"
    saved = safetensors.torch.load_file(str(filename))["kv_cache"]
    assert torch.equal(saved, expected), "KV incorrectly extracted for CPU 5D layout"


# ---------------------------------------------------------------------------
# MLA models must dispatch through the 3D-buffer branch even when
# attn_metadata is None
# ---------------------------------------------------------------------------


def test_start_load_kv_routes_mla_when_attn_metadata_is_none(tmp_path, monkeypatch):
    """KV is routed through the MLA branch when the model uses MLA, even on
    the kv_connector_no_forward path where attn_metadata is None.

    The dispatch falls back to the cached self._uses_mla_kv_layout flag set
    from vllm_config.model_config.use_mla. Without that fallback, an MLA
    model on the no-forward path would land in the 4D / 5D fallback and
    crash on the 3D buffer reshape.
    """
    connector = _make_connector(tmp_path)
    # Force the cached MLA flag — avoids needing a real MLA model_config for
    # what is otherwise a pure dispatch test.
    monkeypatch.setattr(connector, "_uses_mla_kv_layout", True)

    num_tokens = BLOCK_SIZE
    token_ids = torch.arange(num_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    mm_hashes: list[str] = []

    # MLA buffer: [num_pages, page_size, latent_dim] — 3D, no leading 2.
    latent_dim = 128
    dst_kv = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, latent_dim)
    src_kv = torch.randn(num_tokens, latent_dim)

    filename = _filename_for(tmp_path, token_ids, mm_hashes, LAYER_NAME)
    safetensors.torch.save_file({"kv_cache": src_kv}, str(filename))

    meta = ExampleConnectorMetadata()
    meta.requests.append(
        ReqMeta(
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_store=False,
            mm_hashes=mm_hashes,
        )
    )
    connector.bind_connector_metadata(meta)

    forward_context = FakeForwardContext(
        attn_metadata=None,
        no_compile_layers={LAYER_NAME: _layer_with_kv(dst_kv)},
    )
    connector.start_load_kv(forward_context)

    flat = dst_kv.reshape(NUM_BLOCKS * BLOCK_SIZE, -1)
    assert torch.equal(flat[slot_mapping, :], src_kv), (
        "KV not injected through MLA branch when model uses MLA + None metadata"
    )
