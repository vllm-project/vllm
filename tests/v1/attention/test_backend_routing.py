# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for batch-level attention-backend routing: config/CLI, the
`kv_layouts_compatible` shared-layout gate, and the `AttentionGroup` prefill-
builder selection. Config/helper tests are CPU-only; tests that resolve real
backends are gated on CUDA."""

import argparse

import pytest
import torch
from pydantic import ValidationError

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.attention import AttentionConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.utils import kv_layouts_compatible
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.worker.utils import AttentionGroup

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_prefill_backend_default_none():
    assert AttentionConfig().prefill_backend is None


def test_prefill_backend_parses_string_and_auto():
    assert AttentionConfig(prefill_backend="FLASHINFER").prefill_backend.name == (
        "FLASHINFER"
    )
    assert AttentionConfig(prefill_backend="auto").prefill_backend is None


def test_decode_and_prefill_backend_both_allowed():
    # Unlike the abandoned composite design, these are NOT mutually exclusive.
    cfg = AttentionConfig(backend="FLASH_ATTN", prefill_backend="FLASHINFER")
    assert cfg.backend.name == "FLASH_ATTN"
    assert cfg.prefill_backend.name == "FLASHINFER"


def test_prefill_backend_participates_in_hash():
    h0 = AttentionConfig(backend="FLASH_ATTN").compute_hash()
    h1 = AttentionConfig(
        backend="FLASH_ATTN", prefill_backend="FLASHINFER"
    ).compute_hash()
    assert h0 != h1


def test_bad_prefill_backend_raises():
    with pytest.raises((ValidationError, KeyError, ValueError)):
        AttentionConfig(prefill_backend="NOT_A_BACKEND")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_flag_registers():
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    dests = {a.option_strings[0]: a.dest for a in parser._actions if a.option_strings}
    assert dests.get("--attention-prefill-backend") == "attention_prefill_backend"


# --------------------------------------------------------------------------- #
# kv_layouts_compatible
# --------------------------------------------------------------------------- #


class _WritesInForwardBackend(AttentionBackend):
    """A minimal non-externalized backend (writes KV inside forward)."""

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "WRITES_IN_FORWARD"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


def _selector_config():
    return AttentionSelectorConfig(
        head_size=128, dtype=torch.bfloat16, kv_cache_dtype=None, block_size=16
    )


def test_kv_layouts_incompatible_when_not_externalized():
    assert not kv_layouts_compatible(
        _WritesInForwardBackend, _WritesInForwardBackend, _selector_config()
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="resolves real CUDA backends"
)
def test_kv_layouts_compatible_same_backend():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    with set_current_vllm_config(VllmConfig()):
        assert kv_layouts_compatible(
            FlashAttentionBackend, FlashAttentionBackend, _selector_config()
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="imports MLA + standard backends"
)
def test_kv_layouts_incompatible_across_families():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend

    with set_current_vllm_config(VllmConfig()):
        assert not kv_layouts_compatible(
            FlashMLABackend, FlashAttentionBackend, _selector_config()
        )


# --------------------------------------------------------------------------- #
# AttentionGroup prefill-builder selection
# --------------------------------------------------------------------------- #


class _FakeBuilder:
    tag = "?"

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.reorder_batch_threshold = 1


def _fake_backend(name, builder_cls):
    return type(
        f"Fake{name}Backend",
        (AttentionBackend,),
        {
            "get_name": staticmethod(lambda: name),
            "get_builder_cls": classmethod(lambda cls: builder_cls),
            "get_impl_cls": staticmethod(lambda: object),
            "get_kv_cache_shape": staticmethod(lambda *a, **k: (1, 2, 3, 4, 5)),
        },
    )


class _FakeSpec:
    def copy_with_new_block_size(self, block_size):
        return self


def test_attention_group_selects_prefill_builder():
    decode_builder = type("DecodeB", (_FakeBuilder,), {"tag": "decode"})
    prefill_builder = type("PrefillB", (_FakeBuilder,), {"tag": "prefill"})
    decode = _fake_backend("DEC", decode_builder)
    prefill = _fake_backend("PRE", prefill_builder)

    group = AttentionGroup(
        backend=decode,
        layer_names=["l0"],
        kv_cache_spec=_FakeSpec(),
        kv_cache_group_id=0,
        prefill_backend=prefill,
    )
    group.create_metadata_builders(vllm_config=None, device=torch.device("cpu"))

    assert group.get_metadata_builder(0).tag == "decode"
    assert group.get_metadata_builder(0, use_prefill=True).tag == "prefill"


def test_attention_group_no_prefill_backend_falls_back_to_decode():
    decode_builder = type("DecodeB2", (_FakeBuilder,), {"tag": "decode"})
    decode = _fake_backend("DEC2", decode_builder)

    group = AttentionGroup(
        backend=decode,
        layer_names=["l0"],
        kv_cache_spec=_FakeSpec(),
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(vllm_config=None, device=torch.device("cpu"))

    # With no prefill backend configured, use_prefill=True still returns decode.
    assert group.get_metadata_builder(0, use_prefill=True).tag == "decode"
    assert group.prefill_metadata_builders == []
