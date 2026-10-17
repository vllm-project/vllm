# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared vocabulary for the KV-cache key-partitioning conformance suite.

A *partitioning dimension* is any request attribute that must separate the
KV-cache keyspace: two requests that differ in it must never reuse each
other's blocks, and two requests that agree in it must (the positive
control that keeps the negative assertion meaningful).

Each :class:`Dimension` builds ``Request`` keyword arguments from a variant
label, so a test can produce "same" and "different" pairs without knowing
how the dimension is encoded into block hashes. Nothing here inspects the
extra-key tuple, which keeps the suite independent of the encoding.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

BLOCK_SIZE = 16
NUM_FULL_BLOCKS = 3
# Sized past a block boundary so no arm can pass by never hashing a full block.
PROMPT_LEN = NUM_FULL_BLOCKS * BLOCK_SIZE + 5
EMBED_DIM = 8


@dataclass(frozen=True)
class Dimension:
    """A request attribute that must partition the KV-cache keyspace.

    Attributes:
        name: Stable identifier, used as the pytest id.
        build: Maps a variant label to ``Request`` kwargs. Equal labels must
            yield requests that may share blocks; different labels must not.
    """

    name: str
    build: Callable[[str], dict[str, Any]]
    #: Set when the *negative* arm is a known open bug, so the suite records
    #: it as a strict xfail instead of a failure. Strict, so whichever fix
    #: lands turns it into a loud XPASS rather than quiet dead weight.
    negative_bug: str | None = None


def _cache_salt(variant: str) -> dict[str, Any]:
    return {"cache_salt": f"salt-{variant}"}


def _lora_name(variant: str) -> dict[str, Any]:
    return {
        "lora_request": LoRARequest(
            lora_name=f"lora-{variant}", lora_int_id=7, lora_path="/nonexistent"
        )
    }


def _lora_version(variant: str) -> dict[str, Any]:
    """Same adapter name and id, different adapter content.

    ``load_inplace`` replaces an adapter's weights while keeping its name and
    reusing its ``lora_int_id`` (``entrypoints/openai/models/serving.py``), so
    the name alone does not identify what the blocks were computed with.
    """
    return {
        "lora_request": LoRARequest(
            lora_name="pinned", lora_int_id=7, lora_path=f"/adapters/{variant}"
        )
    }


def _prompt_embeds(variant: str) -> dict[str, Any]:
    fill = float(sum(map(ord, variant)))
    return {
        "prompt_token_ids": None,
        "prompt_embeds": torch.full((PROMPT_LEN, EMBED_DIM), fill),
    }


DIMENSIONS: list[Dimension] = [
    Dimension("cache_salt", _cache_salt),
    Dimension("lora_name", _lora_name),
    Dimension(
        "lora_version",
        _lora_version,
        negative_bug=(
            "#42125: the LoRA extra key is the adapter *name* "
            "(kv_cache_utils.py:_gen_lora_extra_hash_keys), so an in-place "
            "reload keeps the key while replacing the weights and the blocks "
            "the old adapter computed are served for the new one. Fixed by "
            "#48352."
        ),
    ),
    Dimension("prompt_embeds", _prompt_embeds),
]


def make_request(request_id: str, **overrides: Any) -> Request:
    """Build a request whose prompt spans ``NUM_FULL_BLOCKS`` full blocks."""
    kwargs: dict[str, Any] = {
        "request_id": request_id,
        "prompt_token_ids": [i // BLOCK_SIZE for i in range(PROMPT_LEN)],
        "sampling_params": SamplingParams(max_tokens=17),
        "pooling_params": None,
        "block_hasher": get_request_block_hasher(BLOCK_SIZE, sha256),
    }
    kwargs.update(overrides)
    return Request(**kwargs)


def make_kv_cache_manager(num_blocks: int = 64) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        scheduler_block_size=BLOCK_SIZE,
        log_stats=True,
    )


def negative_params() -> list:
    """``DIMENSIONS`` as pytest params, with known open bugs marked xfail."""
    out = []
    for dim in DIMENSIONS:
        marks = (
            [pytest.mark.xfail(strict=True, reason=dim.negative_bug)]
            if dim.negative_bug
            else []
        )
        out.append(pytest.param(dim, id=dim.name, marks=marks))
    return out
