# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExampleConnector keys external KV on the request's block hashes.

Regression tests for #53495: a connector that re-derives its storage key
from raw prompt tokens drops every dimension that partitions the in-engine
prefix cache. Each isolation assertion is paired with its positive control
so a broken store path cannot pass by never hitting.
"""

import tempfile
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config, make_kv_cache_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

BLOCK_SIZE = 16
NUM_FULL_BLOCKS = 3
PROMPT_LEN = NUM_FULL_BLOCKS * BLOCK_SIZE + 5
CACHED_TOKENS = NUM_FULL_BLOCKS * BLOCK_SIZE
EMBED_DIM = 8
NUM_KV_HEADS = 2
HEAD_DIM = 4


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


@pytest.fixture
def connector() -> Iterator[ExampleConnector]:
    with tempfile.TemporaryDirectory() as path:
        vllm_config = create_vllm_config(
            block_size=BLOCK_SIZE,
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": path},
        )
        yield ExampleConnector(
            vllm_config, KVConnectorRole.SCHEDULER, make_kv_cache_config(BLOCK_SIZE)
        )


def make_request(request_id: str, **overrides: Any) -> Request:
    kwargs: dict[str, Any] = {
        "request_id": request_id,
        "prompt_token_ids": [i // BLOCK_SIZE for i in range(PROMPT_LEN)],
        "sampling_params": SamplingParams(max_tokens=17),
        "pooling_params": None,
        "block_hasher": get_request_block_hasher(BLOCK_SIZE, sha256),
    }
    kwargs.update(overrides)
    return Request(**kwargs)


def with_cache_salt(variant: str) -> dict[str, Any]:
    return {"cache_salt": f"salt-{variant}"}


def with_lora(variant: str) -> dict[str, Any]:
    return {
        "lora_request": LoRARequest(
            lora_name=f"lora-{variant}", lora_int_id=7, lora_path="/nonexistent"
        )
    }


def with_prompt_embeds(variant: str) -> dict[str, Any]:
    return {
        "prompt_token_ids": None,
        "prompt_embeds": torch.full((PROMPT_LEN, EMBED_DIM), float(ord(variant[0]))),
    }


DIMENSIONS: list[Callable[[str], dict[str, Any]]] = [
    with_cache_salt,
    with_lora,
    with_prompt_embeds,
]


def schedule_and_store(connector: ExampleConnector, request: Request) -> None:
    """Run one scheduling step and the worker save for ``request``."""
    connector.update_state_after_alloc(request, None, num_external_tokens=0)
    new_req = NewRequestData(
        req_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=[],
        sampling_params=request.sampling_params,
        pooling_params=None,
        block_ids=(list(range(NUM_FULL_BLOCKS + 1)),),
        num_computed_tokens=0,
        lora_request=request.lora_request,
        prompt_embeds=request.prompt_embeds,
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: PROMPT_LEN},
        total_num_scheduled_tokens=PROMPT_LEN,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        preempted_req_ids=set(),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    meta = connector.build_connector_meta(scheduler_output)
    assert len(meta.requests) == 1 and meta.requests[0].is_store
    assert meta.requests[0].slot_mapping.shape == (CACHED_TOKENS,)
    connector.bind_connector_metadata(meta)
    kv_layer = torch.zeros(NUM_FULL_BLOCKS + 1, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    connector.save_kv_layer("layer0", kv_layer, attn_metadata=None)
    connector.clear_connector_metadata()


def lookup(connector: ExampleConnector, request: Request) -> int:
    matched, _ = connector.get_num_new_matched_tokens(request, 0)
    return matched


@pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda f: f.__name__)
def test_same_value_hits(connector: ExampleConnector, dimension):
    schedule_and_store(connector, make_request("first", **dimension("x")))
    assert lookup(connector, make_request("second", **dimension("x"))) == CACHED_TOKENS


@pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda f: f.__name__)
def test_different_value_misses(connector: ExampleConnector, dimension):
    schedule_and_store(connector, make_request("first", **dimension("x")))
    assert lookup(connector, make_request("other", **dimension("y"))) == 0


def test_unset_value_misses_set_value(connector: ExampleConnector):
    schedule_and_store(connector, make_request("plain"))
    assert lookup(connector, make_request("salted", cache_salt="s")) == 0
    assert lookup(connector, make_request("plain2")) == CACHED_TOKENS


def test_short_prompt_is_never_cached(connector: ExampleConnector):
    short = make_request("short", prompt_token_ids=list(range(BLOCK_SIZE)))
    assert lookup(connector, short) == 0
    connector.update_state_after_alloc(short, None, num_external_tokens=0)
    assert connector._pending == {}


def test_hit_reports_tokens_beyond_computed(connector: ExampleConnector):
    schedule_and_store(connector, make_request("first"))
    matched, _ = connector.get_num_new_matched_tokens(
        make_request("second"), BLOCK_SIZE
    )
    assert matched == CACHED_TOKENS - BLOCK_SIZE
