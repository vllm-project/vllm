# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector tier of the KV-cache key-partitioning conformance suite.

Same invariant and same dimensions as the in-engine tier, asserted one tier
out: after request A's KV is stored through a connector's own save path,
``get_num_new_matched_tokens`` for request B must report A's blocks only
when B agrees with A in every partitioning dimension. That method is the
one lookup every connector implements and is pure scheduler side, so the
tier runs on CPU with no model forward.

A connector that keys external storage from ``request.block_hashes`` gets
every dimension right for free; one that re-derives keys from raw tokens
gets every dimension wrong. The ``KEYS_FROM_BLOCK_HASHES`` flag on each
harness records which kind it is, and the expected failures follow from it.
"""

import tempfile
from collections.abc import Iterator

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config, make_kv_cache_config
from tests.v1.simple_kv_offload.test_scheduler import (
    _alloc_and_register,
    make_scheduler,
    make_scheduler_output,
    simulate_store_completion,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

from .dimensions import (
    BLOCK_SIZE,
    DIMENSIONS,
    NUM_FULL_BLOCKS,
    PROMPT_LEN,
    Dimension,
    make_request,
)

FULL_PREFIX = NUM_FULL_BLOCKS * BLOCK_SIZE


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


class ConnectorHarness:
    """Store a request's KV through a connector, then look another one up."""

    name: str
    keys_from_block_hashes: bool

    def store(self, request: Request) -> None:
        raise NotImplementedError

    def lookup(self, request: Request) -> int:
        raise NotImplementedError


class ExampleConnectorHarness(ConnectorHarness):
    name = "ExampleConnector"
    keys_from_block_hashes = False

    def __init__(self, storage_path: str):
        vllm_config = create_vllm_config(
            block_size=BLOCK_SIZE,
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": storage_path},
        )
        self.connector = ExampleConnector(
            vllm_config, KVConnectorRole.SCHEDULER, make_kv_cache_config(BLOCK_SIZE)
        )

    def store(self, request: Request) -> None:
        # The scheduler always reports the allocation before building the
        # step's metadata, and connectors may key on it.
        self.connector.update_state_after_alloc(request, None, num_external_tokens=0)
        new_req = NewRequestData(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=[],
            sampling_params=request.sampling_params,
            pooling_params=None,
            block_ids=(list(range(NUM_FULL_BLOCKS)),),
            num_computed_tokens=0,
            lora_request=request.lora_request,
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
        connector = self.connector
        connector.bind_connector_metadata(
            connector.build_connector_meta(scheduler_output)
        )
        # save_kv_layer gathers from whatever paged buffer it is handed.
        connector.save_kv_layer(
            "layer0", torch.zeros(NUM_FULL_BLOCKS, 2, BLOCK_SIZE, 4), attn_metadata=None
        )
        connector.clear_connector_metadata()

    def lookup(self, request: Request) -> int:
        matched, _ = self.connector.get_num_new_matched_tokens(request, 0)
        return matched


class SimpleCPUOffloadHarness(ConnectorHarness):
    name = "SimpleCPUOffloadConnector"
    keys_from_block_hashes = True

    def __init__(self):
        self.fixture = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16)

    def store(self, request: Request) -> None:
        scheduler = self.fixture.scheduler
        kv_blocks = _alloc_and_register(self.fixture, request, NUM_FULL_BLOCKS)
        scheduler.update_state_after_alloc(request, kv_blocks, num_external_tokens=0)
        scheduler_output = make_scheduler_output(
            {request.request_id: FULL_PREFIX},
            new_reqs={request.request_id: kv_blocks.get_block_ids()},
        )
        meta = scheduler.build_connector_meta(scheduler_output)
        assert meta.store_event >= 0
        simulate_store_completion(scheduler, meta.store_event)

    def lookup(self, request: Request) -> int:
        matched, _ = self.fixture.scheduler.get_num_new_matched_tokens(request, 0)
        return matched


HARNESSES = ["ExampleConnector", "SimpleCPUOffloadConnector"]


@pytest.fixture
def harness(request) -> Iterator[ConnectorHarness]:
    if request.param == "ExampleConnector":
        with tempfile.TemporaryDirectory() as path:
            yield ExampleConnectorHarness(path)
    else:
        yield SimpleCPUOffloadHarness()


def _cases(negative: bool) -> list:
    """Cross harnesses with dimensions, marking the known re-derivation bugs."""
    cases = []
    for name in HARNESSES:
        for dim in DIMENSIONS:
            marks = []
            if negative and dim.negative_bug:
                marks.append(pytest.mark.xfail(strict=True, reason=dim.negative_bug))
            elif name == "ExampleConnector" and (
                negative or dim.name == "prompt_embeds"
            ):
                marks.append(
                    pytest.mark.xfail(
                        strict=True,
                        reason="ExampleConnector keys storage on raw prompt "
                        "tokens, so it drops every partitioning dimension and "
                        "keys prompt_embeds requests on an empty prompt.",
                    )
                )
            cases.append(pytest.param(name, dim, id=f"{name}-{dim.name}", marks=marks))
    return cases


@pytest.mark.parametrize("harness,dim", _cases(negative=False), indirect=["harness"])
def test_same_value_reuses_external_blocks(harness: ConnectorHarness, dim: Dimension):
    harness.store(make_request("first", **dim.build("x")))
    assert harness.lookup(make_request("second", **dim.build("x"))) == FULL_PREFIX


@pytest.mark.parametrize("harness,dim", _cases(negative=True), indirect=["harness"])
def test_different_value_never_reuses_external_blocks(
    harness: ConnectorHarness, dim: Dimension
):
    harness.store(make_request("first", **dim.build("x")))
    assert harness.lookup(make_request("other", **dim.build("y"))) == 0
