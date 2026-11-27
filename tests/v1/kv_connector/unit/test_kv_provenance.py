# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV a connector loads under a `KVProvenance` is prefix-cached only for
requests with the same provenance, never for native requests."""

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVProvenance
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.v1.request import Request, RequestStatus

from .utils import (
    MockKVConnector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BLOCK_SIZE = 16
NUM_LOADED_TOKENS = 4 * BLOCK_SIZE


def _provenance(request: Request) -> KVProvenance | None:
    tag = (request.kv_transfer_params or {}).get("provenance")
    return KVProvenance(tag) if tag else None


class TaggedKVConnector(MockKVConnector):
    """Loads `num_tokens` tokens of KV under the `provenance` a request sends."""

    def get_kv_provenance(self, request: Request) -> KVProvenance | None:
        return _provenance(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params or {}
        num_tokens = max(params.get("num_tokens", 0) - num_computed_tokens, 0)
        return num_tokens, num_tokens > 0 and self.config.is_async


class TaggedExampleConnector(ExampleConnector):
    """ExampleConnector that loads only for requests with a provenance."""

    def get_kv_provenance(self, request: Request) -> KVProvenance | None:
        return _provenance(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if _provenance(request) is None:
            return 0, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)


for _cls in (TaggedKVConnector, TaggedExampleConnector):
    KVConnectorFactory.register_connector(_cls.__name__, __name__, _cls.__name__)


def _num_prefix_hit_tokens(scheduler, request: Request) -> int:
    """Run a request to completion; return the prompt tokens it did not prefill
    (local and external prefix-cache hits)."""
    scheduler.add_request(request)
    output = scheduler.schedule()
    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        scheduler.update_from_output(
            output,
            create_model_runner_output([], finished_recving={request.request_id}),
        )
        output = scheduler.schedule()
    num_prefilled = output.num_scheduled_tokens[request.request_id]
    scheduler.update_from_output(
        output, create_model_runner_output([request], use_eos=True)
    )
    assert request.is_finished()
    return request.num_prompt_tokens - num_prefilled


@pytest.mark.cpu_test
@pytest.mark.parametrize("is_async", [False, True])
def test_loaded_kv_is_reused_only_under_its_provenance(is_async: bool):
    scheduler = create_scheduler(
        create_vllm_config(
            block_size=BLOCK_SIZE,
            max_num_batched_tokens=1024,
            kv_connector="TaggedKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"matched_tokens": 0, "is_async": is_async},
        )
    )

    def make_request(params: dict | None = None, cache_salt: str | None = None):
        # Every request has the same prompt.
        num_tokens = NUM_LOADED_TOKENS + 1
        request = create_request(
            num_tokens=num_tokens, common_prefix_len=num_tokens, cache_salt=cache_salt
        )
        request.kv_transfer_params = params
        return request

    loaded = {"num_tokens": NUM_LOADED_TOKENS, "provenance": "translated:a"}
    assert _num_prefix_hit_tokens(scheduler, make_request(loaded)) == NUM_LOADED_TOKENS

    # Native requests, a client salt equal to the tag and another tag must
    # not hit the loaded blocks.
    assert _num_prefix_hit_tokens(scheduler, make_request()) == 0
    assert (
        _num_prefix_hit_tokens(scheduler, make_request(cache_salt="translated:a")) == 0
    )
    assert _num_prefix_hit_tokens(scheduler, make_request({"provenance": "b"})) == 0
    # The same tag reuses them from the prefix cache, without a load.
    assert (
        _num_prefix_hit_tokens(scheduler, make_request({"provenance": "translated:a"}))
        == NUM_LOADED_TOKENS
    )


def test_llm_native_request_misses_kv_loaded_under_a_provenance(tmp_path, monkeypatch):
    # Keep the engine in this process, where the test connector is registered.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model="facebook/opt-125m",
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        kv_transfer_config=KVTransferConfig(
            kv_connector="TaggedExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
    )
    # ExampleConnector loads for a prompt that extends a stored one by a token.
    stored = list(range(100, 101 + NUM_LOADED_TOKENS))
    query = stored + [100 + len(stored)]

    def num_cached_tokens(prompt: list[int], provenance: str | None = None) -> int:
        params = SamplingParams(temperature=0, max_tokens=1)
        if provenance is not None:
            params.extra_args = {"kv_transfer_params": {"provenance": provenance}}
        (output,) = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        assert output.num_cached_tokens is not None
        return output.num_cached_tokens

    assert num_cached_tokens(stored) == 0  # The connector stores its KV.
    assert llm.reset_prefix_cache()
    assert num_cached_tokens(query, "translated:a") == NUM_LOADED_TOKENS  # Loaded.
    assert num_cached_tokens(query) == 0
    assert num_cached_tokens(query, "translated:a") == NUM_LOADED_TOKENS
