# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types
from typing import Optional
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from vllm.distributed.disaggregated.factory import (
    DisaggregatedRequestManagerFactory)
from vllm.distributed.disaggregated.prefill_local_decode_remote_manager import (
    PrefillLocalDecodeRemoteManager)
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)
from vllm.distributed.disaggregated.server_mixin import (
    DisaggregatedServerMixin)
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request


@pytest.fixture(autouse=True)
def _isolate_factory_registry(monkeypatch):
    original = DisaggregatedRequestManagerFactory._registry.copy()
    monkeypatch.setattr(DisaggregatedRequestManagerFactory,
                        "_registry", {},
                        raising=False)
    try:
        yield
    finally:
        DisaggregatedRequestManagerFactory._registry = original


def _dummy_config(enabled: bool):
    cfg = types.SimpleNamespace()
    cfg.kv_transfer_config = object() if enabled else None
    return cfg


class DummyManagerRemoteOnly(DisaggregatedRequestManager):
    priority = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0

    async def dispatch_request(self, request: Request,
                               local_output: Optional[RequestOutput],
                               client: httpx.AsyncClient,
                               local_executed: bool):
        if local_executed and request.kv_transfer_params.get(
                "do_remote_decode", False):
            self._call_count += 1
            return True, {"ok": True}
        return False, None


def test_register_request_manager():
    # Fresh registry
    assert DisaggregatedRequestManagerFactory._registry == {}

    @DisaggregatedRequestManagerFactory.register("low")
    class _DummyManagerLow(DisaggregatedRequestManager):
        priority = 1

        def dispatch_request(self, *args, **kwargs):
            return False, None

    class _DummyManagerHigh(DisaggregatedRequestManager):
        priority = 0

        def dispatch_request(self, *args, **kwargs):
            return False, None

    # Register the two managers
    DisaggregatedRequestManagerFactory.register_request_manager(
        "high", lambda cfg: _DummyManagerHigh(cfg))

    assert set(DisaggregatedRequestManagerFactory._registry.keys()) == {
        "low", "high"
    }

    # Duplicate registration should fail
    with pytest.raises(ValueError):
        DisaggregatedRequestManagerFactory.register_request_manager(
            "low", lambda cfg: _DummyManagerLow(cfg))

    # Creation should order by ascending priority (high comes before low)
    cfg = _dummy_config(enabled=True)
    managers = DisaggregatedRequestManagerFactory.create_request_managers(cfg)
    assert isinstance(managers[0], _DummyManagerHigh)
    assert isinstance(managers[1], _DummyManagerLow)


@pytest.mark.asyncio
async def test_basic_disaggregated_server_mixin_lifecycle(monkeypatch):
    """
    Basic lifecycle test showcasing how DisaggregatedServerMixin is meant 
    to plug into an endpoint.
    """
    DisaggregatedRequestManagerFactory.register_request_manager(
        "dummy", lambda cfg: DummyManagerRemoteOnly(cfg))

    # Disabled path: no managers, context manager yields None
    m_disabled = DisaggregatedServerMixin(vllm_config=_dummy_config(
        enabled=False))
    m_disabled.maybe_setup_disaggregated_server()
    assert m_disabled.managers == []

    dummy_req = Request(
        request_id="r1",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=None,
    )
    dummy_req.kv_transfer_params = dict(
        # Additional custom params for OOT plugins go here
        do_remote_decode=True, )
    # Enabled path: setup should call factory and set managers
    m_enabled = DisaggregatedServerMixin(vllm_config=_dummy_config(
        enabled=True))
    m_enabled.maybe_setup_disaggregated_server()
    assert len(m_enabled.managers) == 1
    manager = m_enabled.managers[0]
    assert isinstance(manager, DummyManagerRemoteOnly)

    # 0. Run manager before_local hook: maybe_run_before_local
    res = await m_enabled.maybe_run_disaggregated_before_local(dummy_req)
    assert res is None
    assert manager._call_count == 0
    # 1. Run local (generate..), mock here
    local_output = RequestOutput(request_id="r1",
                                 prompt="",
                                 prompt_token_ids=[1, 2],
                                 prompt_logprobs=None,
                                 outputs=[],
                                 finished=True)
    # 2. Run manager after_local hook: maybe_run_after_local
    res = await m_enabled.maybe_run_disaggregated_after_local(
        dummy_req, local_output)
    assert manager._call_count == 1
    # 3. Return result, manager has priority if remote returns, mock here
    assert res


@pytest.mark.asyncio
async def test_prefill_local_decode_remote_manager():
    manager = PrefillLocalDecodeRemoteManager(_dummy_config(enabled=True))

    def _make_request(request_id: str) -> Request:
        req = Request(
            request_id=request_id,
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            eos_token_id=None,
        )
        return req

    # Remote decode requests should fall through without modification.
    non_disagg_req = _make_request("r-non-disagg")
    non_disagg_req.kv_transfer_params = {"do_remote_decode": True}
    dispatched, response = await manager.dispatch_request(non_disagg_req,
                                                          None,
                                                          AsyncMock(),
                                                          local_executed=False)
    assert not dispatched
    assert response is None
    assert non_disagg_req.max_tokens == 16

    # Prefill stage should disable streaming and clamp token limits locally.
    prefill_req = _make_request("r-prefill")
    prefill_req.kv_transfer_params = {}
    dispatched, response = await manager.dispatch_request(prefill_req,
                                                          None,
                                                          AsyncMock(),
                                                          local_executed=False)
    assert dispatched is True
    assert response is None
    assert not prefill_req.stream
    assert prefill_req.max_tokens == 1
    assert prefill_req.max_completion_tokens == 1
    assert prefill_req.stream_options is None

    # After local prefill completes, manager should forward to remote decode
    decode_req = _make_request("r-decode")
    decode_req.kv_transfer_params = {}
    remote_params = {"remote_host": "127.0.0.1", "remote_port": 8080}
    local_output = types.SimpleNamespace(kv_transfer_params=remote_params)

    # Mock http client and call
    client = AsyncMock()
    response_payload = {"ok": True}
    dummy_response = Mock()
    dummy_response.raise_for_status = Mock()
    dummy_response.json = Mock(return_value=response_payload)
    client.post.return_value = dummy_response

    dispatched, response = await manager.dispatch_request(decode_req,
                                                          local_output,
                                                          client,
                                                          local_executed=True)

    client.post.assert_awaited_once_with("", json=decode_req)
    dummy_response.raise_for_status.assert_called_once()
    dummy_response.json.assert_called_once()
    assert dispatched
    assert response == response_payload
    assert decode_req.kv_transfer_params == remote_params
