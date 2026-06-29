# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MISSING = object()


class _QuartStub:
    def __init__(self, *args, **kwargs):
        pass

    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def post(self, *args, **kwargs):
        return self.route(*args, **kwargs)


class _RequestStub:
    pass


async def _make_response_stub(value):
    return value


def _module(name, **attrs):
    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    return module


def _package(name):
    module = _module(name)
    module.__path__ = []
    return module


@contextlib.contextmanager
def _proxy_import_stubs():
    class _MoRIIOConstants:
        TRANSFER_PREFIX = "moriio-transfer"

    common_name = "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
    stubs = {
        "aiohttp": _module("aiohttp"),
        "msgpack": _module("msgpack"),
        "zmq": _module("zmq"),
        "quart": _module(
            "quart",
            Quart=_QuartStub,
            Request=_RequestStub,
            make_response=_make_response_stub,
            request=object(),
        ),
        "vllm": _package("vllm"),
        "vllm.distributed": _package("vllm.distributed"),
        "vllm.distributed.kv_transfer": _package("vllm.distributed.kv_transfer"),
        "vllm.distributed.kv_transfer.kv_connector": _package(
            "vllm.distributed.kv_transfer.kv_connector"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1": _package(
            "vllm.distributed.kv_transfer.kv_connector.v1"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio": _package(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio"
        ),
        common_name: _module(
            common_name,
            MoRIIOConstants=_MoRIIOConstants,
        ),
    }
    saved_modules = {}
    for name, module in stubs.items():
        if name not in sys.modules:
            saved_modules[name] = _MISSING
            sys.modules[name] = module

    try:
        yield
    finally:
        for name, previous_module in saved_modules.items():
            if previous_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def _load_proxy_module():
    module_path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location(
        "moriio_toy_proxy_server_under_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _proxy_import_stubs():
        spec.loader.exec_module(module)
    return module


@pytest.fixture()
def proxy_module():
    return _load_proxy_module()


def _instance(name, dp_size):
    return {
        "request_address": f"http://{name}.example/v1",
        "http_address": f"{name}.example:8000",
        "zmq_address": f"host:{name}.example,handshake:1000,notify:2000",
        "dp_size": dp_size,
        "tp_size": 1,
        "transfer_mode": "READ",
        "node_hosts": [f"{name}-host"],
    }


def test_decode_dp1_omits_header_and_keeps_prefill_remote_rank(proxy_module):
    prefill_instance = _instance("prefill", dp_size=2)
    decode_instance = _instance("decode", dp_size=1)

    selected_prefill_rank, selected_decode_rank = proxy_module.select_request_dp_ranks(
        1,
        prefill_instance,
        decode_instance,
    )

    assert selected_prefill_rank == 1
    assert selected_decode_rank is None
    assert "X-data-parallel-rank" not in proxy_module.make_request_headers(
        "request-id", selected_decode_rank
    )

    kv_transfer_params = {}
    proxy_module.add_remote_prefill_kv_params(
        kv_transfer_params,
        prefill_instance,
        selected_prefill_rank,
    )
    assert kv_transfer_params["remote_dp_rank"] == selected_prefill_rank


def test_prefill_request_targets_decode_dp_rank(proxy_module):
    decode_instance = _instance("decode", dp_size=2)
    kv_transfer_params = {}

    proxy_module.add_remote_decode_kv_params(
        kv_transfer_params,
        decode_instance,
        selected_decode_dp_rank=1,
        transfer_id="transfer-0",
    )

    assert kv_transfer_params["remote_dp_size"] == 2
    assert kv_transfer_params["remote_dp_rank"] == 1
    assert (
        kv_transfer_params["remote_zmq_address"]
        == "host:decode.example,handshake:1000,notify:2000"
    )
    assert kv_transfer_params["transfer_id"] == "transfer-0"


def test_prefill_request_omits_decode_dp_rank_for_dp1(proxy_module):
    decode_instance = _instance("decode", dp_size=1)
    kv_transfer_params = {}

    proxy_module.add_remote_decode_kv_params(
        kv_transfer_params,
        decode_instance,
        selected_decode_dp_rank=None,
        transfer_id="transfer-0",
    )

    assert kv_transfer_params["remote_dp_size"] == 1
    assert (
        kv_transfer_params["remote_zmq_address"]
        == "host:decode.example,handshake:1000,notify:2000"
    )
    assert "remote_dp_rank" not in kv_transfer_params


@pytest.mark.parametrize(
    ("request_number", "decode_dp_size", "expected_decode_rank"),
    [
        (1, 2, 1),
        (2, 2, 0),
        (5, 3, 2),
        (6, 3, 0),
    ],
)
def test_decode_dp_rank_cycles_by_decode_endpoint_size(
    proxy_module, request_number, decode_dp_size, expected_decode_rank
):
    selected_prefill_rank, selected_decode_rank = proxy_module.select_request_dp_ranks(
        request_number,
        _instance("prefill", dp_size=8),
        _instance("decode", dp_size=decode_dp_size),
    )

    assert selected_prefill_rank == request_number % 8
    assert selected_decode_rank == expected_decode_rank
    assert 0 <= selected_decode_rank < decode_dp_size


def test_decode_header_rank_is_separate_from_prefill_remote_rank(proxy_module):
    prefill_instance = _instance("prefill", dp_size=4)
    decode_instance = _instance("decode", dp_size=2)

    selected_prefill_rank, selected_decode_rank = proxy_module.select_request_dp_ranks(
        3,
        prefill_instance,
        decode_instance,
    )

    assert selected_prefill_rank == 3
    assert selected_decode_rank == 1
    assert (
        proxy_module.make_request_headers("request-id", selected_decode_rank)[
            "X-data-parallel-rank"
        ]
        == "1"
    )

    kv_transfer_params = {}
    proxy_module.add_remote_prefill_kv_params(
        kv_transfer_params,
        prefill_instance,
        selected_prefill_rank,
    )
    assert kv_transfer_params["remote_dp_rank"] == 3
    assert (
        kv_transfer_params["remote_zmq_address"]
        == "host:prefill.example,handshake:1000,notify:2000"
    )
