# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("grpc")
pytest.importorskip("torch")

import grpc

from vllm.grpc import vllm_engine_pb2 as local_pb2
from vllm.grpc import vllm_engine_pb2_grpc as local_pb2_grpc


def _install_grpc_reflection_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    reflection_module: Any = ModuleType("grpc_reflection.v1alpha.reflection")
    reflection_module.SERVICE_NAME = "grpc.reflection.v1alpha.ServerReflection"
    reflection_module.enable_server_reflection = lambda *_args, **_kwargs: None

    v1alpha_module: Any = ModuleType("grpc_reflection.v1alpha")
    v1alpha_module.reflection = reflection_module

    root_module: Any = ModuleType("grpc_reflection")
    root_module.v1alpha = v1alpha_module

    monkeypatch.setitem(sys.modules, "grpc_reflection", root_module)
    monkeypatch.setitem(sys.modules, "grpc_reflection.v1alpha", v1alpha_module)
    monkeypatch.setitem(
        sys.modules,
        "grpc_reflection.v1alpha.reflection",
        reflection_module,
    )


def _install_smg_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    smg_proto_module: Any = ModuleType("smg_grpc_proto")
    smg_proto_module.vllm_engine_pb2 = local_pb2
    smg_proto_module.vllm_engine_pb2_grpc = local_pb2_grpc

    servicer_module: Any = ModuleType("smg_grpc_servicer.vllm.servicer")
    servicer_module.VllmEngineServicer = local_pb2_grpc.VllmEngineServicer

    vllm_module: Any = ModuleType("smg_grpc_servicer.vllm")
    vllm_module.servicer = servicer_module

    root_module: Any = ModuleType("smg_grpc_servicer")
    root_module.vllm = vllm_module

    monkeypatch.setitem(sys.modules, "smg_grpc_proto", smg_proto_module)
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer", root_module)
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer.vllm", vllm_module)
    monkeypatch.setitem(
        sys.modules,
        "smg_grpc_servicer.vllm.servicer",
        servicer_module,
    )


@pytest.fixture
def grpc_server_module(monkeypatch: pytest.MonkeyPatch):
    _install_grpc_reflection_stub(monkeypatch)
    _install_smg_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "vllm.entrypoints.grpc_server", raising=False)
    return importlib.import_module("vllm.entrypoints.grpc_server")


class _DummyRequest:
    @staticmethod
    def FromString(data):
        return data


class _DummyResponse:
    @staticmethod
    def SerializeToString(_message):
        return b""


class _FakeServer:
    def __init__(self) -> None:
        self.generic_handlers: list[Any] = []

    def add_generic_rpc_handlers(self, handlers):
        self.generic_handlers.extend(handlers)


class _DummyStreamer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def subscribe(self, _request, _context):
        if False:
            yield None


class _ProtoAwareDummyStreamer:
    def __init__(self, **kwargs):
        self._pb2 = kwargs["pb2_module"]

    async def subscribe(self, request, _context):
        yield self._pb2.KvEventBatch(
            sequence_number=max(0, int(request.start_sequence_number)),
            timestamp=1.0,
        )


def _make_async_llm():
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_events_config=SimpleNamespace(),
            parallel_config=SimpleNamespace(data_parallel_size=1),
        )
    )


def _make_async_llm_with_kv_events_config(kv_events_config: Any):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_events_config=kv_events_config,
            parallel_config=SimpleNamespace(data_parallel_size=1),
        )
    )


def _make_async_llm_with_kv_events_disabled():
    return _make_async_llm_with_kv_events_config(
        SimpleNamespace(
            enable_kv_cache_events=False,
            publisher="null",
        )
    )


def _make_async_llm_with_kv_events_enabled_local_only():
    return _make_async_llm_with_kv_events_config(
        SimpleNamespace(
            enable_kv_cache_events=True,
            publisher="zmq",
            allow_remote_subscribe=False,
        )
    )


def _subscribe_kv_events_rpc(channel: grpc.aio.Channel):
    return channel.unary_stream(
        "/vllm.grpc.engine.VllmEngine/SubscribeKvEvents",
        request_serializer=local_pb2.SubscribeKvEventsRequest.SerializeToString,
        response_deserializer=local_pb2.KvEventBatch.FromString,
    )


def test_register_subscribe_kv_events_overrides_servicer(
    monkeypatch,
    grpc_server_module,
):
    server = _FakeServer()
    servicer = SimpleNamespace()
    async_llm = _make_async_llm()

    monkeypatch.setattr(grpc_server_module, "GrpcKvEventStreamer", _DummyStreamer)
    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (_DummyRequest, _DummyResponse, SimpleNamespace()),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: True,
    )

    grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)

    assert callable(servicer.SubscribeKvEvents)
    assert server.generic_handlers == []


def test_register_subscribe_kv_events_fallback_uses_descriptor_full_name(
    monkeypatch,
    grpc_server_module,
):
    server = _FakeServer()
    servicer = SimpleNamespace()
    async_llm = _make_async_llm()

    monkeypatch.setattr(grpc_server_module, "GrpcKvEventStreamer", _DummyStreamer)
    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (_DummyRequest, _DummyResponse, SimpleNamespace()),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: False,
    )

    fake_service = SimpleNamespace(full_name="custom.grpc.engine.VllmEngine")
    fake_descriptor = SimpleNamespace(services_by_name={"VllmEngine": fake_service})
    monkeypatch.setattr(
        grpc_server_module,
        "vllm_engine_pb2",
        SimpleNamespace(DESCRIPTOR=fake_descriptor),
    )

    def _fake_unary_stream_rpc_method_handler(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs)

    def _fake_method_handlers_generic_handler(service_name, method_handlers):
        return SimpleNamespace(
            service_name=service_name,
            method_handlers=method_handlers,
        )

    monkeypatch.setattr(
        grpc_server_module,
        "grpc",
        SimpleNamespace(
            unary_stream_rpc_method_handler=_fake_unary_stream_rpc_method_handler,
            method_handlers_generic_handler=_fake_method_handlers_generic_handler,
        ),
    )

    grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)

    assert len(server.generic_handlers) == 1
    handler = server.generic_handlers[0]
    assert handler.service_name == "custom.grpc.engine.VllmEngine"
    assert "SubscribeKvEvents" in handler.method_handlers


def test_register_subscribe_kv_events_fallback_raises_when_service_missing(
    monkeypatch,
    grpc_server_module,
):
    server = _FakeServer()
    servicer = SimpleNamespace()
    async_llm = _make_async_llm()

    monkeypatch.setattr(grpc_server_module, "GrpcKvEventStreamer", _DummyStreamer)
    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (_DummyRequest, _DummyResponse, SimpleNamespace()),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: False,
    )
    monkeypatch.setattr(
        grpc_server_module,
        "vllm_engine_pb2",
        SimpleNamespace(DESCRIPTOR=SimpleNamespace(services_by_name={})),
    )

    with pytest.raises(RuntimeError, match="VllmEngine service descriptor is missing"):
        grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)

    assert server.generic_handlers == []


@pytest.mark.asyncio
async def test_subscribe_kv_events_round_trip_with_servicer_override(
    monkeypatch,
    grpc_server_module,
):
    server = grpc.aio.server()
    servicer = local_pb2_grpc.VllmEngineServicer()
    async_llm = _make_async_llm()

    monkeypatch.setattr(
        grpc_server_module,
        "GrpcKvEventStreamer",
        _ProtoAwareDummyStreamer,
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (
            local_pb2.SubscribeKvEventsRequest,
            local_pb2.KvEventBatch,
            local_pb2,
        ),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: True,
    )

    grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)
    local_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        rpc = _subscribe_kv_events_rpc(channel)
        call = rpc(local_pb2.SubscribeKvEventsRequest(start_sequence_number=42))
        responses = [msg async for msg in call]
        assert len(responses) == 1
        assert responses[0].sequence_number == 42
    finally:
        await channel.close()
        await server.stop(None)


@pytest.mark.asyncio
async def test_subscribe_kv_events_round_trip_with_fallback_handler(
    monkeypatch,
    grpc_server_module,
):
    server = grpc.aio.server()
    servicer = SimpleNamespace()
    async_llm = _make_async_llm()

    monkeypatch.setattr(
        grpc_server_module,
        "GrpcKvEventStreamer",
        _ProtoAwareDummyStreamer,
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (
            local_pb2.SubscribeKvEventsRequest,
            local_pb2.KvEventBatch,
            local_pb2,
        ),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: False,
    )
    monkeypatch.setattr(grpc_server_module, "vllm_engine_pb2", local_pb2)

    def _fake_add_servicer_to_server(_servicer, target_server):
        health_handler = grpc.unary_unary_rpc_method_handler(
            lambda _request, _context: local_pb2.HealthCheckResponse(
                healthy=True,
                message="ok",
            ),
            request_deserializer=local_pb2.HealthCheckRequest.FromString,
            response_serializer=local_pb2.HealthCheckResponse.SerializeToString,
        )
        target_server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "vllm.grpc.engine.VllmEngine",
                    {"HealthCheck": health_handler},
                ),
            )
        )

    monkeypatch.setattr(
        grpc_server_module,
        "vllm_engine_pb2_grpc",
        SimpleNamespace(add_VllmEngineServicer_to_server=_fake_add_servicer_to_server),
    )
    grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)
    grpc_server_module.vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(
        servicer,
        server,
    )

    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        rpc = _subscribe_kv_events_rpc(channel)
        call = rpc(local_pb2.SubscribeKvEventsRequest(start_sequence_number=7))
        responses = [msg async for msg in call]
        assert len(responses) == 1
        assert responses[0].sequence_number == 7
    finally:
        await channel.close()
        await server.stop(None)


@pytest.mark.asyncio
async def test_subscribe_kv_events_returns_unimplemented_when_disabled(
    monkeypatch,
    grpc_server_module,
):
    server = grpc.aio.server()
    servicer = local_pb2_grpc.VllmEngineServicer()
    async_llm = _make_async_llm_with_kv_events_disabled()

    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (
            local_pb2.SubscribeKvEventsRequest,
            local_pb2.KvEventBatch,
            local_pb2,
        ),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: True,
    )

    grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)
    local_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        rpc = _subscribe_kv_events_rpc(channel)
        call = rpc(local_pb2.SubscribeKvEventsRequest(start_sequence_number=0))
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await call.read()

        assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
        assert "KV cache events are not enabled" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(None)


@pytest.mark.asyncio
async def test_subscribe_kv_events_returns_permission_denied_for_remote_peer(
    monkeypatch,
    grpc_server_module,
):
    server = grpc.aio.server()
    servicer = local_pb2_grpc.VllmEngineServicer()
    async_llm = _make_async_llm_with_kv_events_enabled_local_only()

    monkeypatch.setattr(
        grpc_server_module,
        "_resolve_kv_proto_bindings",
        lambda _pb2_module: (
            local_pb2.SubscribeKvEventsRequest,
            local_pb2.KvEventBatch,
            local_pb2,
        ),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "_supports_subscribe_kv_events",
        lambda _pb2_module: True,
    )
    monkeypatch.setattr(
        grpc_server_module.GrpcKvEventStreamer,
        "_peer",
        staticmethod(lambda _context: "ipv4:10.1.2.3:50051"),
    )

    grpc_server_module._register_subscribe_kv_events(server, servicer, async_llm)
    local_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        rpc = _subscribe_kv_events_rpc(channel)
        call = rpc(local_pb2.SubscribeKvEventsRequest(start_sequence_number=0))
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await call.read()

        assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED
        assert "allow_remote_subscribe=true" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(None)
