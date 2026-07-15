# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GET /v1/server/kv-cache.

These tests do not spin up a real vLLM server or load model weights.
They exercise:
  - _build_kv_cache_response() directly (pure function, fast, no I/O)
  - _build_group_spec() directly (discriminator dispatch, pure function)
  - The FastAPI route via TestClient (HTTP stack, no engine)
  - The Pydantic protocol models (schema correctness)
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.kv_cache.api_router import (
    _build_group_spec,
    _build_kv_cache_response,
    attach_router,
    router,
)
from vllm.entrypoints.serve.kv_cache.protocol import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    FullAttentionGroupSpec,
    KVCacheRuntimeInfo,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    UniformTypeGroupSpec,
)

# ---------------------------------------------------------------------------
# Helpers — serialized group dicts (mirrors what engine core produces)
# ---------------------------------------------------------------------------

_BASE_GROUP = {
    "group_id": 0,
    "layer_names": ["model.layers.0.self_attn", "model.layers.1.self_attn"],
    "block_size": 16,
    "page_size_bytes": 131072,
}


def _full_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "FullAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": "bfloat16",
        "sliding_window": None,
        "attention_chunk_size": None,
        **overrides,
    }


def _mla_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "MLAAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 64,
        "dtype": "bfloat16",
        "sliding_window": None,
        "attention_chunk_size": None,
        "cache_dtype_str": "float8_e4m3fn",
        **overrides,
    }


def _sliding_window_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "SlidingWindowSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "float16",
        "sliding_window": 4096,
        **overrides,
    }


def _chunked_local_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "ChunkedLocalAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "float16",
        "attention_chunk_size": 2048,
        **overrides,
    }


def _mamba_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "MambaSpec",
        "block_size": 1,
        "page_size_bytes": 4096,
        "shapes": [[16, 128], [16, 64]],
        "dtypes": ["float32", "float32"],
        "mamba_type": "mamba2",
        "mamba_cache_mode": "none",
        **overrides,
    }


def _cross_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "CrossAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "bfloat16",
        **overrides,
    }


def _sink_full_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "SinkFullAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": "bfloat16",
        "sliding_window": 2048,
        "attention_chunk_size": None,
        "sink_len": 4,
        **overrides,
    }


def _uniform_type_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "UniformTypeKVCacheSpecs",
        "layer_specs": [
            {"head_size": 128, "dtype": "bfloat16"},
            {"head_size": 64, "dtype": "bfloat16"},
        ],
        **overrides,
    }


def _kv_cache_data(**overrides) -> dict:
    """Build a minimal kv_cache_config dict as produced by EngineCore."""
    return {
        "kv_cache_size_tokens": 16384,
        "max_concurrency": 0.5,
        "num_gpu_blocks": 1024,
        "num_cpu_blocks": 256,
        "groups": [_full_attention_dict()],
        **overrides,
    }


def _make_app(kv_cache_config: dict | None) -> FastAPI:
    """Return a minimal FastAPI app with the kv-cache router and state injected."""
    app = FastAPI()
    attach_router(app)
    app.state.kv_cache_config = kv_cache_config
    return app


def _make_client(kv_cache_config: dict | None) -> TestClient:
    """Return a TestClient wrapping a minimal kv-cache app."""
    return TestClient(_make_app(kv_cache_config))


# ---------------------------------------------------------------------------
# _build_group_spec unit tests — discriminator dispatch
# ---------------------------------------------------------------------------


class TestBuildGroupSpecDispatch:
    def test_full_attention_spec(self):
        result = _build_group_spec(_full_attention_dict())
        assert isinstance(result, FullAttentionGroupSpec)
        assert result.spec_type == "FullAttentionSpec"

    def test_mla_attention_spec(self):
        result = _build_group_spec(_mla_attention_dict())
        assert isinstance(result, MLAAttentionGroupSpec)
        assert result.spec_type == "MLAAttentionSpec"

    def test_sliding_window_spec(self):
        result = _build_group_spec(_sliding_window_dict())
        assert isinstance(result, SlidingWindowGroupSpec)
        assert result.spec_type == "SlidingWindowSpec"

    def test_chunked_local_attention_spec(self):
        result = _build_group_spec(_chunked_local_attention_dict())
        assert isinstance(result, ChunkedLocalAttentionGroupSpec)
        assert result.spec_type == "ChunkedLocalAttentionSpec"

    def test_mamba_spec(self):
        result = _build_group_spec(_mamba_dict())
        assert isinstance(result, MambaGroupSpec)
        assert result.spec_type == "MambaSpec"

    def test_cross_attention_spec(self):
        result = _build_group_spec(_cross_attention_dict())
        assert isinstance(result, CrossAttentionGroupSpec)
        assert result.spec_type == "CrossAttentionSpec"

    def test_sink_full_attention_spec(self):
        result = _build_group_spec(_sink_full_attention_dict())
        assert isinstance(result, SinkFullAttentionGroupSpec)
        assert result.spec_type == "SinkFullAttentionSpec"

    def test_uniform_type_spec(self):
        result = _build_group_spec(_uniform_type_dict())
        assert isinstance(result, UniformTypeGroupSpec)
        assert result.spec_type == "UniformTypeKVCacheSpecs"

    def test_unknown_spec_type_raises_value_error(self):
        group = {**_BASE_GROUP, "spec_type": "UnknownSpec"}
        with pytest.raises(ValueError, match="Unhandled KVCacheSpec type"):
            _build_group_spec(group)

    def test_preserves_base_fields(self):
        result = _build_group_spec(_full_attention_dict(group_id=3))
        assert result.group_id == 3
        assert result.layer_names == _BASE_GROUP["layer_names"]
        assert result.block_size == 16
        assert result.page_size_bytes == 131072


# ---------------------------------------------------------------------------
# _build_kv_cache_response unit tests — pure function
# ---------------------------------------------------------------------------


class TestBuildKVCacheResponseNullState:
    def test_none_state(self):
        resp = _build_kv_cache_response(None)
        assert isinstance(resp, KVCacheRuntimeInfo)
        assert resp.kv_cache_size_tokens is None
        assert resp.max_concurrency is None
        assert resp.num_gpu_blocks is None
        assert resp.num_cpu_blocks is None
        assert resp.groups == []


class TestBuildKVCacheResponseCapacityFields:
    def test_kv_cache_size_tokens(self):
        resp = _build_kv_cache_response(_kv_cache_data(kv_cache_size_tokens=32768))
        assert resp.kv_cache_size_tokens == 32768

    def test_max_concurrency(self):
        resp = _build_kv_cache_response(_kv_cache_data(max_concurrency=1.25))
        assert resp.max_concurrency == pytest.approx(1.25)

    def test_num_gpu_blocks(self):
        resp = _build_kv_cache_response(_kv_cache_data(num_gpu_blocks=2048))
        assert resp.num_gpu_blocks == 2048

    def test_num_cpu_blocks(self):
        resp = _build_kv_cache_response(_kv_cache_data(num_cpu_blocks=512))
        assert resp.num_cpu_blocks == 512

    def test_num_cpu_blocks_zero(self):
        resp = _build_kv_cache_response(_kv_cache_data(num_cpu_blocks=0))
        assert resp.num_cpu_blocks == 0

    def test_missing_capacity_fields_resolve_to_none(self):
        resp = _build_kv_cache_response({"groups": []})
        assert resp.kv_cache_size_tokens is None
        assert resp.max_concurrency is None
        assert resp.num_gpu_blocks is None
        assert resp.num_cpu_blocks is None


class TestBuildKVCacheResponseGroups:
    def test_empty_groups_list(self):
        resp = _build_kv_cache_response(_kv_cache_data(groups=[]))
        assert resp.groups == []

    def test_single_full_attention_group(self):
        resp = _build_kv_cache_response(_kv_cache_data(groups=[_full_attention_dict()]))
        assert len(resp.groups) == 1
        assert isinstance(resp.groups[0], FullAttentionGroupSpec)

    def test_single_mamba_group(self):
        resp = _build_kv_cache_response(_kv_cache_data(groups=[_mamba_dict()]))
        assert len(resp.groups) == 1
        assert isinstance(resp.groups[0], MambaGroupSpec)

    def test_hybrid_model_two_groups(self):
        groups = [
            _full_attention_dict(group_id=0),
            _mamba_dict(group_id=1),
        ]
        resp = _build_kv_cache_response(_kv_cache_data(groups=groups))
        assert len(resp.groups) == 2
        assert isinstance(resp.groups[0], FullAttentionGroupSpec)
        assert isinstance(resp.groups[1], MambaGroupSpec)

    def test_all_spec_types_in_groups(self):
        groups = [
            _full_attention_dict(group_id=0),
            _mla_attention_dict(group_id=1),
            _sliding_window_dict(group_id=2),
            _chunked_local_attention_dict(group_id=3),
            _mamba_dict(group_id=4),
            _cross_attention_dict(group_id=5),
            _sink_full_attention_dict(group_id=6),
            _uniform_type_dict(group_id=7),
        ]
        expected_types = [
            FullAttentionGroupSpec,
            MLAAttentionGroupSpec,
            SlidingWindowGroupSpec,
            ChunkedLocalAttentionGroupSpec,
            MambaGroupSpec,
            CrossAttentionGroupSpec,
            SinkFullAttentionGroupSpec,
            UniformTypeGroupSpec,
        ]
        resp = _build_kv_cache_response(_kv_cache_data(groups=groups))
        assert len(resp.groups) == 8
        for group, expected_cls in zip(resp.groups, expected_types):
            assert isinstance(group, expected_cls)

    def test_group_id_preserved(self):
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_full_attention_dict(group_id=5)])
        )
        assert resp.groups[0].group_id == 5

    def test_layer_names_preserved(self):
        names = ["model.layers.2.self_attn", "model.layers.3.self_attn"]
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_full_attention_dict(layer_names=names)])
        )
        assert resp.groups[0].layer_names == names


class TestBuildKVCacheResponseGroupFields:
    def test_full_attention_optional_fields_none(self):
        group = _full_attention_dict(sliding_window=None, attention_chunk_size=None)
        resp = _build_kv_cache_response(_kv_cache_data(groups=[group]))
        spec = resp.groups[0]
        assert isinstance(spec, FullAttentionGroupSpec)
        assert spec.sliding_window is None
        assert spec.attention_chunk_size is None

    def test_full_attention_optional_fields_set(self):
        group = _full_attention_dict(sliding_window=2048, attention_chunk_size=512)
        resp = _build_kv_cache_response(_kv_cache_data(groups=[group]))
        spec = resp.groups[0]
        assert isinstance(spec, FullAttentionGroupSpec)
        assert spec.sliding_window == 2048
        assert spec.attention_chunk_size == 512

    def test_mla_attention_cache_dtype_str(self):
        group = _mla_attention_dict(cache_dtype_str="float8_e4m3fn")
        resp = _build_kv_cache_response(_kv_cache_data(groups=[group]))
        spec = resp.groups[0]
        assert isinstance(spec, MLAAttentionGroupSpec)
        assert spec.cache_dtype_str == "float8_e4m3fn"

    def test_mla_attention_cache_dtype_str_none(self):
        group = _mla_attention_dict(cache_dtype_str=None)
        resp = _build_kv_cache_response(_kv_cache_data(groups=[group]))
        spec = resp.groups[0]
        assert isinstance(spec, MLAAttentionGroupSpec)
        assert spec.cache_dtype_str is None

    def test_sliding_window_size(self):
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_sliding_window_dict(sliding_window=8192)])
        )
        spec = resp.groups[0]
        assert isinstance(spec, SlidingWindowGroupSpec)
        assert spec.sliding_window == 8192

    def test_chunked_local_attention_chunk_size(self):
        resp = _build_kv_cache_response(
            _kv_cache_data(
                groups=[_chunked_local_attention_dict(attention_chunk_size=4096)]
            )
        )
        spec = resp.groups[0]
        assert isinstance(spec, ChunkedLocalAttentionGroupSpec)
        assert spec.attention_chunk_size == 4096

    def test_mamba_shapes_and_dtypes(self):
        shapes = [[32, 256], [32, 128]]
        dtypes = ["float32", "float16"]
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_mamba_dict(shapes=shapes, dtypes=dtypes)])
        )
        spec = resp.groups[0]
        assert isinstance(spec, MambaGroupSpec)
        assert spec.shapes == shapes
        assert spec.dtypes == dtypes

    def test_mamba_type_and_cache_mode(self):
        resp = _build_kv_cache_response(
            _kv_cache_data(
                groups=[_mamba_dict(mamba_type="mamba1", mamba_cache_mode="chunked")]
            )
        )
        spec = resp.groups[0]
        assert isinstance(spec, MambaGroupSpec)
        assert spec.mamba_type == "mamba1"
        assert spec.mamba_cache_mode == "chunked"

    def test_sink_full_attention_sink_len(self):
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_sink_full_attention_dict(sink_len=8)])
        )
        spec = resp.groups[0]
        assert isinstance(spec, SinkFullAttentionGroupSpec)
        assert spec.sink_len == 8

    def test_sink_full_attention_sink_len_none(self):
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_sink_full_attention_dict(sink_len=None)])
        )
        spec = resp.groups[0]
        assert isinstance(spec, SinkFullAttentionGroupSpec)
        assert spec.sink_len is None

    def test_uniform_type_layer_specs(self):
        layer_specs = [{"head_size": 128}, {"head_size": 64}]
        resp = _build_kv_cache_response(
            _kv_cache_data(groups=[_uniform_type_dict(layer_specs=layer_specs)])
        )
        spec = resp.groups[0]
        assert isinstance(spec, UniformTypeGroupSpec)
        assert spec.layer_specs == layer_specs


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetKVCacheEndpoint:
    def test_returns_200_with_data(self):
        with _make_client(_kv_cache_data()) as client:
            resp = client.get("/v1/server/kv-cache")
        assert resp.status_code == 200

    def test_returns_200_when_kv_cache_config_is_none(self):
        with _make_client(None) as client:
            resp = client.get("/v1/server/kv-cache")
        assert resp.status_code == 200

    def test_content_type_is_json(self):
        with _make_client(_kv_cache_data()) as client:
            resp = client.get("/v1/server/kv-cache")
        assert "application/json" in resp.headers["content-type"]

    def test_response_parses_as_kv_cache_runtime_info(self):
        with _make_client(_kv_cache_data()) as client:
            resp = client.get("/v1/server/kv-cache")
        parsed = KVCacheRuntimeInfo.model_validate(resp.json())
        assert parsed.num_gpu_blocks == 1024

    def test_top_level_keys_present(self):
        with _make_client(_kv_cache_data()) as client:
            resp = client.get("/v1/server/kv-cache")
        data = resp.json()
        assert set(data.keys()) == {
            "kv_cache_size_tokens",
            "max_concurrency",
            "num_gpu_blocks",
            "num_cpu_blocks",
            "groups",
        }

    def test_null_state_all_fields_null_except_groups(self):
        with _make_client(None) as client:
            resp = client.get("/v1/server/kv-cache")
        data = resp.json()
        assert data["kv_cache_size_tokens"] is None
        assert data["max_concurrency"] is None
        assert data["num_gpu_blocks"] is None
        assert data["num_cpu_blocks"] is None
        assert data["groups"] == []

    def test_capacity_fields_correct_values(self):
        with _make_client(
            _kv_cache_data(
                kv_cache_size_tokens=16384,
                max_concurrency=0.5,
                num_gpu_blocks=1024,
                num_cpu_blocks=256,
            )
        ) as client:
            resp = client.get("/v1/server/kv-cache")
        data = resp.json()
        assert data["kv_cache_size_tokens"] == 16384
        assert data["max_concurrency"] == pytest.approx(0.5)
        assert data["num_gpu_blocks"] == 1024
        assert data["num_cpu_blocks"] == 256

    def test_groups_serialized_correctly(self):
        groups = [
            _full_attention_dict(group_id=0),
            _mamba_dict(group_id=1),
        ]
        with _make_client(_kv_cache_data(groups=groups)) as client:
            resp = client.get("/v1/server/kv-cache")
        data = resp.json()["groups"]
        assert len(data) == 2
        assert data[0]["spec_type"] == "FullAttentionSpec"
        assert data[1]["spec_type"] == "MambaSpec"

    def test_group_keys_for_full_attention(self):
        with _make_client(_kv_cache_data(groups=[_full_attention_dict()])) as client:
            resp = client.get("/v1/server/kv-cache")
        group = resp.json()["groups"][0]
        assert "group_id" in group
        assert "spec_type" in group
        assert "layer_names" in group
        assert "block_size" in group
        assert "page_size_bytes" in group
        assert "num_kv_heads" in group
        assert "head_size" in group
        assert "head_size_v" in group
        assert "dtype" in group

    def test_group_keys_for_mamba(self):
        with _make_client(_kv_cache_data(groups=[_mamba_dict()])) as client:
            resp = client.get("/v1/server/kv-cache")
        group = resp.json()["groups"][0]
        assert "shapes" in group
        assert "dtypes" in group
        assert "mamba_type" in group
        assert "mamba_cache_mode" in group

    def test_wrong_method_returns_405(self):
        with _make_client(_kv_cache_data()) as client:
            resp = client.post("/v1/server/kv-cache")
        assert resp.status_code == 405

    def test_missing_state_returns_500(self):
        app = FastAPI()
        attach_router(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/server/kv-cache")
        assert resp.status_code == 500

    def test_full_response_full_attention_model(self):
        """Integration-style check: full data round-trips through the HTTP layer."""
        groups = [_full_attention_dict(group_id=0)]
        with _make_client(
            _kv_cache_data(
                kv_cache_size_tokens=16384,
                max_concurrency=0.5,
                num_gpu_blocks=1024,
                num_cpu_blocks=0,
                groups=groups,
            )
        ) as client:
            resp = client.get("/v1/server/kv-cache")
        data = resp.json()

        assert data["kv_cache_size_tokens"] == 16384
        assert data["max_concurrency"] == pytest.approx(0.5)
        assert data["num_gpu_blocks"] == 1024
        assert data["num_cpu_blocks"] == 0

        g = data["groups"][0]
        assert g["group_id"] == 0
        assert g["spec_type"] == "FullAttentionSpec"
        assert g["layer_names"] == _BASE_GROUP["layer_names"]
        assert g["block_size"] == 16
        assert g["page_size_bytes"] == 131072
        assert g["num_kv_heads"] == 8
        assert g["head_size"] == 128
        assert g["head_size_v"] == 128
        assert g["dtype"] == "bfloat16"

    def test_full_response_hybrid_model(self):
        """Hybrid model (attention + mamba) round-trips through the HTTP layer."""
        groups = [
            _full_attention_dict(group_id=0),
            _mamba_dict(group_id=1),
        ]
        with _make_client(
            _kv_cache_data(
                kv_cache_size_tokens=8192,
                max_concurrency=0.25,
                num_gpu_blocks=512,
                num_cpu_blocks=0,
                groups=groups,
            )
        ) as client:
            resp = client.get("/v1/server/kv-cache")
        data = resp.json()

        assert data["kv_cache_size_tokens"] == 8192
        assert len(data["groups"]) == 2

        mamba = data["groups"][1]
        assert mamba["spec_type"] == "MambaSpec"
        assert mamba["mamba_type"] == "mamba2"
        assert mamba["shapes"] == [[16, 128], [16, 64]]
        assert mamba["dtypes"] == ["float32", "float32"]


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


class TestRouterRegistration:
    def test_attach_router_registers_route(self):
        app = FastAPI()
        attach_router(app)
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/v1/server/kv-cache" in routes

    def test_router_has_get_method(self):
        assert any(
            r.path == "/v1/server/kv-cache"  # type: ignore[attr-defined]
            and "GET" in r.methods  # type: ignore[attr-defined]
            for r in router.routes
        )


# ---------------------------------------------------------------------------
# Handler isolation tests (mock request)
# ---------------------------------------------------------------------------


class TestHandlerIsolation:
    @pytest.mark.asyncio
    async def test_with_kv_cache_state(self):
        """Call the handler directly using a fully mocked request."""
        from vllm.entrypoints.serve.kv_cache.api_router import get_kv_cache

        mock_request = MagicMock()
        mock_request.app.state.kv_cache_config = _kv_cache_data()

        result = await get_kv_cache(mock_request)

        assert isinstance(result, KVCacheRuntimeInfo)
        assert result.num_gpu_blocks == 1024
        assert len(result.groups) == 1

    @pytest.mark.asyncio
    async def test_with_none_state(self):
        """Handler returns empty KVCacheRuntimeInfo when state is None."""
        from vllm.entrypoints.serve.kv_cache.api_router import get_kv_cache

        mock_request = MagicMock()
        mock_request.app.state.kv_cache_config = None

        result = await get_kv_cache(mock_request)

        assert isinstance(result, KVCacheRuntimeInfo)
        assert result.groups == []


# ---------------------------------------------------------------------------
# Pydantic protocol model tests
# ---------------------------------------------------------------------------


class TestProtocolModels:
    def test_kv_cache_runtime_info_defaults(self):
        info = KVCacheRuntimeInfo()
        assert info.kv_cache_size_tokens is None
        assert info.max_concurrency is None
        assert info.num_gpu_blocks is None
        assert info.num_cpu_blocks is None
        assert info.groups == []

    def test_kv_cache_runtime_info_roundtrip(self):
        original = KVCacheRuntimeInfo(
            kv_cache_size_tokens=16384,
            max_concurrency=0.5,
            num_gpu_blocks=1024,
            num_cpu_blocks=256,
            groups=[],
        )
        restored = KVCacheRuntimeInfo.model_validate(original.model_dump())
        assert restored == original

    def test_kv_cache_runtime_info_json_schema_has_expected_properties(self):
        schema = KVCacheRuntimeInfo.model_json_schema()
        props = schema.get("properties", {})
        assert "kv_cache_size_tokens" in props
        assert "max_concurrency" in props
        assert "num_gpu_blocks" in props
        assert "num_cpu_blocks" in props
        assert "groups" in props

    def test_full_attention_roundtrip(self):
        spec = FullAttentionGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.self_attn"],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype="bfloat16",
        )
        restored = FullAttentionGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_full_attention_optional_fields_default_none(self):
        spec = FullAttentionGroupSpec(
            group_id=0,
            layer_names=[],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype="bfloat16",
        )
        assert spec.sliding_window is None
        assert spec.attention_chunk_size is None

    def test_mla_attention_roundtrip(self):
        spec = MLAAttentionGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.self_attn"],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            head_size_v=64,
            dtype="bfloat16",
            cache_dtype_str="float8_e4m3fn",
        )
        restored = MLAAttentionGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_mla_attention_cache_dtype_str_optional(self):
        spec = MLAAttentionGroupSpec(
            group_id=0,
            layer_names=[],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype="bfloat16",
        )
        assert spec.cache_dtype_str is None

    def test_sliding_window_roundtrip(self):
        spec = SlidingWindowGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.self_attn"],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            dtype="float16",
            sliding_window=4096,
        )
        restored = SlidingWindowGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_chunked_local_attention_roundtrip(self):
        spec = ChunkedLocalAttentionGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.self_attn"],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            dtype="float16",
            attention_chunk_size=2048,
        )
        restored = ChunkedLocalAttentionGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_mamba_roundtrip(self):
        spec = MambaGroupSpec(
            group_id=1,
            layer_names=["model.layers.0.mamba"],
            block_size=1,
            page_size_bytes=4096,
            shapes=[[16, 128]],
            dtypes=["float32"],
            mamba_type="mamba2",
            mamba_cache_mode="none",
        )
        restored = MambaGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_cross_attention_roundtrip(self):
        spec = CrossAttentionGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.cross_attn"],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            dtype="bfloat16",
        )
        restored = CrossAttentionGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_sink_full_attention_roundtrip(self):
        spec = SinkFullAttentionGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.self_attn"],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype="bfloat16",
            sliding_window=2048,
            sink_len=4,
        )
        restored = SinkFullAttentionGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    def test_sink_full_attention_sink_len_optional(self):
        spec = SinkFullAttentionGroupSpec(
            group_id=0,
            layer_names=[],
            block_size=16,
            page_size_bytes=131072,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype="bfloat16",
        )
        assert spec.sink_len is None

    def test_uniform_type_roundtrip(self):
        spec = UniformTypeGroupSpec(
            group_id=0,
            layer_names=["model.layers.0.self_attn"],
            block_size=16,
            page_size_bytes=131072,
            layer_specs=[{"head_size": 128}, {"head_size": 64}],
        )
        restored = UniformTypeGroupSpec.model_validate(spec.model_dump())
        assert restored == spec

    @pytest.mark.parametrize(
        "spec_type,model_cls",
        [
            ("FullAttentionSpec", FullAttentionGroupSpec),
            ("MLAAttentionSpec", MLAAttentionGroupSpec),
            ("SlidingWindowSpec", SlidingWindowGroupSpec),
            ("ChunkedLocalAttentionSpec", ChunkedLocalAttentionGroupSpec),
            ("MambaSpec", MambaGroupSpec),
            ("CrossAttentionSpec", CrossAttentionGroupSpec),
            ("SinkFullAttentionSpec", SinkFullAttentionGroupSpec),
            ("UniformTypeKVCacheSpecs", UniformTypeGroupSpec),
        ],
    )
    def test_spec_type_literal_matches_class(self, spec_type, model_cls):
        assert model_cls.model_fields["spec_type"].default == spec_type
