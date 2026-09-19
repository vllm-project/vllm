# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GET /v1/server/config.

These tests do not spin up a real vLLM server or load model weights.
They exercise:
  - _build_response() directly (pure function, fast, no I/O)
  - The FastAPI route via TestClient (HTTP stack, no engine)
  - The Pydantic protocol models (schema correctness)
"""

from unittest.mock import MagicMock

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.config.api_router import _build_response, attach_router
from vllm.entrypoints.serve.config.protocol import (
    FeaturesInfo,
    KVCacheInfo,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    ServerConfigResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_config(
    *,
    model_name: str = "meta-llama/Llama-3.1-8B",
    served_model_name: str = "meta-llama/Llama-3.1-8B",
    model_dtype: torch.dtype = torch.bfloat16,
    quantization: str | None = None,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    cache_dtype: str = "auto",
    enable_prefix_caching: bool = True,
    max_num_seqs: int = 128,
    max_num_batched_tokens: int = 2048,
    enable_chunked_prefill: bool = True,
    policy: str = "fcfs",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    disable_hybrid_kv_cache_manager: bool = False,
    speculative_config=None,
    lora_config=None,
) -> MagicMock:
    """Build a MagicMock VllmConfig with fully-controlled sub-config attributes."""
    model_cfg = MagicMock()
    model_cfg.model = model_name
    model_cfg.served_model_name = served_model_name
    model_cfg.dtype = model_dtype
    model_cfg.quantization = quantization
    model_cfg.max_model_len = max_model_len

    cache_cfg = MagicMock()
    cache_cfg.gpu_memory_utilization = gpu_memory_utilization
    cache_cfg.cache_dtype = cache_dtype
    cache_cfg.enable_prefix_caching = enable_prefix_caching

    scheduler_cfg = MagicMock()
    scheduler_cfg.max_num_seqs = max_num_seqs
    scheduler_cfg.max_num_batched_tokens = max_num_batched_tokens
    scheduler_cfg.enable_chunked_prefill = enable_chunked_prefill
    scheduler_cfg.policy = policy
    scheduler_cfg.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager

    parallel_cfg = MagicMock()
    parallel_cfg.tensor_parallel_size = tensor_parallel_size
    parallel_cfg.pipeline_parallel_size = pipeline_parallel_size
    parallel_cfg.data_parallel_size = data_parallel_size
    parallel_cfg.data_parallel_rank = data_parallel_rank

    vllm_config = MagicMock()
    vllm_config.model_config = model_cfg
    vllm_config.cache_config = cache_cfg
    vllm_config.scheduler_config = scheduler_cfg
    vllm_config.parallel_config = parallel_cfg
    vllm_config.speculative_config = speculative_config
    vllm_config.lora_config = lora_config

    return vllm_config


def _make_test_app(
    vllm_config: MagicMock,
    served_model_name: list[str] | None = None,
) -> FastAPI:
    """Return a minimal FastAPI app with the config router and state injected."""
    app = FastAPI()
    attach_router(app)

    args = MagicMock()
    args.served_model_name = served_model_name

    app.state.vllm_config = vllm_config
    app.state.args = args
    return app


# ---------------------------------------------------------------------------
# _build_response unit tests
# ---------------------------------------------------------------------------


class TestBuildResponseModelSection:
    def test_model_name_is_first_served_name(self):
        cfg = _make_vllm_config(model_name="base-model")
        resp = _build_response(cfg, ["alias-a", "alias-b"])
        assert resp.model.name == "alias-a"

    def test_model_served_names_matches_input(self):
        cfg = _make_vllm_config()
        resp = _build_response(cfg, ["m1", "m2", "m3"])
        assert resp.model.served_names == ["m1", "m2", "m3"]

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.bfloat16, "bfloat16"),
            (torch.float16, "float16"),
            (torch.float32, "float32"),
        ],
    )
    def test_model_dtype_format(self, dtype, expected):
        cfg = _make_vllm_config(model_dtype=dtype)
        resp = _build_response(cfg, ["m"])
        assert resp.model.dtype == expected

    def test_model_quantization_none(self):
        cfg = _make_vllm_config(quantization=None)
        resp = _build_response(cfg, ["m"])
        assert resp.model.quantization is None

    @pytest.mark.parametrize("quant", ["awq", "gptq", "fp8"])
    def test_model_quantization_string(self, quant):
        cfg = _make_vllm_config(quantization=quant)
        resp = _build_response(cfg, ["m"])
        assert resp.model.quantization == quant

    def test_model_max_model_len(self):
        cfg = _make_vllm_config(max_model_len=131072)
        resp = _build_response(cfg, ["m"])
        assert resp.model.max_model_len == 131072


class TestBuildResponseKVCacheSection:
    def test_gpu_memory_utilization(self):
        cfg = _make_vllm_config(gpu_memory_utilization=0.85)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.gpu_memory_utilization == pytest.approx(0.85)

    @pytest.mark.parametrize(
        "model_dtype,expected",
        [
            (torch.bfloat16, "bfloat16"),
            (torch.float16, "float16"),
        ],
    )
    def test_cache_dtype_auto_resolves(self, model_dtype, expected):
        # "auto" must be resolved to the concrete model dtype, never left as "auto"
        cfg = _make_vllm_config(cache_dtype="auto", model_dtype=model_dtype)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.dtype == expected

    def test_enable_prefix_caching_true(self):
        cfg = _make_vllm_config(enable_prefix_caching=True)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.enable_prefix_caching is True

    def test_enable_prefix_caching_false(self):
        cfg = _make_vllm_config(enable_prefix_caching=False)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.enable_prefix_caching is False


class TestBuildResponseSchedulerSection:
    def test_max_num_seqs(self):
        cfg = _make_vllm_config(max_num_seqs=256)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.max_num_seqs == 256

    def test_max_num_batched_tokens(self):
        cfg = _make_vllm_config(max_num_batched_tokens=4096)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.max_num_batched_tokens == 4096

    def test_enable_chunked_prefill_true(self):
        cfg = _make_vllm_config(enable_chunked_prefill=True)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.enable_chunked_prefill is True

    def test_enable_chunked_prefill_false(self):
        cfg = _make_vllm_config(enable_chunked_prefill=False)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.enable_chunked_prefill is False

    @pytest.mark.parametrize("policy", ["fcfs", "priority"])
    def test_scheduler_policy(self, policy):
        cfg = _make_vllm_config(policy=policy)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.policy == policy


class TestBuildResponseParallelismSection:
    def test_defaults_single_device(self):
        cfg = _make_vllm_config(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
        )
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.tensor_parallel_size == 1
        assert resp.parallelism.pipeline_parallel_size == 1
        assert resp.parallelism.data_parallel_size == 1
        assert resp.parallelism.data_parallel_rank == 0

    def test_tensor_parallel(self):
        cfg = _make_vllm_config(tensor_parallel_size=4)
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.tensor_parallel_size == 4

    def test_pipeline_parallel(self):
        cfg = _make_vllm_config(pipeline_parallel_size=2)
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.pipeline_parallel_size == 2

    def test_data_parallel_with_rank(self):
        cfg = _make_vllm_config(data_parallel_size=4, data_parallel_rank=3)
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.data_parallel_size == 4
        assert resp.parallelism.data_parallel_rank == 3


class TestBuildResponseFeaturesSection:
    def test_all_features_disabled_by_default(self):
        cfg = _make_vllm_config(
            speculative_config=None,
            lora_config=None,
            disable_hybrid_kv_cache_manager=False,
        )
        resp = _build_response(cfg, ["m"])
        assert resp.features.speculative_decoding is False
        assert resp.features.lora is False
        assert resp.features.hma is True  # hma = not False

    def test_speculative_decoding_enabled(self):
        spec_cfg = MagicMock()  # any non-None value enables it
        cfg = _make_vllm_config(speculative_config=spec_cfg)
        resp = _build_response(cfg, ["m"])
        assert resp.features.speculative_decoding is True

    def test_speculative_decoding_disabled(self):
        cfg = _make_vllm_config(speculative_config=None)
        resp = _build_response(cfg, ["m"])
        assert resp.features.speculative_decoding is False

    def test_lora_enabled(self):
        lora_cfg = MagicMock()
        cfg = _make_vllm_config(lora_config=lora_cfg)
        resp = _build_response(cfg, ["m"])
        assert resp.features.lora is True

    def test_lora_disabled(self):
        cfg = _make_vllm_config(lora_config=None)
        resp = _build_response(cfg, ["m"])
        assert resp.features.lora is False

    def test_hma_enabled_when_manager_not_disabled(self):
        cfg = _make_vllm_config(disable_hybrid_kv_cache_manager=False)
        resp = _build_response(cfg, ["m"])
        assert resp.features.hma is True

    def test_hma_disabled_when_manager_disabled(self):
        cfg = _make_vllm_config(disable_hybrid_kv_cache_manager=True)
        resp = _build_response(cfg, ["m"])
        assert resp.features.hma is False

    def test_all_features_enabled(self):
        cfg = _make_vllm_config(
            speculative_config=MagicMock(),
            lora_config=MagicMock(),
            disable_hybrid_kv_cache_manager=False,
        )
        resp = _build_response(cfg, ["m"])
        assert resp.features.speculative_decoding is True
        assert resp.features.lora is True
        assert resp.features.hma is True


class TestBuildResponseReturnType:
    def test_returns_server_config_response(self):
        cfg = _make_vllm_config()
        resp = _build_response(cfg, ["m"])
        assert isinstance(resp, ServerConfigResponse)

    def test_all_sub_objects_present(self):
        cfg = _make_vllm_config()
        resp = _build_response(cfg, ["m"])
        assert isinstance(resp.model, ModelInfo)
        assert isinstance(resp.kv_cache, KVCacheInfo)
        assert isinstance(resp.scheduler, SchedulerInfo)
        assert isinstance(resp.parallelism, ParallelismInfo)
        assert isinstance(resp.features, FeaturesInfo)


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetServerConfigEndpoint:
    def test_response_parses_as_server_config_response(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["my-model"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        parsed = ServerConfigResponse.model_validate(resp.json())
        assert parsed.model.name == "my-model"

    def test_served_names_from_args_single(self):
        cfg = _make_vllm_config()
        app = _make_test_app(cfg, served_model_name=["my-alias"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        data = resp.json()
        assert data["model"]["name"] == "my-alias"
        assert data["model"]["served_names"] == ["my-alias"]

    def test_served_names_from_args_multiple(self):
        cfg = _make_vllm_config()
        app = _make_test_app(cfg, served_model_name=["alias-a", "alias-b"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        data = resp.json()
        assert data["model"]["name"] == "alias-a"
        assert data["model"]["served_names"] == ["alias-a", "alias-b"]

    def test_served_names_fallback_from_model_config(self):
        # When args.served_model_name is falsy, fall back to
        # model_config.served_model_name
        cfg = _make_vllm_config(served_model_name="model-from-config")
        app = _make_test_app(cfg, served_model_name=None)
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        data = resp.json()
        assert data["model"]["name"] == "model-from-config"
        assert data["model"]["served_names"] == ["model-from-config"]

    def test_content_type_is_json(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        assert "application/json" in resp.headers["content-type"]

    def test_top_level_keys_present(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        data = resp.json()
        assert set(data.keys()) == {
            "model",
            "kv_cache",
            "scheduler",
            "parallelism",
            "features",
        }

    def test_model_section_keys(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        model = resp.json()["model"]
        assert set(model.keys()) == {
            "name",
            "served_names",
            "dtype",
            "quantization",
            "max_model_len",
        }

    def test_kv_cache_section_keys(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        kv = resp.json()["kv_cache"]
        assert set(kv.keys()) == {
            "gpu_memory_utilization",
            "dtype",
            "enable_prefix_caching",
        }

    def test_scheduler_section_keys(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        sched = resp.json()["scheduler"]
        assert set(sched.keys()) == {
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "policy",
        }

    def test_parallelism_section_keys(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        par = resp.json()["parallelism"]
        assert set(par.keys()) == {
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "data_parallel_rank",
        }

    def test_features_section_keys(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        feat = resp.json()["features"]
        assert set(feat.keys()) == {"speculative_decoding", "lora", "hma"}

    def test_wrong_method_returns_405(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.post("/v1/server/config")
        assert resp.status_code == 405

    def test_missing_state_returns_500(self):
        app = FastAPI()
        attach_router(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/server/config")
        assert resp.status_code == 500

    def test_full_response_values(self):
        cfg = _make_vllm_config(
            model_dtype=torch.bfloat16,
            quantization=None,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
            max_num_seqs=128,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            policy="fcfs",
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=2,
            data_parallel_rank=0,
            disable_hybrid_kv_cache_manager=False,
            speculative_config=None,
            lora_config=None,
        )
        app = _make_test_app(cfg, served_model_name=["my-model", "my-model-alias"])
        with TestClient(app) as client:
            resp = client.get("/v1/server/config")
        data = resp.json()

        assert data["model"]["name"] == "my-model"
        assert data["model"]["served_names"] == ["my-model", "my-model-alias"]
        assert data["model"]["dtype"] == "bfloat16"
        assert data["model"]["quantization"] is None
        assert data["model"]["max_model_len"] == 32768

        assert data["kv_cache"]["gpu_memory_utilization"] == pytest.approx(0.9)
        assert data["kv_cache"]["dtype"] == "bfloat16"
        assert data["kv_cache"]["enable_prefix_caching"] is True

        assert data["scheduler"]["max_num_seqs"] == 128
        assert data["scheduler"]["max_num_batched_tokens"] == 2048
        assert data["scheduler"]["enable_chunked_prefill"] is True
        assert data["scheduler"]["policy"] == "fcfs"

        assert data["parallelism"]["tensor_parallel_size"] == 4
        assert data["parallelism"]["pipeline_parallel_size"] == 1
        assert data["parallelism"]["data_parallel_size"] == 2
        assert data["parallelism"]["data_parallel_rank"] == 0

        assert data["features"]["speculative_decoding"] is False
        assert data["features"]["lora"] is False
        assert data["features"]["hma"] is True


# ---------------------------------------------------------------------------
# Pydantic protocol model tests
# ---------------------------------------------------------------------------


class TestProtocolModels:
    def test_server_config_response_roundtrip(self):
        original = ServerConfigResponse(
            model=ModelInfo(
                name="llama",
                served_names=["llama"],
                dtype="bfloat16",
                quantization=None,
                max_model_len=4096,
            ),
            kv_cache=KVCacheInfo(
                gpu_memory_utilization=0.9,
                dtype="bfloat16",
                enable_prefix_caching=True,
            ),
            scheduler=SchedulerInfo(
                max_num_seqs=64,
                max_num_batched_tokens=1024,
                enable_chunked_prefill=True,
                policy="fcfs",
            ),
            parallelism=ParallelismInfo(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                data_parallel_rank=0,
            ),
            features=FeaturesInfo(
                speculative_decoding=False,
                lora=False,
                hma=True,
            ),
        )
        as_dict = original.model_dump()
        restored = ServerConfigResponse.model_validate(as_dict)
        assert restored == original

    def test_quantization_null_serialization(self):
        model_info = ModelInfo(
            name="m",
            served_names=["m"],
            dtype="float16",
            quantization=None,
            max_model_len=2048,
        )
        dumped = model_info.model_dump()
        assert dumped["quantization"] is None

    def test_quantization_string_serialization(self):
        model_info = ModelInfo(
            name="m",
            served_names=["m"],
            dtype="float16",
            quantization="gptq",
            max_model_len=2048,
        )
        assert model_info.model_dump()["quantization"] == "gptq"

    def test_features_info_all_bool(self):
        feat = FeaturesInfo(speculative_decoding=True, lora=False, hma=True)
        dumped = feat.model_dump()
        assert dumped == {"speculative_decoding": True, "lora": False, "hma": True}

    def test_server_config_response_json_schema_has_required_fields(self):
        schema = ServerConfigResponse.model_json_schema()
        assert "model" in schema["properties"]
        assert "kv_cache" in schema["properties"]
        assert "scheduler" in schema["properties"]
        assert "parallelism" in schema["properties"]
        assert "features" in schema["properties"]
