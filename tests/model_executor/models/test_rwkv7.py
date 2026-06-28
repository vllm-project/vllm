# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.model_executor.models.rwkv7 as rwkv7
import vllm.v1.worker.gpu.model_states.rwkv as rwkv_state
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.config import RWKV7ForCausalLMConfig
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.model_states.rwkv import RWKV7ModelState


def test_rwkv7_cuda_ops_match_torch_reference():
    if (
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("CUDA is required for RWKV7 custom op numerical test")

    import vllm.rwkv7_ops  # noqa: F401

    eps = 1e-5
    x = torch.tensor(
        [
            [1.0, -2.0, 0.5, 4.0],
            [-3.5, 0.25, 2.0, -0.75],
        ],
        dtype=torch.float16,
        device="cuda",
    ).contiguous()
    weight = torch.tensor(
        [0.5, -1.25, 2.0, 0.75], dtype=torch.float16, device="cuda"
    ).contiguous()
    bias = torch.tensor(
        [0.125, -0.5, 1.0, -1.5], dtype=torch.float16, device="cuda"
    ).contiguous()

    y = torch.ops.rwkv7_v3a_ops.layer_norm_f16(x, weight, bias, eps)
    z = torch.ops.rwkv7_fast_ops_fp16.relu_square(x)
    torch.accelerator.synchronize()

    expected_y = F.layer_norm(
        x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
    ).to(torch.float16)
    expected_z = torch.relu(x).square()
    torch.testing.assert_close(y, expected_y, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(z, expected_z, rtol=0, atol=0)


def test_rwkv7_cuda_ops_register_expected_namespaces_and_schemas():
    if (
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("CUDA is required for RWKV7 custom op schema smoke")

    import vllm.rwkv7_ops  # noqa: F401

    expected_ops = {
        "rwkv7_v3a_ops": [
            "layer_norm_f16",
            "add_layer_norm_tmix_mix6_f16",
            "advance_i32",
        ],
        "rwkv7_fast_ops_fp16": [
            "tmix_mix6",
            "relu_square",
            "act_sigmoid",
        ],
        "rwkv7_wkv_fp16_v2": [
            "wkv_seq",
            "wkv_seq_w0",
            "wkv_one",
        ],
        "rwkv7_wkv_fp32_v2": [
            "forward",
            "forward_seq",
            "forward_small",
            "forward_block",
        ],
    }

    for namespace, op_names in expected_ops.items():
        ops = getattr(torch.ops, namespace)
        for op_name in op_names:
            op = getattr(ops, op_name)
            assert op._schemas
            assert "" in op._schemas


def _new_request(req_id: str) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=(),
        num_computed_tokens=0,
        lora_request=None,
    )


def _new_rwkv7_forward_test_model(**attrs: Any) -> RWKV7ForCausalLM:
    model = object.__new__(RWKV7ForCausalLM)
    nn.Module.__init__(model)
    model._is_pp_first_rank = lambda: True
    model._is_pp_last_rank = lambda: True
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def _new_rwkv7_for_weight_tests() -> RWKV7ForCausalLM:
    model = _new_rwkv7_forward_test_model()
    model.z = {"old": torch.tensor([1])}
    model.emb_cpu = True
    model.emb_cache = {(1, 1): (torch.empty(1), torch.empty(1))}
    model._dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)
    return model


def _new_default_loader_for_weight_tests(
    load_format: str = "auto",
) -> DefaultModelLoader:
    return DefaultModelLoader(
        SimpleNamespace(
            load_format=load_format,
            model_loader_extra_config={},
            download_dir=None,
            ignore_patterns=None,
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
            safetensors_prefetch_num_threads=1,
            safetensors_prefetch_block_size=1024,
            pt_load_map_location="cpu",
        )
    )


def test_rwkv7_registry_module_exports_model():
    assert rwkv7.RWKV7ForCausalLM is RWKV7ForCausalLM
    assert RWKV7ForCausalLM.__module__.endswith("rwkv7")


def _rwkv7_vllm_config(
    *,
    enforce_eager: bool,
    compilation_mode: CompilationMode = CompilationMode.NONE,
) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=CompilationConfig(mode=compilation_mode),
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            hf_config=SimpleNamespace(
                hidden_size=64,
                vocab_size=128,
                head_size=64,
                num_hidden_layers=1,
            ),
        ),
    )


def _new_rwkv7_model_state(
    *,
    max_num_reqs: int = 4,
    num_hidden_layers: int = 1,
    hidden_size: int = 64,
    head_size: int = 64,
    num_attention_heads: int = 1,
) -> RWKV7ModelState:
    hf_config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        head_size=head_size,
        num_attention_heads=num_attention_heads,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
    )
    return RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )


def _assert_same_storage_view(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()


def test_rwkv7_rejects_torch_compile():
    with pytest.raises(ValueError, match="RWKV7 does not support torch.compile"):
        RWKV7ForCausalLM(
            vllm_config=_rwkv7_vllm_config(
                enforce_eager=False,
                compilation_mode=CompilationMode.VLLM_COMPILE,
            )
        )


def test_rwkv7_init_preserves_process_wide_torch_state(monkeypatch):
    monkeypatch.setattr(rwkv7, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rwkv7, "get_tensor_model_parallel_rank", lambda: 0)

    old_grad_enabled = torch.is_grad_enabled()
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    old_cuda_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_grad_enabled(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        RWKV7ForCausalLM(vllm_config=_rwkv7_vllm_config(enforce_eager=False))

        assert torch.is_grad_enabled() is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_grad_enabled(old_grad_enabled)
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = old_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_matmul_allow_tf32
        torch.set_float32_matmul_precision(old_matmul_precision)


def test_rwkv7_declares_pipeline_parallel_support():
    assert supports_pp(RWKV7ForCausalLM)


def test_rwkv7_raw_pth_file_uses_single_weight_file(tmp_path):
    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()
    loader = _new_default_loader_for_weight_tests()

    _, weight_files, use_safetensors = loader._prepare_weights(
        str(checkpoint),
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert weight_files == [str(checkpoint)]
    assert use_safetensors is False


def test_rwkv7_raw_pth_file_allows_hf_load_format(tmp_path):
    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()
    loader = _new_default_loader_for_weight_tests(load_format="hf")

    _, weight_files, use_safetensors = loader._prepare_weights(
        str(checkpoint),
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert weight_files == [str(checkpoint)]
    assert use_safetensors is False


def test_rwkv7_raw_pth_url_downloads_single_weight_file(tmp_path, monkeypatch):
    downloaded = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    downloaded.touch()
    calls: list[Any] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(
        "vllm.transformers_utils.configs.rwkv7.hf_hub_download",
        fake_hf_hub_download,
    )
    loader = _new_default_loader_for_weight_tests()

    _, weight_files, use_safetensors = loader._prepare_weights(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert calls == [
        {
            "repo_id": "BlinkDL/rwkv7-g1",
            "filename": "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
            "revision": "main",
            "cache_dir": None,
        }
    ]
    assert weight_files == [str(downloaded)]
    assert use_safetensors is False


def test_rwkv7_raw_pth_preprocess_validates_config_shape():
    model = _new_rwkv7_for_weight_tests()
    model.tp_size = 1
    model.tp_rank = 0
    model.org_vocab_size = 65536
    model.config = SimpleNamespace(
        hidden_size=2048,
        vocab_size=65536,
        head_size=64,
        num_hidden_layers=24,
    )
    weights = {
        "emb.weight": torch.empty(65536, 2048),
        "blocks.0.att.r_k": torch.empty(24, 64),
    }

    with pytest.raises(ValueError, match="RWKV7 config hidden_size"):
        RWKV7ForCausalLM._validate_raw_weight_shapes(model, weights)


@pytest.mark.parametrize(
    "compilation_mode",
    [
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ],
)
def test_rwkv7_config_rejects_torch_compile(compilation_mode):
    vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(mode=compilation_mode)
    )

    with pytest.raises(ValueError, match="RWKV7 does not support torch.compile"):
        RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)


def test_rwkv7_config_defaults_to_no_compile():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert (
        vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )


def test_rwkv7_config_disables_prefix_caching():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            max_num_scheduled_tokens=None,
        ),
        compilation_config=CompilationConfig(),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.enable_prefix_caching is False


def test_rwkv7_config_rejects_decode_budget_below_max_running_reqs():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            max_num_scheduled_tokens=2,
        ),
        compilation_config=CompilationConfig(),
    )

    with pytest.raises(ValueError, match="max_num_seqs"):
        RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    "cudagraph_mode",
    [None, CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_rwkv7_config_uses_decode_cudagraph(cudagraph_mode):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=cudagraph_mode,
        ),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert (
        vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )


def test_rwkv7_config_preserves_disabled_cudagraph():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_rwkv7_config_rejects_invalid_albatross_knob(monkeypatch):
    import vllm.model_executor.models.config as model_config

    monkeypatch.setattr(model_config.envs, "VLLM_RWKV7_RKV_MODE", "bad")
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(),
    )

    with pytest.raises(ValueError, match="VLLM_RWKV7_RKV_MODE"):
        RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)


def test_rwkv7_select_path_treats_batched_rkv_as_enabled(monkeypatch):
    monkeypatch.setattr(rwkv7, "RKV_MODE", "batched")
    monkeypatch.setattr(rwkv7, "ORIG_LINEAR_GROUPS", set())

    path = rwkv7.select_path(B=320, T=1)

    assert path.use_batched_rkv


def test_rwkv7_dummy_inputs_decode_capture_advertises_compact_rows():
    state = object.__new__(RWKV7ModelState)
    state.num_layers = 2
    state.max_num_reqs = 4
    state.hidden_size = 8
    state.num_heads = 2
    state.head_size = 4
    state.device = torch.device("cpu")
    state.shift_state = torch.zeros((2, 2, 4, 8), dtype=torch.float16)
    state.wkv_state = torch.zeros((2, 4, 2, 4, 4), dtype=torch.float32)
    state.elapsed = torch.zeros((4,), dtype=torch.int32)
    state.execution_idx_mapping = torch.arange(4, dtype=torch.int32)

    inputs = state.prepare_dummy_inputs(num_reqs=3, num_tokens=3)

    assert inputs["idx_mapping"].tolist() == [0, 1, 2]
    assert inputs["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert inputs["rwkv_decode_batch_size"] == 3
    assert inputs["rwkv_decode_rows"] == [0, 1, 2]
    assert inputs["rwkv_decode_token_positions"] == [0, 1, 2]


def test_rwkv7_dummy_inputs_decode_capture_uses_persistent_state_buffers():
    state = object.__new__(RWKV7ModelState)
    state.num_layers = 2
    state.max_num_reqs = 4
    state.hidden_size = 8
    state.num_heads = 2
    state.head_size = 4
    state.device = torch.device("cpu")
    state.shift_state = torch.ones((2, 2, 4, 8), dtype=torch.float16)
    state.wkv_state = torch.ones((2, 4, 2, 4, 4), dtype=torch.float32)
    state.elapsed = torch.ones((4,), dtype=torch.int32)

    inputs = state.prepare_dummy_inputs(num_reqs=3, num_tokens=3)

    assert inputs["idx_mapping"].tolist() == [0, 1, 2]
    assert inputs["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert inputs["shift_state"] is state.shift_state
    assert inputs["wkv_state"] is state.wkv_state
    assert inputs["elapsed"] is state.elapsed
    assert inputs["rwkv_decode_batch_size"] == 3
    assert inputs["rwkv_decode_rows"] == [0, 1, 2]
    assert inputs["rwkv_decode_token_positions"] == [0, 1, 2]


def test_rwkv7_embed_uses_cached_device_buffer_during_cuda_graph_capture(
    monkeypatch,
):
    model = object.__new__(RWKV7ForCausalLM)
    model.emb_cpu = True
    host = torch.empty((2, 4), dtype=torch.float16)
    dev = torch.empty((2, 1, 4), dtype=torch.float16)
    model.emb_cache = {(2, 1): (host, dev)}
    model.z = {"emb.weight": torch.empty((8, 4), dtype=torch.float16)}
    monkeypatch.setattr(rwkv7, "C", 4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch,
        "index_select",
        lambda *args, **kwargs: pytest.fail(
            "CUDA graph capture must not run CPU embedding lookup"
        ),
    )

    out = model.embed(torch.tensor([[1], [2]], dtype=torch.long))

    assert out is dev


def test_rwkv7_prepare_cudagraph_embedding_updates_cached_device_buffer(
    monkeypatch,
):
    model = object.__new__(RWKV7ForCausalLM)
    model.emb_cpu = True
    model.z = {
        "emb.weight": torch.arange(24, dtype=torch.float16).view(8, 3),
    }
    host = torch.empty((2, 3), dtype=torch.float16)
    dev = torch.empty((2, 1, 3), dtype=torch.float16)
    model.emb_cache = {(2, 1): (host, dev)}
    monkeypatch.setattr(rwkv7, "C", 3)

    model.prepare_cudagraph_embedding(torch.tensor([[1], [3]], dtype=torch.long))

    expected = model.z["emb.weight"][torch.tensor([1, 3])].view(2, 1, 3)
    torch.testing.assert_close(dev, expected)


def test_rwkv7_prepare_cudagraph_embedding_treats_flat_tokens_as_decode_batch(
    monkeypatch,
):
    model = object.__new__(RWKV7ForCausalLM)
    model.emb_cpu = True
    model.z = {
        "emb.weight": torch.arange(24, dtype=torch.float16).view(8, 3),
    }
    host = torch.empty((2, 3), dtype=torch.float16)
    decode_dev = torch.empty((2, 1, 3), dtype=torch.float16)
    wrong_dev = torch.full((1, 2, 3), -1, dtype=torch.float16)
    model.emb_cache = {
        (2, 1): (host, decode_dev),
        (1, 2): (torch.empty((2, 3), dtype=torch.float16), wrong_dev),
    }
    monkeypatch.setattr(rwkv7, "C", 3)

    model.prepare_cudagraph_embedding(torch.tensor([1, 3], dtype=torch.long))

    expected = model.z["emb.weight"][torch.tensor([1, 3])].view(2, 1, 3)
    torch.testing.assert_close(decode_dev, expected)
    torch.testing.assert_close(wrong_dev, torch.full_like(wrong_dev, -1))


def test_rwkv7_prepare_cudagraph_inputs_rejects_non_compact_decode_rows():
    model = object.__new__(RWKV7ForCausalLM)
    seen = []
    model.prepare_cudagraph_embedding = lambda tokens: seen.append(tokens.clone())

    with pytest.raises(RuntimeError, match="compact prefix rows"):
        model.prepare_cudagraph_inputs(
            {
                "input_ids": torch.tensor([10, 20], dtype=torch.int64),
                "rwkv_decode_batch_size": 2,
                "rwkv_decode_rows": [1, 0],
                "rwkv_decode_token_positions": [0, 1],
            }
        )

    assert seen == []


def test_rwkv7_prepare_cudagraph_inputs_uses_dense_view_for_compact_rows():
    model = object.__new__(RWKV7ForCausalLM)
    seen = []
    input_ids = torch.tensor([10, 20], dtype=torch.int64)
    model.prepare_cudagraph_embedding = lambda tokens: seen.append(tokens)

    model.prepare_cudagraph_inputs(
        {
            "input_ids": input_ids,
            "rwkv_decode_batch_size": 2,
            "rwkv_decode_rows": [0, 1],
            "rwkv_decode_token_positions": [0, 1],
        }
    )

    assert len(seen) == 1
    assert seen[0].shape == (2, 1)
    assert seen[0].tolist() == [[10], [20]]
    _assert_same_storage_view(seen[0].view(-1), input_ids[:2])


def test_rwkv7_prepare_cudagraph_inputs_rejects_non_dense_decode_positions():
    model = object.__new__(RWKV7ForCausalLM)
    seen = []
    model.prepare_cudagraph_embedding = lambda tokens: seen.append(tokens.clone())

    with pytest.raises(RuntimeError, match="dense contiguous"):
        model.prepare_cudagraph_inputs(
            {
                "input_ids": torch.tensor([10, 20, 30], dtype=torch.int64),
                "rwkv_decode_batch_size": 2,
                "rwkv_decode_rows": [0, 1],
                "rwkv_decode_token_positions": [0, 2],
            }
        )

    assert seen == []


def test_rwkv7_load_weights_preprocesses_full_raw_weights(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    raw_weights = {
        "emb.weight": torch.tensor([1.0]),
        "head.weight": torch.tensor([2.0]),
    }
    calls: list[set[str]] = []

    def fake_preprocess(self, z):
        calls.append(set(z))
        z["processed"] = torch.tensor([3.0])
        self.z = z
        self.emb_cache = {}

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    loaded = model.load_weights(raw_weights.items())

    assert loaded == {"emb.weight", "head.weight"}
    assert model.raw_weight_names == {"emb.weight", "head.weight"}
    assert calls == [{"emb.weight", "head.weight"}]
    assert "processed" in model.z
    assert model.emb_cache == {}


def test_rwkv7_default_loader_validation_uses_raw_weight_names():
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight", "head.weight"}

    DefaultModelLoader.track_weights_loading(
        object.__new__(DefaultModelLoader),
        model,
        {"emb.weight", "head.weight"},
    )


def test_rwkv7_online_weight_update_preprocesses_only_on_finish(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight", "head.weight"}
    old_z = model.z
    calls: list[Any] = []

    def fake_preprocess(self, z):
        calls.append(dict(z))
        z["processed"] = torch.tensor([5.0])
        self.z = z

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    assert model.start_weight_update() is True
    model.load_weights([("emb.weight", torch.tensor([10.0]))])
    model.load_weights([("head.weight", torch.tensor([20.0]))])

    assert model.z is old_z
    assert calls == []

    model.finish_weight_update()

    assert calls[0]["emb.weight"].item() == 10.0
    assert calls[0]["head.weight"].item() == 20.0
    assert model.z["processed"].item() == 5.0
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_reuses_existing_tensor_storage(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.z = {
        "emb.weight": torch.tensor([1.0]),
        "head.weight": torch.tensor([2.0]),
        "stale.weight": torch.tensor([99.0]),
    }
    model.raw_weight_names = {"emb.weight", "head.weight"}
    old_z = model.z
    old_emb = model.z["emb.weight"]
    old_head = model.z["head.weight"]

    def fake_preprocess(self, z):
        z["emb.weight"] = torch.tensor([10.0])
        z["head.weight"] = torch.tensor([20.0])
        z["derived.weight"] = torch.tensor([30.0])

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([10.0]))])
    model.load_weights([("head.weight", torch.tensor([20.0]))])
    model.finish_weight_update()

    assert model.z is old_z
    assert model.z["emb.weight"] is old_emb
    assert model.z["head.weight"] is old_head
    assert model.z["emb.weight"].item() == 10.0
    assert model.z["head.weight"].item() == 20.0
    assert model.z["derived.weight"].item() == 30.0
    assert "stale.weight" not in model.z


def test_rwkv7_online_weight_update_rejects_missing_and_unexpected_keys(
    monkeypatch,
):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight", "head.weight"}
    old_z = model.z
    monkeypatch.setattr(
        RWKV7ForCausalLM,
        "_preprocess_weights",
        lambda self, z: pytest.fail("preprocess should not run"),
    )

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])

    with pytest.raises(ValueError, match="missing.*head.weight"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")

    model.start_weight_update()
    model.load_weights(
        [
            ("emb.weight", torch.tensor([1.0])),
            ("head.weight", torch.tensor([2.0])),
            ("extra.weight", torch.tensor([3.0])),
        ]
    )

    with pytest.raises(ValueError, match="unexpected.*extra.weight"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_keeps_old_weights_on_preprocess_error(
    monkeypatch,
):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    old_z = model.z

    def fail_preprocess(self, z):
        raise RuntimeError("bad preprocess")

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fail_preprocess)

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])

    with pytest.raises(RuntimeError, match="bad preprocess"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_restores_old_weights_after_partial_preprocess(
    monkeypatch,
):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    old_z = model.z

    def fail_after_partial_preprocess(self, z):
        self.z = {"partial": torch.tensor([9.0])}
        self.emb_cache = {}
        raise RuntimeError("post preprocess failure")

    monkeypatch.setattr(
        RWKV7ForCausalLM, "_preprocess_weights", fail_after_partial_preprocess
    )

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])

    with pytest.raises(RuntimeError, match="post preprocess failure"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_cpu_copies_pending_tensors(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    source = torch.tensor([1.0])
    captured = {}

    def fake_preprocess(self, z):
        captured.update(z)
        self.z = z

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    model.start_weight_update()
    model.load_weights([("emb.weight", source)])
    pending = model._pending_weight_update["emb.weight"]
    source.fill_(99.0)

    assert pending.device.type == "cpu"
    assert pending.data_ptr() != source.data_ptr()

    model.finish_weight_update()

    assert captured["emb.weight"].item() == 1.0


def test_rwkv7_abort_weight_update_clears_pending_buffer():
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])
    model.abort_weight_update()

    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_direct_parameter_update_is_not_supported():
    model = _new_rwkv7_for_weight_tests()

    with pytest.raises(NotImplementedError, match="checkpoint-format"):
        model.get_parameter("emb.weight")


def test_rwkv7_model_state_reset_after_weight_update_preserves_mappings():
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )
    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.add_request(1, _new_request("req-1"))
    state.shift_state.fill_(1)
    state.wkv_state.fill_(2)
    state.elapsed.fill_(3)

    state.reset_after_weight_update()

    assert torch.count_nonzero(state.shift_state) == 0
    assert torch.count_nonzero(state.wkv_state) == 0
    assert torch.count_nonzero(state.elapsed) == 0
    assert state.req_id_to_index == {"req-1": 1}
    assert state.req_slot_to_row[1] != -1


def test_rwkv7_model_state_allocates_only_local_pp_layers():
    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )
    model = _new_rwkv7_forward_test_model(
        wkv_mode="fp16",
        start_layer=1,
        end_layer=3,
    )

    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=model,
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    assert state.layer_offset == 1
    assert state.num_layers == 2
    assert state.shift_state.shape == (2, 2, 3, 64)
    assert state.wkv_state.shape == (2, 3, 1, 64, 64)


def test_rwkv7_model_state_allocates_only_local_tp_heads():
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=256,
        head_size=64,
        num_attention_heads=4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )
    model = _new_rwkv7_forward_test_model(wkv_mode="fp16", tp_num_heads=2)

    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=model,
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    assert state.num_heads == 2
    assert state.shift_state.shape == (2, 2, 3, 256)
    assert state.wkv_state.shape == (2, 3, 2, 64, 64)


def test_rwkv7_pipeline_rank_keeps_only_stage_weights():
    middle = object.__new__(RWKV7ForCausalLM)
    middle.start_layer = 1
    middle.end_layer = 3
    middle.total_num_layers = 4

    assert not middle._is_weight_needed_on_rank("emb.weight")
    assert not middle._is_weight_needed_on_rank("blocks.0.att.r_k")
    assert middle._is_weight_needed_on_rank("blocks.1.att.r_k")
    assert middle._is_weight_needed_on_rank("blocks.2.ffn.key.weight")
    assert not middle._is_weight_needed_on_rank("blocks.3.att.r_k")
    assert not middle._is_weight_needed_on_rank("ln_out.weight")
    assert not middle._is_weight_needed_on_rank("head.weight")

    last = object.__new__(RWKV7ForCausalLM)
    last.start_layer = 3
    last.end_layer = 4
    last.total_num_layers = 4

    assert last._is_weight_needed_on_rank("blocks.3.att.r_k")
    assert last._is_weight_needed_on_rank("ln_out.weight")
    assert last._is_weight_needed_on_rank("head.weight")


def test_rwkv7_tensor_parallel_shards_weights_for_rank():
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 2
    model.tp_rank = 1
    model.tp_num_heads = 2
    model.tp_hidden_size = 128

    def values(*shape: int) -> torch.Tensor:
        return torch.arange(int(np.prod(shape)), dtype=torch.float32).view(*shape)

    for key in ("emb.weight", "head.weight"):
        weight = values(10, 256)
        torch.testing.assert_close(
            model._shard_weight_for_tp(key, weight),
            weight[5:10],
            rtol=0,
            atol=0,
        )

    weight = values(4, 64)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.r_k", weight),
        weight[2:4],
        rtol=0,
        atol=0,
    )

    weight = values(256, 256)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.receptance.weight", weight),
        weight[128:256],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.output.weight", weight),
        weight[:, 128:256],
        rtol=0,
        atol=0,
    )

    weight = values(512, 256)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.ffn.key.weight", weight),
        weight[256:512],
        rtol=0,
        atol=0,
    )

    weight = values(256, 512)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.ffn.value.weight", weight),
        weight[:, 256:512],
        rtol=0,
        atol=0,
    )

    weight = values(32, 256)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.w2", weight),
        weight[:, 128:256],
        rtol=0,
        atol=0,
    )

    weight = values(256)
    assert model._shard_weight_for_tp("blocks.1.att.x_r", weight) is weight
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.ln_x.weight", weight),
        weight[128:256],
        rtol=0,
        atol=0,
    )


def test_rwkv7_compute_logits_all_gathers_tensor_parallel_vocab(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 2
    model.vocab_size = 5
    model.linear_head = lambda hidden_states: hidden_states.new_ones((2, 3))
    model.logits_processor = lambda lm_head, logits: logits

    def fake_all_gather(logits):
        assert logits.shape == (2, 3)
        return torch.cat([logits, logits + 10], dim=-1)

    monkeypatch.setattr(rwkv7, "tensor_model_parallel_all_gather", fake_all_gather)

    logits = model.compute_logits(torch.empty(2, 4))

    assert logits.shape == (2, 5)
    assert logits.tolist() == [
        [1.0, 1.0, 1.0, 11.0, 11.0],
        [1.0, 1.0, 1.0, 11.0, 11.0],
    ]


def test_rwkv7_compute_sampling_logits_uses_packed_decode_view():
    model = _new_rwkv7_forward_test_model()
    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    logits_indices = torch.arange(3, dtype=torch.int64)
    expected_view = hidden_states[:3]
    expected_logits = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    seen_sample_hidden_states = []

    def compute_logits(sample_hidden_states):
        seen_sample_hidden_states.append(sample_hidden_states)
        _assert_same_storage_view(sample_hidden_states, expected_view)
        return expected_logits

    model.compute_logits = compute_logits
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_draft_tokens=0,
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
        is_prefilling_np=np.zeros(3, dtype=np.bool_),
        logits_indices=logits_indices,
    )

    logits = model.compute_sampling_logits(hidden_states, logits_indices, input_batch)

    assert logits is expected_logits
    assert len(seen_sample_hidden_states) == 1


@pytest.mark.parametrize(
    ("case_name", "logits_indices", "input_batch"),
    [
        (
            "prefill",
            torch.tensor([0, 2], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.array([1, 2], dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 3], dtype=np.int32),
                is_prefilling_np=np.array([False, True], dtype=np.bool_),
            ),
        ),
        (
            "spec_decode",
            torch.tensor([0, 1], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=1,
                num_draft_tokens=1,
                num_draft_tokens_per_req=np.array([1], dtype=np.int32),
                num_scheduled_tokens=np.array([1], dtype=np.int32),
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1], dtype=np.int32),
                is_prefilling_np=np.array([False], dtype=np.bool_),
            ),
        ),
        (
            "non_contiguous_logits",
            torch.tensor([0, 2], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
                is_prefilling_np=np.zeros(2, dtype=np.bool_),
            ),
        ),
        (
            "missing_prefill_metadata",
            torch.tensor([0, 1], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
            ),
        ),
        (
            "floating_logits_indices",
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
                is_prefilling_np=np.zeros(2, dtype=np.bool_),
            ),
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_rwkv7_compute_sampling_logits_declines_non_decode_or_spec_batch(
    case_name,
    logits_indices,
    input_batch,
):
    del case_name
    model = _new_rwkv7_forward_test_model()
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    def compute_logits(_sample_hidden_states):
        pytest.fail("declined sampling logits hook must not compute logits")

    model.compute_logits = compute_logits

    logits = model.compute_sampling_logits(hidden_states, logits_indices, input_batch)

    assert logits is None


def test_rwkv7_tensor_parallel_embedding_masks_remote_tokens(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 2
    model.tp_rank = 0
    model.vocab_size = 5
    model.emb_cpu = False
    model.z = {
        "emb.weight": torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
    }
    monkeypatch.setattr(rwkv7, "tensor_model_parallel_all_reduce", lambda x: x)

    out = model.embed(torch.tensor([[0, 4]], dtype=torch.int64))

    assert out.tolist() == [[[1.0, 1.0], [0.0, 0.0]]]


def test_rwkv7_model_state_lifecycle_resets_reused_rows():
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        hidden_size=128,
        head_size=64,
        num_attention_heads=2,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    model = _new_rwkv7_forward_test_model(wkv_mode="fp16")

    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=model,
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    state.shift_state[:, :, 0].fill_(1)
    state.wkv_state[:, 0].fill_(1)
    state.elapsed[0] = 7
    state.add_request(2, _new_request("req-2"))
    row = state.req_slot_to_row[2]
    assert row == 0
    assert torch.count_nonzero(state.shift_state[:, :, row]) == 0
    assert torch.count_nonzero(state.wkv_state[:, row]) == 0
    assert state.elapsed[row].item() == 0
    assert state.req_id_to_index["req-2"] == 2

    state.elapsed[row] = 11
    state.remove_request("req-2")
    assert "req-2" not in state.req_id_to_index
    assert state.elapsed[row].item() == 0


def test_rwkv7_model_state_rejects_permuted_decode_schedule():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.elapsed[:2] = torch.tensor([10, 20], dtype=torch.int32)

    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([1, 0], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )

    with pytest.raises(RuntimeError, match="resident-row order"):
        state.prepare_inputs(input_batch, req_states=None)

    assert state.req_slot_to_row[:2] == [0, 1]
    assert state.row_to_req_slot[:2] == [0, 1]
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 0] == 10)
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert state.elapsed.tolist()[:2] == [10, 20]


def test_rwkv7_model_state_sorts_decode_wave_by_resident_row():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))

    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1, 2, 3], dtype=np.int32),
        num_reqs=4,
        query_start_loc=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    req_states = SimpleNamespace(
        req_id_to_index={f"req-{req_slot}": req_slot for req_slot in range(4)},
        num_computed_prefill_tokens=np.array([3, 3, 3, 3], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([3, 3, 3, 3], dtype=np.int32)),
    )

    req_ids = ["req-1", "req-2", "req-3", "req-0"]
    sorted_req_ids = state.sort_scheduled_req_ids(
        req_ids,
        {req_id: 1 for req_id in req_ids},
        req_states,
    )

    assert sorted_req_ids == ["req-0", "req-1", "req-2", "req-3"]


def test_rwkv7_model_state_sort_keeps_one_token_prefill_after_decode():
    state = _new_rwkv7_model_state(max_num_reqs=3)
    for req_slot in range(3):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    req_states = SimpleNamespace(
        req_id_to_index={f"req-{req_slot}": req_slot for req_slot in range(3)},
        num_computed_prefill_tokens=np.array([3, 3, 0], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([3, 3, 1], dtype=np.int32)),
    )

    req_ids = ["req-2", "req-1", "req-0"]
    sorted_req_ids = state.sort_scheduled_req_ids(
        req_ids,
        {req_id: 1 for req_id in req_ids},
        req_states,
    )

    assert sorted_req_ids == ["req-0", "req-1", "req-2"]


def test_rwkv7_model_state_reports_current_decode_context_size_not_capacity():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["rwkv_decode_batch_size"] == 1
    assert inputs["rwkv_decode_rows"] == [0]
    assert inputs["rwkv_decode_token_positions"] == [0]


def test_rwkv7_model_state_compacts_steady_decode_after_row_removal():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 1].fill_(30)
    state.elapsed[1] = 40
    state.remove_request("req-0")

    assert state.req_slot_to_row[:2] == [-1, 0]
    assert state.row_to_req_slot[:2] == [1, -1]
    assert torch.all(state.shift_state[:, :, 0] == 20)
    assert torch.all(state.wkv_state[:, 0] == 30)
    assert state.elapsed.tolist()[:2] == [40, 0]

    state.reset_state_movement_stats()
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([1], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )

    inputs = state.prepare_inputs(decode_batch, req_states=None)

    assert inputs["rwkv_decode_batch_size"] == 1
    assert inputs["idx_mapping"].tolist() == [0]
    assert inputs["rwkv_decode_rows"] == [0]
    assert inputs["rwkv_decode_token_positions"] == [0]
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert state.get_state_movement_stats() == {
        "resident_to_decode_copies": 0,
        "decode_compactions": 0,
        "decode_compaction_rows": 0,
    }


def test_rwkv7_model_state_keeps_decode_prefix_when_prefill_reuses_free_row():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("decode-low"))
    state.add_request(1, _new_request("decode-high"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 1].fill_(20)
    state.elapsed[1] = 20

    state.remove_request("decode-low")
    state.add_request(2, _new_request("prefill-low"))
    decode_row = state.req_slot_to_row[1]
    prefill_row = state.req_slot_to_row[2]
    assert decode_row == 0
    assert prefill_row == 1

    mixed_batch = SimpleNamespace(
        idx_mapping_np=np.array([1, 2], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        is_prefilling_np=np.array([False, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(mixed_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [0, 1]
    assert inputs["rwkv_decode_batch_size"] == 1
    assert inputs["rwkv_decode_rows"] == [0]
    assert inputs["rwkv_decode_token_positions"] == [0]
    assert inputs["rwkv_prefill_rows"] == [prefill_row]
    assert inputs["rwkv_prefill_groups"] == [(1, 2, 3, 1, 4, prefill_row)]
    assert inputs["prefill_idx_mapping"].tolist() == [-1, -1]
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert inputs["prefill_shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["prefill_wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["prefill_elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert state.num_decode_rows == 1
    assert state.decode_req_slots == {1}
    assert state._prefill_decode_rows == [prefill_row]
    assert state.req_slot_to_row[:3] == [-1, decode_row, prefill_row]
    assert state.row_to_req_slot[:3] == [1, 2, -1]
    assert 2 in state.free_rows
    assert torch.count_nonzero(state.shift_state[:, :, prefill_row]) == 0
    assert torch.all(state.shift_state[:, :, decode_row] == 20)
    assert torch.count_nonzero(state.shift_state[:, :, 2]) == 0
    assert torch.count_nonzero(state.wkv_state[:, prefill_row]) == 0
    assert torch.all(state.wkv_state[:, decode_row] == 20)
    assert torch.count_nonzero(state.wkv_state[:, 2]) == 0
    assert state.elapsed.tolist()[:3] == [20, 0, 0]


def test_rwkv7_prepare_permuted_decode_rejects_before_forward():
    state = _new_rwkv7_model_state(
        max_num_reqs=2,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 0].fill_(30)
    state.wkv_state[:, 1].fill_(40)
    state.elapsed[:2] = torch.tensor([50, 60], dtype=torch.int32)
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([1, 0], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )

    with pytest.raises(RuntimeError, match="resident-row order"):
        state.prepare_inputs(input_batch, req_states=None)

    assert state.req_slot_to_row[:2] == [0, 1]
    assert state.row_to_req_slot[:2] == [0, 1]
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 0] == 30)
    assert torch.all(state.wkv_state[:, 1] == 40)
    assert state.elapsed.tolist() == [50, 60]


def test_rwkv7_model_state_rejects_decode_blocked_by_resident_prefill():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.elapsed[:2] = torch.tensor([10, 20], dtype=torch.int32)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
        is_prefilling_np=np.array([True, False], dtype=np.bool_),
        num_scheduled_tokens=np.array([3, 1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([3, 1], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="cannot compact decode rows"):
        state.prepare_inputs(input_batch, req_states=None)

    assert state.req_slot_to_row[:2] == [0, 1]
    assert state.row_to_req_slot[:2] == [0, 1]
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 0] == 10)
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert state.elapsed.tolist()[:2] == [10, 20]


def test_rwkv7_model_state_rejects_partial_live_decode_wave():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)

    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )

    with pytest.raises(RuntimeError, match="all live decode rows"):
        state.prepare_inputs(input_batch, req_states=None)
    assert state.num_decode_rows == 2
    assert state.req_slot_to_row[:2] == [0, 1]


def test_rwkv7_model_state_does_not_park_unscheduled_decode_rows():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([1], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )

    with pytest.raises(RuntimeError, match="all live decode rows"):
        state.prepare_inputs(input_batch, req_states=None)
    assert state.num_decode_rows == 2
    assert state.req_slot_to_row[:2] == [0, 1]


def test_rwkv7_model_state_rejects_permuted_decode_with_resident_prefill():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.elapsed[:3] = torch.tensor([10, 20, 30], dtype=torch.int32)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([1, 0, 2], dtype=np.int32),
        num_reqs=3,
        query_start_loc=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 1, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 1, 3], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="resident-row order"):
        state.prepare_inputs(input_batch, req_states=None)

    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.shift_state[:, :, 2] == 30)
    assert state.elapsed.tolist()[:3] == [10, 20, 30]


def test_rwkv7_model_state_compacts_prefill_to_decode_transition():
    state = _new_rwkv7_model_state(
        max_num_reqs=4,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    for req_slot in range(3):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    state.remove_request("req-1")
    assert state.req_slot_to_row[:3] == [0, -1, 2]

    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 2].fill_(20)
    state.wkv_state[:, 2].fill_(21)
    state.elapsed[2] = 22

    prefill_to_decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([2], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([2], dtype=np.int32),
    )
    prefill_inputs = state.prepare_inputs(prefill_to_decode_batch, req_states=None)

    state.postprocess_state(
        prefill_inputs["idx_mapping"],
        torch.tensor([1], dtype=torch.int32),
    )

    assert state.decode_req_slots == {0, 2}
    assert state.req_slot_to_row[:3] == [0, -1, 1]
    assert state.row_to_req_slot[:3] == [0, 2, -1]
    assert 2 in state.free_rows
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 1] == 21)
    assert state.elapsed.tolist()[:3] == [0, 22, 0]


def _fragmented_mixed_rwkv7_inputs() -> tuple[RWKV7ModelState, dict[str, Any]]:
    state = _new_rwkv7_model_state(
        max_num_reqs=4,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    state.remove_request("req-1")

    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)

    prefill_to_decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([2], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([2], dtype=np.int32),
    )
    prefill_inputs = state.prepare_inputs(prefill_to_decode_batch, req_states=None)
    state.postprocess_state(
        prefill_inputs["idx_mapping"],
        torch.tensor([1], dtype=torch.int32),
    )

    assert state.req_slot_to_row[:4] == [0, -1, 1, 3]
    assert state.row_to_req_slot[:4] == [0, 2, -1, 3]
    assert state.decode_req_slots == {0, 2}
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 3].fill_(30)
    state.wkv_state[:, 0].fill_(11)
    state.wkv_state[:, 1].fill_(21)
    state.wkv_state[:, 3].fill_(31)
    state.elapsed[:4] = torch.tensor([12, 22, 0, 32], dtype=torch.int32)
    state.reset_state_movement_stats()

    mixed_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 2, 3], dtype=np.int32),
        num_reqs=3,
        query_start_loc=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 1, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 1, 3], dtype=np.int32),
    )

    return state, state.prepare_inputs(mixed_batch, req_states=None)


def test_rwkv7_model_state_keeps_prefill_transition_compaction_before_forward():
    state, inputs = _fragmented_mixed_rwkv7_inputs()

    assert inputs["rwkv_decode_batch_size"] == 2
    assert inputs["rwkv_decode_rows"] == [0, 1]
    assert inputs["rwkv_decode_token_positions"] == [0, 1]
    assert inputs["idx_mapping"].tolist() == [0, 1, 3]
    assert inputs["rwkv_prefill_rows"] == [3]
    assert inputs["rwkv_prefill_groups"] == [(2, 3, 3, 2, 5, 3)]
    assert state.req_slot_to_row[:4] == [0, -1, 1, 3]
    assert state.row_to_req_slot[:4] == [0, 2, -1, 3]
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.shift_state[:, :, 3] == 30)
    assert state.elapsed.tolist()[:4] == [12, 22, 0, 32]


def test_rwkv7_model_state_reports_zero_prepare_compaction_after_transition():
    state, _inputs = _fragmented_mixed_rwkv7_inputs()

    assert state.get_state_movement_stats() == {
        "resident_to_decode_copies": 0,
        "decode_compactions": 0,
        "decode_compaction_rows": 0,
    }


def test_rwkv7_model_state_reports_zero_steady_decode_copies():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )

    state.prepare_inputs(decode_batch, req_states=None)
    state.reset_state_movement_stats()
    state.prepare_inputs(decode_batch, req_states=None)

    assert state.get_state_movement_stats() == {
        "resident_to_decode_copies": 0,
        "decode_compactions": 0,
        "decode_compaction_rows": 0,
    }


def test_rwkv7_model_state_prefill_uses_resident_state():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(2, _new_request("req-2"))
    state.shift_state.fill_(7)
    state.wkv_state.fill_(8)
    state.elapsed.fill_(9)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([2], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [0]
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert inputs["rwkv_prefill_rows"] == [0]
    assert inputs["rwkv_prefill_groups"] == [(0, 1, 3, 0, 3, 0)]
    assert torch.all(state.shift_state == 7)
    assert torch.all(state.wkv_state == 8)
    assert torch.all(state.elapsed == 9)


def test_rwkv7_model_state_keeps_resident_prefill_row_when_decode_row_starts():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    row = state.req_slot_to_row[2]
    assert row == 2
    state.shift_state[:, :, 0].fill_(3)
    state.shift_state[:, :, 1].fill_(5)
    state.wkv_state[:, 0].fill_(7)
    state.wkv_state[:, 1].fill_(9)
    state.elapsed[:2] = torch.tensor([11, 13], dtype=torch.int32)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([2], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)
    inputs["shift_state"][:, :, row].fill_(11)
    inputs["wkv_state"][:, row].fill_(13)
    inputs["elapsed"][row].fill_(17)

    state.postprocess_state(inputs["idx_mapping"], torch.tensor([1], dtype=torch.int32))

    decode_row = state.req_slot_to_row[2]
    assert 2 in state.decode_req_slots
    assert decode_row == row
    assert torch.all(state.shift_state[:, :, decode_row] == 11)
    assert torch.all(state.wkv_state[:, decode_row] == 13)
    assert state.elapsed[decode_row].item() == 17
    assert state.req_slot_to_row[:3] == [0, 1, 2]
    assert state.row_to_req_slot[:3] == [0, 1, 2]
    assert torch.all(state.shift_state[:, :, 0] == 3)
    assert torch.all(state.shift_state[:, :, 1] == 5)
    assert torch.all(state.wkv_state[:, 0] == 7)
    assert torch.all(state.wkv_state[:, 1] == 9)
    assert state.elapsed.tolist()[:3] == [11, 13, 17]


def test_rwkv7_model_state_prefill_becomes_decode_without_resident_copy():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    row = state.req_slot_to_row[2]
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([2], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)
    inputs["shift_state"][:, :, row].fill_(11)
    inputs["wkv_state"][:, row].fill_(13)
    inputs["elapsed"][row].fill_(17)

    state.postprocess_state(inputs["idx_mapping"], torch.tensor([1], dtype=torch.int32))

    assert 2 in state.decode_req_slots
    assert state.num_decode_rows == 1
    assert state.req_slot_to_row[2] == row
    assert state.row_to_req_slot[row] == 2
    assert torch.all(state.shift_state[:, :, row] == 11)
    assert torch.all(state.wkv_state[:, row] == 13)
    assert state.elapsed[row].item() == 17
    assert not state.has_pending_postprocess_state()


def test_rwkv7_model_state_reports_pending_prefill_state_postprocess():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    assert not state.has_pending_postprocess_state()
    inputs = state.prepare_inputs(input_batch, req_states=None)
    assert state.has_pending_postprocess_state()

    state.postprocess_state(inputs["idx_mapping"], torch.tensor([1], dtype=torch.int32))

    assert not state.has_pending_postprocess_state()


def test_rwkv7_non_last_pp_postprocesses_pending_prefill_when_all_decode_next():
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([0, 1], dtype=torch.int32))
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.is_last_pp_rank = False
    runner.pp_handler = SimpleNamespace(receive=lambda _input_batch: True)
    runner.kv_connector = SimpleNamespace(
        post_forward=lambda _finished_req_ids: SimpleNamespace(is_empty=lambda: True)
    )
    runner.eplb = SimpleNamespace(step=lambda **_kwargs: None)

    calls: list[Any] = []
    runner.postprocess_num_computed_tokens = lambda _input_batch: calls.append(
        "num_computed"
    )
    runner.model_state = SimpleNamespace(
        has_pending_postprocess_state=lambda: True,
        postprocess_state=lambda idx_mapping, num_sampled: calls.append(
            ("state", idx_mapping.tolist(), num_sampled)
        ),
    )

    GPUModelRunner.sample_tokens(runner, grammar_output=None)

    assert calls == ["num_computed", ("state", [0, 1], 0)]


def test_non_rwkv_non_last_pp_rank_does_not_require_pending_state_hook():
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([0, 1], dtype=torch.int32))
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.is_last_pp_rank = False
    runner.pp_handler = SimpleNamespace(receive=lambda _input_batch: True)
    runner.kv_connector = SimpleNamespace(
        post_forward=lambda _finished_req_ids: SimpleNamespace(is_empty=lambda: True)
    )
    runner.eplb = SimpleNamespace(step=lambda **_kwargs: None)

    calls: list[Any] = []
    runner.postprocess_num_computed_tokens = lambda _input_batch: calls.append(
        "num_computed"
    )
    runner.model_state = SimpleNamespace(
        postprocess_state=lambda idx_mapping, num_sampled: calls.append(
            ("state", idx_mapping.tolist(), num_sampled)
        ),
    )

    GPUModelRunner.sample_tokens(runner, grammar_output=None)

    assert calls == ["num_computed"]


def test_rwkv7_model_state_remove_request_recycles_stable_row():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.elapsed[:3] = torch.tensor([10, 20, 30], dtype=torch.int32)

    state.remove_request("req-1")

    assert state.req_slot_to_row[0] == 0
    assert state.req_slot_to_row[1] == -1
    assert state.req_slot_to_row[2] == 2
    assert state.row_to_req_slot[:3] == [0, -1, 2]
    assert 1 in state.free_rows
    assert torch.count_nonzero(state.shift_state[:, :, 1]) == 0
    assert torch.count_nonzero(state.wkv_state[:, 1]) == 0
    assert torch.all(state.shift_state[:, :, 2] == 30)
    assert torch.all(state.wkv_state[:, 2] == 30)
    assert state.elapsed.tolist()[:3] == [10, 0, 30]

    state.add_request(3, _new_request("req-3"))

    assert state.req_slot_to_row[3] == 1


def test_rwkv7_model_state_remove_decode_row_keeps_other_resident_rows_stable():
    state = _new_rwkv7_model_state(max_num_reqs=5)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(input_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.shift_state[:, :, 3].fill_(40)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.wkv_state[:, 3].fill_(40)
    state.elapsed[:4] = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    state.remove_request("req-0")

    assert state.num_decode_rows == 1
    assert state.req_slot_to_row[:4] == [-1, 0, 2, 3]
    assert state.row_to_req_slot[:4] == [1, -1, 2, 3]
    assert 1 in state.free_rows
    assert torch.all(state.shift_state[:, :, 0] == 20)
    assert torch.count_nonzero(state.shift_state[:, :, 1]) == 0
    assert torch.all(state.shift_state[:, :, 2] == 30)
    assert torch.all(state.shift_state[:, :, 3] == 40)
    assert torch.all(state.wkv_state[:, 0] == 20)
    assert torch.count_nonzero(state.wkv_state[:, 1]) == 0
    assert torch.all(state.wkv_state[:, 2] == 30)
    assert torch.all(state.wkv_state[:, 3] == 40)
    assert state.elapsed.tolist()[:4] == [20, 0, 30, 40]


def test_rwkv7_model_state_remove_prefill_row_preserves_decode_prefix():
    state = _new_rwkv7_model_state(max_num_reqs=5)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(input_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.shift_state[:, :, 3].fill_(40)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.wkv_state[:, 3].fill_(40)
    state.elapsed[:4] = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    state.remove_request("req-2")

    assert state.num_decode_rows == 2
    assert state.req_slot_to_row[:4] == [0, 1, -1, 3]
    assert state.row_to_req_slot[:4] == [0, 1, -1, 3]
    assert 2 in state.free_rows
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.count_nonzero(state.shift_state[:, :, 2]) == 0
    assert torch.all(state.shift_state[:, :, 3] == 40)
    assert torch.all(state.wkv_state[:, 0] == 10)
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert torch.count_nonzero(state.wkv_state[:, 2]) == 0
    assert torch.all(state.wkv_state[:, 3] == 40)
    assert state.elapsed.tolist()[:4] == [10, 20, 0, 40]


def test_rwkv7_model_state_dummy_batch_uses_scratch_state():
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.req_slot_to_row = [3, 2, 1, 0]
    state.row_to_req_slot = [3, 2, 1, 0]
    state.free_rows.clear()
    state.shift_state.fill_(1)
    state.wkv_state.fill_(2)
    state.elapsed.fill_(3)
    input_batch = SimpleNamespace(
        req_ids=["_warmup_0_", "_warmup_1_"],
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [0, 1]
    assert inputs["shift_state"].data_ptr() != state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() != state.wkv_state.data_ptr()
    assert torch.count_nonzero(inputs["shift_state"]) == 0
    assert torch.count_nonzero(inputs["wkv_state"]) == 0
    assert torch.count_nonzero(inputs["elapsed"]) == 0
    assert state.req_slot_to_row == [-1, -1, -1, -1]
    assert state.row_to_req_slot == [-1, -1, -1, -1]
    assert state.free_rows == {0, 1, 2, 3}
    assert torch.all(state.shift_state == 1)
    assert torch.all(state.wkv_state == 2)
    assert torch.all(state.elapsed == 3)


def test_rwkv7_uses_default_vllm_sampler():
    assert not hasattr(rwkv_state, "_sample_albatross_logits")
    assert RWKV7ModelState.custom_sampler(SimpleNamespace(), SimpleNamespace()) is None


def test_rwkv7_dummy_inputs_cover_all_tokens():
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp32io16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.shift_state.fill_(1)
    state.wkv_state.fill_(2)
    state.elapsed.fill_(3)

    inputs = state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)

    assert inputs["query_start_loc"].tolist() == [0, 3, 6, 8]
    assert inputs["idx_mapping"].tolist() == [0, 1, 2]
    assert inputs["wkv_state"].dtype == torch.float32
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert torch.all(state.shift_state == 1)
    assert torch.all(state.wkv_state == 2)
    assert torch.all(state.elapsed == 3)


def test_rwkv7_vllm_forward_groups_equal_length_prefill_requests(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    calls = []
    decode_state_ptrs = []

    def forward_tokens(tokens, state):
        decode_state_ptrs.append(tuple(tensor.data_ptr() for tensor in state))
        calls.append(("tokens", tuple(tokens.shape), tuple(state[0].shape)))
        state[0].fill_(10)
        state[1].fill_(20)
        state[2].fill_(1)
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    def forward_all_hidden(tokens, state):
        calls.append(("all_hidden", tuple(tokens.shape), tuple(state[0].shape)))
        _assert_same_storage_view(state[0], shift_state[:, :, 2:4, :])
        _assert_same_storage_view(state[1], wkv_state[:, 2:4, :, :, :])
        _assert_same_storage_view(state[2], elapsed[2:4])
        state[0].fill_(30)
        state[1].fill_(40)
        state[2].fill_(2)
        assert torch.all(shift_state[:, :, 2:4] == 30)
        assert torch.all(wkv_state[:, 2:4] == 40)
        assert elapsed.tolist() == [1, 1, 2, 2]
        return tokens.to(torch.float32).unsqueeze(-1).expand(*tokens.shape, 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=forward_all_hidden,
    )
    input_ids = torch.tensor([10, 20, 30, 31, 40, 41], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 4, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((4,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_prefill_groups=[(2, 4, 2, 2, 6, 2)],
    )

    assert calls == [
        ("tokens", (2, 1), (1, 2, 2, 3)),
        ("all_hidden", (2, 2), (1, 2, 2, 3)),
    ]
    assert decode_state_ptrs == [
        (shift_state.data_ptr(), wkv_state.data_ptr(), elapsed.data_ptr())
    ]
    assert out.tolist() == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [31.0, 31.0, 31.0],
        [40.0, 40.0, 40.0],
        [41.0, 41.0, 41.0],
    ]
    assert torch.all(shift_state[:, :, :2] == 10)
    assert torch.all(shift_state[:, :, 2:] == 30)
    assert torch.all(wkv_state[:, :2] == 20)
    assert torch.all(wkv_state[:, 2:] == 40)
    assert elapsed.tolist() == [1, 1, 2, 2]


def test_rwkv7_model_state_reports_equal_length_prefill_groups():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0, 1, 2, 3], dtype=np.int32),
        num_reqs=4,
        query_start_loc=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 1, 2, 2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0, 0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["rwkv_prefill_groups"] == [(2, 4, 2, 2, 6, 2)]


def test_rwkv7_vllm_forward_uses_dense_decode_input_view_for_compact_rows(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    seen_tokens = []
    input_ids = torch.tensor([10, 20], dtype=torch.int64)
    shift_state = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((2,), dtype=torch.int32)

    def forward_tokens(tokens, state):
        seen_tokens.append(tokens.tolist())
        _assert_same_storage_view(tokens.view(-1), input_ids[:2])
        _assert_same_storage_view(state[0], shift_state[:, :, :2, :])
        _assert_same_storage_view(state[1], wkv_state[:, :2, :, :, :])
        _assert_same_storage_view(state[2], elapsed[:2])
        state[0].add_(1)
        state[1].add_(2)
        state[2].add_(3)
        return torch.tensor(
            [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]],
            dtype=torch.float32,
        )

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail(
            "pure compact decode must not run prefill path"
        ),
    )

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
    )

    assert seen_tokens == [[[10], [20]]]
    assert out.tolist() == [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]]
    assert torch.all(shift_state == 1)
    assert torch.all(wkv_state == 2)
    assert elapsed.tolist() == [3, 3]


def test_rwkv7_vllm_forward_rejects_permuted_decode_rows(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    seen_tokens = []

    def forward_tokens(tokens, state):
        seen_tokens.append(tokens.tolist())
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail(
            "pure decode must not run prefill path"
        ),
    )
    shift_state = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((2,), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="compact prefix rows"):
        RWKV7ForCausalLM.forward(
            model,
            torch.tensor([10, 20], dtype=torch.int64),
            positions=None,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
            shift_state=shift_state,
            wkv_state=wkv_state,
            elapsed=elapsed,
            rwkv_decode_batch_size=2,
            rwkv_decode_rows=[1, 0],
            rwkv_decode_token_positions=[0, 1],
        )

    assert seen_tokens == []
    assert torch.count_nonzero(shift_state) == 0
    assert torch.count_nonzero(wkv_state) == 0
    assert elapsed.tolist() == [0, 0]


def test_rwkv7_vllm_forward_rejects_non_prefix_decode_rows_before_gather(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    def forward_tokens(tokens, state):
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail("decode must stay T=1"),
    )

    with pytest.raises(RuntimeError, match="compact prefix rows"):
        RWKV7ForCausalLM.forward(
            model,
            torch.tensor([10, 20], dtype=torch.int64),
            positions=None,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
            shift_state=torch.zeros((1, 2, 3, 3), dtype=torch.float32),
            wkv_state=torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32),
            elapsed=torch.zeros((3,), dtype=torch.int32),
            rwkv_decode_batch_size=2,
            rwkv_decode_rows=[2, 0],
            rwkv_decode_token_positions=[0, 1],
        )


def test_rwkv7_compact_decode_token_range_rejects_non_prefix_rows():
    with pytest.raises(RuntimeError, match="compact prefix rows"):
        RWKV7ForCausalLM._compact_decode_token_range(2, [1, 2], [0, 1])


def test_rwkv7_vllm_forward_resident_row_without_decode_metadata_uses_singleton_prefill(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    seen = []

    def forward_tokens(tokens, state):
        seen.append((tokens.tolist(), state[0].storage_offset()))
        _assert_same_storage_view(state[0], shift_state[:, :, 1:2, :])
        _assert_same_storage_view(state[1], wkv_state[:, 1:2, :, :, :])
        _assert_same_storage_view(state[2], elapsed[1:2])
        state[0].fill_(11)
        state[1].fill_(13)
        state[2].fill_(17)
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail(
            "singleton row should stay on the per-row path"
        ),
    )
    shift_state = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((2,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        torch.tensor([20], dtype=torch.int64),
        positions=None,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping=torch.tensor([1], dtype=torch.int32),
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
    )

    assert seen == [([[20]], shift_state[:, :, 1:2, :].storage_offset())]
    assert out.tolist() == [[20.0, 20.0, 20.0]]
    assert torch.count_nonzero(shift_state[:, :, 0]) == 0
    assert torch.all(shift_state[:, :, 1] == 11)
    assert torch.count_nonzero(wkv_state[:, 0]) == 0
    assert torch.all(wkv_state[:, 1] == 13)
    assert elapsed.tolist() == [0, 17]


def test_rwkv7_vllm_forward_passes_resident_state_to_single_prefill(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    calls = []

    def forward_tokens(tokens, state):
        calls.append(("tokens", tokens.tolist()))
        assert tuple(tensor.data_ptr() for tensor in state) == (
            shift_state.data_ptr(),
            wkv_state.data_ptr(),
            elapsed.data_ptr(),
        )
        state[0].fill_(10)
        state[1].fill_(20)
        state[2].fill_(1)
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    def forward_all_hidden(tokens, state):
        calls.append(("all_hidden", tokens.tolist()))
        _assert_same_storage_view(state[0], shift_state[:, :, 2:3, :])
        _assert_same_storage_view(state[1], wkv_state[:, 2:3, :, :, :])
        _assert_same_storage_view(state[2], elapsed[2:3])
        state[0].fill_(30)
        state[1].fill_(40)
        state[2].fill_(2)
        assert torch.all(shift_state[:, :, 2:3] == 30)
        assert torch.all(wkv_state[:, 2:3] == 40)
        assert elapsed.tolist() == [1, 1, 2]
        return tokens.to(torch.float32).unsqueeze(-1).expand(*tokens.shape, 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=forward_all_hidden,
    )
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        torch.tensor([10, 20, 30, 31, 32], dtype=torch.int64),
        positions=None,
        query_start_loc=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
        idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_prefill_token_ranges=[(2, 2, 5)],
        rwkv_prefill_rows=[2],
    )

    assert calls == [
        ("tokens", [[10], [20]]),
        ("all_hidden", [[30, 31, 32]]),
    ]
    assert out.tolist() == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [31.0, 31.0, 31.0],
        [32.0, 32.0, 32.0],
    ]
    assert torch.all(shift_state[:, :, :2] == 10)
    assert torch.all(shift_state[:, :, 2] == 30)
    assert torch.all(wkv_state[:, :2] == 20)
    assert torch.all(wkv_state[:, 2] == 40)
    assert elapsed.tolist() == [1, 1, 2]


def test_rwkv7_vllm_forward_uses_compact_mixed_decode_state_without_gather(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))
    state, model_inputs = _fragmented_mixed_rwkv7_inputs()

    def forbid_index_select(*args, **kwargs):
        pytest.fail("model-side decode state gather must not be used")

    monkeypatch.setattr(torch, "index_select", forbid_index_select)
    calls = []

    def forward_tokens(tokens, model_state):
        calls.append(("tokens", tokens.tolist()))
        _assert_same_storage_view(model_state[0], state.shift_state[:, :, :2, :])
        _assert_same_storage_view(model_state[1], state.wkv_state[:, :2, :, :, :])
        _assert_same_storage_view(model_state[2], state.elapsed[:2])
        assert torch.all(model_state[0][:, :, 0] == 10)
        assert torch.all(model_state[0][:, :, 1] == 20)
        assert model_state[2].tolist() == [12, 22]
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    def forward_all_hidden(tokens, model_state):
        calls.append(("all_hidden", tokens.tolist()))
        _assert_same_storage_view(model_state[0], state.shift_state[:, :, 3:4, :])
        _assert_same_storage_view(model_state[1], state.wkv_state[:, 3:4, :, :, :])
        _assert_same_storage_view(model_state[2], state.elapsed[3:4])
        return tokens.to(torch.float32).unsqueeze(-1).expand(*tokens.shape, 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=forward_all_hidden,
    )

    out = RWKV7ForCausalLM.forward(
        model,
        torch.tensor([10, 20, 30, 31, 32], dtype=torch.int64),
        positions=None,
        **model_inputs,
    )

    assert calls == [
        ("tokens", [[10], [20]]),
        ("all_hidden", [[30, 31, 32]]),
    ]
    assert out.tolist() == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [31.0, 31.0, 31.0],
        [32.0, 32.0, 32.0],
    ]


def test_rwkv7_vllm_forward_rejects_sparse_active_decode_rows(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))
    seen = []

    def forward_tokens(tokens, state):
        seen.append(
            (
                tokens.tolist(),
                tuple(state[0].shape),
                state[0].storage_offset(),
            )
        )
        return torch.arange(tokens.shape[0] * 3, dtype=torch.float32).view(
            tokens.shape[0], 3
        )

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail("decode must stay T=1"),
    )
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)
    shift_state[:, :, 1].fill_(5)
    wkv_state[:, 1].fill_(7)
    elapsed[1] = 11

    with pytest.raises(RuntimeError, match="decode rows must match"):
        RWKV7ForCausalLM.forward(
            model,
            torch.tensor([20], dtype=torch.int64),
            positions=None,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            idx_mapping=torch.tensor([1], dtype=torch.int32),
            shift_state=shift_state,
            wkv_state=wkv_state,
            elapsed=elapsed,
            rwkv_decode_batch_size=2,
            rwkv_decode_rows=[1],
            rwkv_decode_token_positions=[0],
        )

    assert seen == []


def test_rwkv7_vllm_pp_non_last_stage_returns_v_first(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.start_layer = 0
    model.end_layer = 1
    model._is_pp_first_rank = lambda: True
    model._is_pp_last_rank = lambda: False
    model.embed = (
        lambda tokens: tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)
    )
    seen_groups = []

    def forward_layer_range(
        x, state, path, *, v_first, final, all_logits, last_indices
    ):
        assert v_first is None
        seen_groups.append(x[:, :, 0].tolist())
        state[0].fill_(5)
        state[1].fill_(7)
        state[2].fill_(11)
        return x + 1, x + 2

    model.forward_layer_range = forward_layer_range
    input_ids = torch.tensor([10, 20, 30, 31], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_prefill_token_ranges=[(2, 2, 4)],
        rwkv_prefill_rows=[2],
    )

    assert isinstance(out, IntermediateTensors)
    assert seen_groups == [[[10.0], [20.0]], [[30.0, 31.0]]]
    assert out["hidden_states"].tolist() == [
        [11.0, 11.0, 11.0],
        [21.0, 21.0, 21.0],
        [31.0, 31.0, 31.0],
        [32.0, 32.0, 32.0],
    ]
    assert out["v_first"].tolist() == [
        [12.0, 12.0, 12.0],
        [22.0, 22.0, 22.0],
        [32.0, 32.0, 32.0],
        [33.0, 33.0, 33.0],
    ]
    assert torch.all(shift_state == 5)
    assert torch.all(wkv_state == 7)
    assert elapsed.tolist() == [11, 11, 11]


def test_rwkv7_vllm_pp_non_last_stage_all_gathers_tp_v_first(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 4)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 2
    model.tp_rank = 0
    model.tp_hidden_size = 2
    model._is_pp_first_rank = lambda: True
    model._is_pp_last_rank = lambda: False
    model.embed = (
        lambda tokens: tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
    )

    def fake_all_gather(value):
        assert value.shape == (1, 1, 2)
        return torch.cat([value, value + 100], dim=-1)

    monkeypatch.setattr(rwkv7, "tensor_model_parallel_all_gather", fake_all_gather)

    def forward_layer_range(
        x, state, path, *, v_first, final, all_logits, last_indices
    ):
        assert v_first is None
        return x + 1, x[..., :2] + 2

    model.forward_layer_range = forward_layer_range
    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=torch.tensor([10], dtype=torch.int64),
        positions=None,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        shift_state=torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        wkv_state=torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32),
        elapsed=torch.zeros((1,), dtype=torch.int32),
    )

    assert isinstance(out, IntermediateTensors)
    assert out["hidden_states"].tolist() == [[11.0, 11.0, 11.0, 11.0]]
    assert out["v_first"].tolist() == [[12.0, 12.0, 112.0, 112.0]]


def test_rwkv7_vllm_pp_last_stage_uses_intermediate_tensors(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.start_layer = 1
    model.end_layer = 2
    model._is_pp_first_rank = lambda: False
    model._is_pp_last_rank = lambda: True
    seen_groups = []

    def forward_layer_range(
        x, state, path, *, v_first, final, all_logits, last_indices
    ):
        assert final
        assert all_logits
        assert last_indices is None
        seen_groups.append(
            (
                x[:, :, 0].tolist(),
                v_first[:, :, 0].tolist(),
            )
        )
        if x.shape[1] > 1:
            _assert_same_storage_view(state[0], shift_state[:, :, 2:3, :])
            _assert_same_storage_view(state[1], wkv_state[:, 2:3, :, :, :])
            _assert_same_storage_view(state[2], elapsed[2:3])
        state[0].fill_(13)
        state[1].fill_(17)
        state[2].fill_(19)
        if x.shape[1] > 1:
            assert torch.all(shift_state[:, :, 2:3] == 13)
            assert torch.all(wkv_state[:, 2:3] == 17)
            assert elapsed.tolist() == [19, 19, 19]
        return x + v_first, None

    model.forward_layer_range = forward_layer_range
    hidden_states = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    v_first = torch.full_like(hidden_states, 100)
    v_first[0].fill_(100)
    v_first[1].fill_(200)
    v_first[2].fill_(300)
    v_first[3].fill_(400)
    intermediate_tensors = IntermediateTensors(
        {"hidden_states": hidden_states, "v_first": v_first}
    )
    query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=None,
        positions=None,
        intermediate_tensors=intermediate_tensors,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_prefill_token_ranges=[(2, 2, 4)],
        rwkv_prefill_rows=[2],
    )

    assert isinstance(out, torch.Tensor)
    assert seen_groups == [
        ([[1.0], [4.0]], [[100.0], [200.0]]),
        ([[7.0, 10.0]], [[300.0, 400.0]]),
    ]
    assert out.tolist() == [
        [101.0, 102.0, 103.0],
        [204.0, 205.0, 206.0],
        [307.0, 308.0, 309.0],
        [410.0, 411.0, 412.0],
    ]
    assert torch.all(shift_state == 13)
    assert torch.all(wkv_state == 17)
    assert elapsed.tolist() == [19, 19, 19]


def test_rwkv7_vllm_pp_last_stage_slices_full_tp_v_first(monkeypatch):
    monkeypatch.setattr(rwkv7, "C", 4)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.start_layer = 1
    model.end_layer = 2
    model.tp_size = 2
    model.tp_rank = 1
    model.tp_hidden_size = 2
    model._is_pp_first_rank = lambda: False
    model._is_pp_last_rank = lambda: True

    def forward_layer_range(
        x, state, path, *, v_first, final, all_logits, last_indices
    ):
        assert v_first.tolist() == [[[30.0, 40.0]]]
        return x, None

    model.forward_layer_range = forward_layer_range
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.ones((1, 4), dtype=torch.float32),
            "v_first": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        }
    )

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=None,
        positions=None,
        intermediate_tensors=intermediate_tensors,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        shift_state=torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        wkv_state=torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32),
        elapsed=torch.zeros((1,), dtype=torch.int32),
    )

    assert isinstance(out, torch.Tensor)
    assert out.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_rwkv7_vllm_pp_last_stage_casts_intermediate_tensors_to_internal_dtype(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "C", 3)
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float16)
    monkeypatch.setattr(rwkv7, "first_device", lambda: torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.start_layer = 1
    model.end_layer = 2
    model._is_pp_first_rank = lambda: False
    model._is_pp_last_rank = lambda: True

    seen_dtypes = []

    def forward_layer_range(
        x, state, path, *, v_first, final, all_logits, last_indices
    ):
        seen_dtypes.append((x.dtype, v_first.dtype))
        return x + v_first, None

    model.forward_layer_range = forward_layer_range
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.ones((2, 3), dtype=torch.bfloat16),
            "v_first": torch.full((2, 3), 2, dtype=torch.bfloat16),
        }
    )
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 2, 3), dtype=torch.float16)
    wkv_state = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((2,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=None,
        positions=None,
        intermediate_tensors=intermediate_tensors,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
    )

    assert seen_dtypes == [(torch.float16, torch.float16)]
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, torch.full((2, 3), 3, dtype=torch.float16))
