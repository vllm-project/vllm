# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod

import pytest
import torch

from vllm import SamplingParams
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.worker import workspace
from vllm.v1.worker.gpu_worker import Worker

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FlashInfer KDA prefill requires SM100 or SM103",
)


@pytest.fixture
def model_path(tmp_path):
    config = KimiLinearConfig(
        architectures=["KimiLinearForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        linear_attn_config={
            "kda_layers": [1, 2],
            "full_attn_layers": [],
            "head_dim": 128,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path)


def _engine_args(model_path, backend, num_tokens=64):
    return dict(
        model=model_path,
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=32,
        max_num_seqs=4,
        max_num_batched_tokens=num_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="bfloat16",
        enforce_eager=True,
        kernel_config={"enable_jit_warmup": False},
        additional_config={
            "kda_prefill_backend": backend,
            "kda_decode_backend": "triton",
            "kda_spec_decode_backend": "native",
        },
    )


@pytest.fixture
def make_kda_layer(model_path, dist_init):
    def make(backend, num_tokens=64):
        config = EngineArgs(
            **_engine_args(model_path, backend, num_tokens)
        ).create_engine_config()
        with set_current_vllm_config(config), torch.device("cuda"):
            return KimiK3DeltaAttention(
                config.model_config.hf_text_config,
                config,
                prefix="model.layers.0.self_attn",
            )

    return make


def _prefill_specs(layer):
    if layer._flashkda_buffer_specs is not None:
        return layer._flashkda_buffer_specs
    assert layer._flashinfer_kda_output_spec is not None
    return (layer._flashinfer_kda_output_spec,)


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer"])
def test_prefill_workspace_reserved_and_shared(make_kda_layer, workspace_init, backend):
    layer = make_kda_layer(backend)
    specs = _prefill_specs(layer)
    assert specs[0] == ((1, 64, 2, 128), torch.bfloat16)
    if backend == "flashkda":
        assert specs[1:3] == (((4, 2, 128, 128), torch.bfloat16),) * 2
        assert specs[3] == (
            (torch.ops._flashkda_C.get_workspace_size(64, 2, 4),),
            torch.uint8,
        )
    else:
        assert len(specs) == 1

    manager = workspace.current_workspace_manager()
    reserved = manager._current_workspaces[0]
    assert reserved is not None
    assert reserved.numel() == sum(
        ((prod(shape) * dtype.itemsize + 255) // 256) * 256 for shape, dtype in specs
    )
    manager.lock()
    make_kda_layer(backend)
    buffers = manager.get_simultaneous(*specs)
    assert manager._current_workspaces[0] is reserved
    assert all(
        buf.untyped_storage().data_ptr() == reserved.data_ptr() for buf in buffers
    )


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer"])
def test_prefill_workspace_reservation_only_grows(
    make_kda_layer, workspace_init, backend
):
    make_kda_layer(backend)
    manager = workspace.current_workspace_manager()
    smaller = manager._current_workspaces[0]
    assert smaller is not None
    make_kda_layer(backend, num_tokens=256)
    larger = manager._current_workspaces[0]
    assert larger is not None
    assert larger.numel() > smaller.numel()
    make_kda_layer(backend)
    assert manager._current_workspaces[0] is larger


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer", "triton"])
def test_prefill_construction_without_workspace_manager(
    make_kda_layer, workspace_init, backend
):
    workspace.reset_workspace_manager()
    layer = make_kda_layer(backend)
    assert layer.kda_prefill_backend == backend
    assert not workspace.is_workspace_manager_initialized()


def test_triton_prefill_does_not_reserve_workspace(make_kda_layer, workspace_init):
    layer = make_kda_layer("triton")
    assert layer.kda_prefill_backend == "triton"
    assert workspace.current_workspace_manager()._current_workspaces == [None]


class WorkspaceCheckingWorker(Worker):
    def determine_available_memory(self) -> int:
        """Verify the real loaded model already fits before profiling runs."""
        layers = [
            layer
            for layer in self.model_runner.get_model().modules()
            if isinstance(layer, KimiK3DeltaAttention)
        ]
        assert len(layers) == 2
        manager = workspace.current_workspace_manager()
        manager.lock()
        try:
            for layer in layers:
                manager.get_simultaneous(*_prefill_specs(layer))
        finally:
            manager.unlock()
        return super().determine_available_memory()


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer"])
def test_engine_reserves_prefill_workspace_before_profiling(
    vllm_runner, model_path, monkeypatch, backend
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    args = _engine_args(model_path, backend)
    model = args.pop("model")
    with vllm_runner(
        model,
        gpu_memory_utilization=0.4,
        worker_cls=("tests.models.kimi_k3.test_kda_workspace.WorkspaceCheckingWorker"),
        **args,
    ) as runner:
        outputs = runner.get_llm().generate(
            {"prompt_token_ids": [1, 3, 5, 7]},
            SamplingParams(
                temperature=0, max_tokens=2, ignore_eos=True, detokenize=False
            ),
        )
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == 2
