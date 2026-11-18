# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA and generation regressions for last-layer FFN pruning."""

import json
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.v1.worker.gpu.last_layer_ffn import (
    LastLayerFFNGraphWrapper,
    last_layer_ffn_graph_key,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_first_capture_executes_and_replay_uses_current_rows(dist_init, monkeypatch):
    """New row-count graphs must execute immediately and reuse live indices."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = VllmConfig()

    def select(x, last_layer_token_indices):
        return x.index_select(0, last_layer_token_indices) + 1

    graph = LastLayerFFNGraphWrapper(select, config, key_fn=last_layer_ffn_graph_key)
    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    indices = torch.tensor([0, 2], device="cuda")
    with set_forward_context(
        None,
        config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=4),
    ):
        for rows, offset, selected_rows in [
            (1, 0, [0, 2]),
            (2, 16, [1, 3]),
            (1, 32, [3, 0]),
            (2, 48, [2, 1]),
        ]:
            x.copy_(torch.arange(16, device="cuda").reshape(4, 4) + offset)
            indices.copy_(torch.tensor(selected_rows, device="cuda"))
            selected = indices[:rows]
            actual = graph(x, last_layer_token_indices=selected)
            torch.testing.assert_close(actual, select(x, selected), rtol=0, atol=0)
    assert len(graph.entries) == 2


class FFNTestWorkerExtension:
    model_runner: "GPUModelRunner"

    def ffn_test_stats(self):
        runner = self.model_runner
        pruner = runner.last_layer_ffn_pruner
        manager = runner.cudagraph_manager
        graph = manager.breakable_cg_runner if manager is not None else None
        return {
            "last_stage": runner.is_last_pp_rank,
            "batches": pruner.batches,
            "rows": pruner.skipped_rows,
            "graphs": len(graph.entries) if graph is not None else 0,
        }


def _stats(model):
    return model.llm.collective_rpc("ffn_test_stats")[0]


def _make_adapters(tmp_path, model_name):
    """Nonzero FFN adapters expose incorrect token-to-adapter row mappings."""
    config = AutoConfig.from_pretrained(maybe_model_redirect(model_name))
    adapters = []
    for adapter_id in (1, 2):
        path = tmp_path / f"adapter-{adapter_id}"
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 8,
                    "lora_alpha": 8,
                    "target_modules": ["gate_proj", "up_proj", "down_proj"],
                    "bias": "none",
                    "base_model_name_or_path": model_name,
                }
            )
        )
        rng = torch.Generator().manual_seed(adapter_id)
        weights = {}
        for layer in range(config.num_hidden_layers - 2, config.num_hidden_layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                input_size, output_size = config.hidden_size, config.intermediate_size
                if name == "down_proj":
                    input_size, output_size = output_size, input_size
                prefix = f"base_model.model.model.layers.{layer}.mlp.{name}"
                weights[f"{prefix}.lora_A.weight"] = (
                    torch.randn(8, input_size, generator=rng) * 0.02
                )
                weights[f"{prefix}.lora_B.weight"] = (
                    torch.randn(output_size, 8, generator=rng) * 0.02
                )
        save_file(weights, path / "adapter_model.safetensors")
        adapters.append(LoRARequest(f"adapter-{adapter_id}", adapter_id, str(path)))
    return adapters


@pytest.mark.parametrize(
    "model_name",
    ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"],
)
@pytest.mark.parametrize(
    "mode", ["eager", "graph", "prefix", "chunked", "lora_eager", "lora_graph"]
)
def test_pruning_preserves_generation_and_fallbacks(
    vllm_runner, monkeypatch, tmp_path, mode, model_name
):
    """Retain sampled outputs with cached/chunked inputs and mixed adapters."""
    eager = mode in ("eager", "lora_eager")
    lora = mode.startswith("lora_")
    adapters = _make_adapters(tmp_path, model_name) if lora else []
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0" if eager else "1")
    chunked = mode == "chunked"
    prefix = mode == "prefix"
    prompts = [
        "The following is a short arithmetic exercise. " * 4 + "7 times 8 equals",
        "The following is a short geography exercise. " * 4
        + "The capital of France is",
    ]
    if chunked:
        prompts = [p * 4 for p in prompts]
    outputs = []
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_ENABLE_LAST_LAYER_FFN_PRUNING", str(int(enabled)))
        with vllm_runner(
            maybe_model_redirect(model_name),
            dtype="bfloat16",
            max_model_len=1024,
            max_num_seqs=2,
            max_num_batched_tokens=64 if chunked else 1024,
            gpu_memory_utilization=0.8,
            enforce_eager=eager,
            enable_lora=lora,
            max_loras=2,
            max_lora_rank=8,
            enable_prefix_caching=prefix,
            enable_chunked_prefill=chunked,
            attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "prefill_cudagraph_capture_token_sizes": [32, 64]
                if chunked
                else [128, 256],
                "decode_cudagraph_capture_request_sizes": [1, 2],
            },
            worker_extension_cls=f"{__name__}.FFNTestWorkerExtension",
        ) as model:
            initial = _stats(model)
            arm = []
            # Revisit the one-request variant after capturing the two-request one.
            for i, batch in enumerate(([prompts[0]], prompts, [prompts[1]], prompts)):
                kwargs = {}
                if lora:
                    kwargs["lora_request"] = (
                        [adapters[0]],
                        [adapters[1], adapters[0]],
                        [None],
                        [adapters[0], adapters[1]],
                    )[i]
                arm.append(model.generate_greedy_logprobs(batch, 8, 5, **kwargs))
            after = _stats(model)
            if enabled:
                assert after["batches"] > initial["batches"]
                assert after["rows"] > initial["rows"]
            else:
                assert after["batches"] == initial["batches"]
            if not eager:
                assert after["graphs"] > 0
            if prefix:
                result = model.llm.generate(
                    [p + " Explain the answer carefully." * 12 for p in prompts],
                    SamplingParams(temperature=0, max_tokens=1, logprobs=5),
                    use_tqdm=False,
                )
                assert all(r.num_cached_tokens > 0 for r in result)
                if enabled:
                    assert _stats(model)["batches"] > after["batches"]
                arm.append(
                    [
                        (
                            list(r.outputs[0].token_ids),
                            r.outputs[0].text,
                            r.outputs[0].logprobs,
                        )
                        for r in result
                    ]
                )
            # Prompt scoring consumes earlier rows, so pruning must be bypassed.
            before_scoring = _stats(model)
            arm.append(
                model.generate_greedy_logprobs(
                    ["A new prompt for scoring: the first prime numbers are"],
                    8,
                    5,
                    num_prompt_logprobs=1,
                )
            )
            assert _stats(model)["batches"] == before_scoring["batches"]
            outputs.append(arm)
    for stock, pruned in zip(*outputs):
        check_logprobs_close(
            outputs_0_lst=stock,
            outputs_1_lst=pruned,
            name_0="stock",
            name_1="last-layer-ffn",
        )


@pytest.mark.distributed
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize(
    "model_name",
    ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B-Instruct"],
)
def test_parallel_pruning_preserves_generation(
    vllm_runner, monkeypatch, model_name, tp, pp, eager
):
    """TP ranks agree on retained rows; only the final PP stage prunes."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0" if eager else "1")
    prompts = [
        "Explain why the sky appears blue. " * 20,
        "Explain why plants need sunlight. " * 16,
    ]
    outputs = []
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_ENABLE_LAST_LAYER_FFN_PRUNING", str(int(enabled)))
        with vllm_runner(
            maybe_model_redirect(model_name),
            dtype="bfloat16",
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            distributed_executor_backend="mp",
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            enforce_eager=eager,
            gpu_memory_utilization=0.8,
            attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "prefill_cudagraph_capture_token_sizes": [32, 64],
                "decode_cudagraph_capture_request_sizes": [1, 2],
            },
            worker_extension_cls=f"{__name__}.FFNTestWorkerExtension",
        ) as model:
            outputs.append(model.generate_greedy_logprobs(prompts, 8, 5))
            stats = model.llm.collective_rpc("ffn_test_stats")
            assert len(stats) == tp * pp
            for rank in stats:
                if enabled and rank["last_stage"]:
                    assert rank["batches"] > 0 and rank["rows"] > 0
                else:
                    assert rank["batches"] == 0
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="stock",
        name_1="last-layer-ffn",
    )
