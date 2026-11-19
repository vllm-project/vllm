# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import pytest
import torch

from vllm import ModelRegistry, SamplingParams
from vllm.config.cache import CacheConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.pp_transport.schema import (
    PPTransportSchema,
    PPTransportTensorSpec,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
)

from ..models.utils import check_logprobs_close
from ..utils import (
    compare_two_settings,
    create_new_process_for_each_test,
    multi_gpu_test,
    wait_for_memory_to_settle,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class Qwen3PPTransportForCausalLM(Qwen3ForCausalLM):
    """Exercise a mixed full/SP boundary without changing Qwen3's layers."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pp_transport_schema = PPTransportSchema(
            {
                "hidden_states": PPTransportTensorSpec(
                    (self.config.hidden_size,), sequence_parallel=True
                ),
                "residual": PPTransportTensorSpec((self.config.hidden_size,)),
            },
            tp_size=get_tp_group().world_size,
        )
        self.make_empty_intermediate_tensors = (
            self.pp_transport_schema.make_empty_intermediate_tensors
        )
        # Take an upstream tap so the last stage receives aux without a drafter.
        self.set_aux_hidden_state_layers((1,))
        reserve_aux_intermediate_tensor_slots(self)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        tp_group = get_tp_group()
        num_tokens = positions.shape[0]
        if intermediate_tensors is not None:
            intermediate_tensors = IntermediateTensors(
                {
                    **intermediate_tensors.tensors,
                    "hidden_states": tp_group.all_gather(
                        intermediate_tensors["hidden_states"], dim=0
                    )[:num_tokens],
                }
            )
        output = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if isinstance(output, IntermediateTensors):
            rows = (num_tokens + tp_group.world_size - 1) // tp_group.world_size
            hidden = torch.nn.functional.pad(
                output["hidden_states"],
                (0, 0, 0, rows * tp_group.world_size - num_tokens),
            )
            output["hidden_states"] = hidden.narrow(
                0, tp_group.rank_in_group * rows, rows
            ).contiguous()
        return output


ModelRegistry.register_model(
    "Qwen3PPTransportForCausalLM", f"{__name__}:Qwen3PPTransportForCausalLM"
)


class PPTransportWorkerExtension:
    """Register the test model in workers and inspect actual capture layouts."""

    model_runner: "GPUModelRunner"

    def pp_transport_capture_shapes(self):
        if get_pp_group().is_last_rank:
            return None
        manager = self.model_runner.cudagraph_manager
        return {
            desc.num_tokens: {key: tuple(value.shape) for key, value in output.items()}
            for desc, output in manager.captured_intermediate_tensors.items()
        }


@multi_gpu_test(num_gpus=4)
def test_pp_transport_preserves_sp_padding(vllm_runner, monkeypatch):
    """Keep all SP rows when one real token runs an eight-token CUDA graph."""
    monkeypatch.setenv("VLLM_PP_TRANSPORT", "nccl")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    # Keep the engine in this process so it shares the test model registry.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        logprobs=0,
        prompt_logprobs=0,
        detokenize=False,
    )
    results = {}
    for sharded, eager in ((False, False), (True, False), (True, True)):
        overrides = (
            {"architectures": ["Qwen3PPTransportForCausalLM"]} if sharded else {}
        )
        with vllm_runner(
            "Qwen/Qwen3-0.6B",
            dtype="bfloat16",
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            enforce_eager=eager,
            max_model_len=32,
            max_num_seqs=4,
            max_num_batched_tokens=32,
            kv_cache_memory_bytes=64 << 20,
            enable_prefix_caching=False,
            hf_overrides=overrides,
            attention_config={"backend": "TRITON_ATTN"},
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL",
                "prefill_cudagraph_capture_token_sizes": [8, 32],
            },
            worker_extension_cls=f"{__name__}.PPTransportWorkerExtension",
        ) as model:
            outputs = []
            for lengths in ((1,), (1, 5, 9), (1,)):
                prompts = [{"prompt_token_ids": [42] * n} for n in lengths]
                outputs.extend(model.llm.generate(prompts, params, use_tqdm=False))
            results[sharded, eager] = [
                (
                    output.outputs[0].token_ids,
                    output.outputs[0].logprobs,
                    output.prompt_logprobs,
                )
                for output in outputs
            ]
            assert all(len(output.outputs[0].token_ids) == 4 for output in outputs)
            if sharded and not eager:
                shapes = model.llm.collective_rpc("pp_transport_capture_shapes")
                senders = [shape for shape in shapes if shape is not None]
                assert len(senders) == 2
                for shape in senders:
                    assert shape[8]["hidden_states"][0] == 4
                    assert shape[8]["residual"][0] == 8
                    assert shape[8]["aux_hidden_states_0"][0] == 8
    assert results[True, False] == results[False, False]
    assert results[True, True] == results[False, False]


@pytest.mark.parametrize(
    "PP_SIZE, MODEL_NAME",
    [
        (2, "vllm-project/llama-160m"),
    ],
)
@pytest.mark.parametrize(
    "ATTN_BACKEND",
    [None] if current_platform.is_rocm() else ["FLASH_ATTN"],
)
@create_new_process_for_each_test()
def test_pp_cudagraph(
    PP_SIZE: int,
    MODEL_NAME: str,
    ATTN_BACKEND: str | None,
):
    if current_platform.is_rocm():
        # ROCm CI can hand this test a device that is still lazily
        # releasing VRAM from a previous process; wait before starting.
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION
        )

    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "mp",
    ]
    # On ROCm, defer to the platform attention selector instead of forcing a backend.
    if ATTN_BACKEND is not None:
        cudagraph_args.append(f"--attention-backend={ATTN_BACKEND}")

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="Below SM90, FlashInfer supports full CUDA graphs for spec-decode batches",
)
@create_new_process_for_each_test()
def test_pp_cudagraph_mode_differs_across_stages(
    vllm_runner, example_prompts, monkeypatch
):
    """Pipeline stages with different CUDA graph modes match eager outputs.

    The draft model's attention layers are on the last stage only. FlashInfer
    without TRT-LLM does not support full CUDA graphs for spec-decode batches, so
    the last stage uses PIECEWISE and the first stage uses FULL decode graphs,
    which pad decode batches to different sizes. NCCL is forced because the
    copy-engine transport receives directly into the persistent tensors.
    """
    monkeypatch.setenv("VLLM_PP_TRANSPORT", "nccl")
    outputs = {}
    for enforce_eager in (True, False):
        with vllm_runner(
            "Qwen/Qwen3-8B",
            trust_remote_code=False,
            max_model_len=2048,
            enforce_eager=enforce_eager,
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
            attention_config={"backend": "TRITON_ATTN", "use_trtllm_attention": False},
            speculative_config={
                "method": "eagle3",
                "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
                "num_speculative_tokens": 2,
                "attention_backend": "FLASHINFER",
            },
        ) as llm:
            outputs[enforce_eager] = llm.generate_greedy_logprobs(
                example_prompts[:4], max_tokens=32, num_logprobs=5
            )

    check_logprobs_close(
        outputs_0_lst=outputs[True],
        outputs_1_lst=outputs[False],
        name_0="eager",
        name_1="cudagraph",
    )
