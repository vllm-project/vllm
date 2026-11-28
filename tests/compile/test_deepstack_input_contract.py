# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import (
    create_forward_context,
    override_forward_context,
    set_forward_context,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.ubatching import make_ubatch_contexts


@support_torch_compile(
    dynamic_arg_dims={"x": 0, "deepstack_input_embeds": 0},
)
class _DeepStackDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        deepstack_input_embeds: IntermediateTensors | None,
    ) -> torch.Tensor:
        if deepstack_input_embeds is not None:
            x = x + deepstack_input_embeds["deepstack_input_embeds_0"]
        return x


def _run_contract(
    deepstack_buffer: Qwen3VLForConditionalGeneration,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with set_current_vllm_config(config):
        model = _DeepStackDecoder(vllm_config=config).eval().cuda()

    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    with torch.inference_mode(), set_forward_context(None, vllm_config=config):
        empty_payload = deepstack_buffer._get_deepstack_input_embeds(num_tokens=2)
        first = model(x, empty_payload)

        payload = torch.full((1, 2, 4), 7.0, device="cuda")
        deepstack_buffer._set_deepstack_input_embeds(payload)
        active_payload = deepstack_buffer._get_deepstack_input_embeds(num_tokens=2)
        assert active_payload is not None
        second = model(x, active_payload)

    return x.cpu(), first.cpu(), second.cpu()


def test_qwen3_vl_deepstack_payload_after_empty_compile(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    deepstack_buffer = object.__new__(Qwen3VLForConditionalGeneration)
    deepstack_buffer.deepstack_input_embeds = [torch.zeros(2, 4, device="cuda")]
    deepstack_buffer.deepstack_input_embeds_num_tokens = 0
    deepstack_buffer.deepstack_num_level = 1

    x, first, second = _run_contract(deepstack_buffer)
    torch.testing.assert_close(first, x)
    torch.testing.assert_close(second, x + 7)


@torch.inference_mode()
def _check_deepstack_microbatch_rows(model: nn.Module) -> None:
    assert isinstance(model, Qwen3VLForConditionalGeneration)
    assert model.deepstack_num_level == 1
    parameter = next(model.parameters())
    rows = torch.arange(1, 5, device=parameter.device, dtype=parameter.dtype)
    payload = rows.view(1, 4, 1).expand(1, 4, model.config.text_config.hidden_size)
    model._set_deepstack_input_embeds(payload)

    config = VllmConfig()
    stream = current_stream()
    contexts = make_ubatch_contexts(
        num_micro_batches=2,
        compute_stream=stream,
        comm_stream=stream,
        forward_contexts=[create_forward_context(None, config) for _ in range(2)],
        ready_barrier=threading.Barrier(1, timeout=5),
        token_offsets=[0, 2],
    )
    with override_forward_context(None):
        contexts[0].cpu_wait_event.set()
        for context, expected in zip(contexts, payload[0].split(2), strict=True):
            assert context.cpu_wait_event.is_set()
            with context:
                scratch = model._get_deepstack_input_embeds(num_tokens=2)
                assert scratch is not None
                torch.testing.assert_close(
                    scratch["deepstack_input_embeds_0"], expected, atol=0, rtol=0
                )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Microbatching needs CUDA or ROCm"
)
@pytest.mark.usefixtures("enable_pickle")
def test_qwen3_vl_deepstack_reads_microbatch_rows(vllm_runner):
    with vllm_runner(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="bfloat16",
        enforce_eager=True,
        load_format="dummy",
        hf_overrides={
            "text_config": {"num_hidden_layers": 2},
            "vision_config": {"depth": 1, "deepstack_visual_indexes": [0]},
        },
        max_model_len=1024,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1, "video": 0},
        skip_mm_profiling=True,
    ) as model:
        model.apply_model(_check_deepstack_microbatch_rows)
