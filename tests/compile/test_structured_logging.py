# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import regex as re
import torch
from torch import nn

import tests.compile.silly_attention  # noqa
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from vllm.config.scheduler import SchedulerConfig
from vllm.forward_context import set_forward_context

MLP_SIZE = 64


@support_torch_compile
class SimpleModel(nn.Module):
    """A simple model with a splitting op for piecewise compilation."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output * 2
        return x


class TraceStructuredCapture:
    """Captures trace_structured calls for testing."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, event_type: str, metadata_fn=None, payload_fn=None, **kwargs):
        """Capture a trace_structured call."""
        metadata = metadata_fn() if metadata_fn else {}
        self.calls.append(
            {
                "event_type": event_type,
                "metadata": metadata,
            }
        )

    def get(self, event_type: str, name_pattern: str) -> list[dict]:
        """Get all calls with the given event type and name matching pattern.

        Args:
            event_type: The event type to filter by (e.g., "artifact", "graph_dump")
            name_pattern: Regex pattern to match against the artifact name
        """
        regex = re.compile(name_pattern)
        return [
            c
            for c in self.calls
            if c["event_type"] == event_type
            and regex.fullmatch(c.get("metadata", {}).get("name", ""))
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vllm_structured_logging_artifacts(use_fresh_inductor_cache):
    """Test that all expected vLLM artifacts are logged during compilation."""
    torch.set_default_device("cuda")

    capture = TraceStructuredCapture()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            compile_sizes=[8],
            splitting_ops=["silly::attention"],
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
    )

    # Patch trace_structured to capture calls
    with (
        patch("vllm.compilation.backends.trace_structured", capture),
        patch("vllm.compilation.piecewise_backend.trace_structured", capture),
        set_current_vllm_config(vllm_config),
    ):
        model = SimpleModel(vllm_config=vllm_config, prefix="test")
        with set_forward_context({}, vllm_config=vllm_config):
            model(torch.randn(8, MLP_SIZE))

    config_artifacts = capture.get("artifact", "vllm_compilation_config")
    assert (
        len(config_artifacts) == 1
    ), f"Expected 1 vllm_compilation_config, got {len(config_artifacts)}"
    vllm_piecewise_split_graph = capture.get("graph_dump", "vllm_piecewise_split_graph")
    assert len(vllm_piecewise_split_graph) == 1, (
        "Expected 1 toplevel piecewise split graph, "
        f"got {len(vllm_piecewise_split_graph)}"
    )
    compile_start_artifacts = capture.get("artifact", "vllm_piecewise_compile_start")
    assert len(compile_start_artifacts) == 2, (
        "Expected 2 vllm_piecewise_compile_start "
        "(one for dynamic ranges, one for compile size), "
        f"got {len(compile_start_artifacts)}"
    )
    submod_dumps = capture.get("graph_dump", r"vllm_submod_.*")
    assert len(submod_dumps) == 2, (
        "Expected 2 submods (one before attention, one after attention), "
        f"got {len(submod_dumps)}"
    )
