# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import pytest
import torch

from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.platforms import current_platform


@dataclass
class _KernelUpdateInfo(WeightTransferUpdateInfo):
    weights: list[tuple[str, torch.Tensor]]


class _InProcessKernelTransferEngine(
    WeightTransferEngine[WeightTransferInitInfo, _KernelUpdateInfo]
):
    """Replace external transport while keeping real worker update sessions."""

    init_info_cls = WeightTransferInitInfo
    update_info_cls = _KernelUpdateInfo

    def init_transfer_engine(self, init_info: WeightTransferInitInfo) -> None:
        pass

    def start_weight_update(self) -> None:
        pass

    def finish_weight_update(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @torch.no_grad()
    def receive_weights(self, update_info: _KernelUpdateInfo) -> None:
        from vllm.model_executor.model_loader.reload.inplace import (
            get_parameter_for_reload,
        )

        for name, weight in update_info.weights:
            get_parameter_for_reload(self.model, name).copy_(weight)


def _check_xpu_moe_reload(worker):
    """Reload through the worker and replay a graph holding model parameters."""
    import torch

    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    model = worker.model_runner.get_model()
    block = next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
    assert block._xpu_moe_shared_fused_plan is not None, "Prepare before graph capture"
    config = worker.vllm_config
    x = torch.ones(1, block.gate.input_size, dtype=torch.float16, device=worker.device)
    stream = torch.xpu.Stream(device=worker.device)
    stream.wait_stream(torch.xpu.current_stream())
    with torch.inference_mode(), set_forward_context(None, config):
        with torch.xpu.stream(stream):
            block(x)
            block(x)
        torch.xpu.current_stream().wait_stream(stream)
        plan = block._xpu_moe_shared_fused_plan
        assert plan is not None, "The native provider must run in this test"
        workspace = plan.workspace(1)
        weight_ptr = plan.shared_w2.data_ptr()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            output = block(x)
        graph.replay()
        torch.xpu.synchronize()
        before = output.clone()
        weight = block.shared_expert.down_proj.weight
        assert plan.shared_w2 is weight
        name = next(n for n, p in model.named_parameters() if p is weight)
        original = weight.clone()
        try:
            worker.reload_weights(
                [(name, torch.zeros_like(weight))], is_checkpoint_format=False
            )
            torch.testing.assert_close(plan.shared_w2, weight, rtol=0, atol=0)
            assert block._xpu_moe_shared_fused_plan is plan
            assert plan.shared_w2.data_ptr() == weight_ptr
            assert plan.workspace(1) is workspace
            x.fill_(0.5)
            expected = Qwen3NextSparseMoeBlock.forward(block, x)
            graph.replay()
            torch.xpu.synchronize()
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
            assert not torch.equal(output, before)
        finally:
            worker.reload_weights([(name, original)], is_checkpoint_format=False)
        graph.replay()
        torch.xpu.synchronize()
        expected = Qwen3NextSparseMoeBlock.forward(block, x)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    return block.tp_size


def _check_xpu_moe_weight_updates(worker):
    """Target and draft update sessions update the selected graph's parameters."""
    from vllm.config import WeightTransferConfig
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    worker = worker.worker
    models = [worker.get_model(), worker.get_draft_model()]
    assert models[1] is not None
    blocks = [
        next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
        for model in models
    ]
    original_engine = worker.weight_transfer_engine
    worker.weight_transfer_engine = _InProcessKernelTransferEngine(
        WeightTransferConfig(backend="ipc"),
        worker.vllm_config,
        worker.device,
        models[0],
    )
    try:
        for is_draft in (False, True):
            model, block = models[is_draft], blocks[is_draft]
            weight = block.shared_expert.down_proj.weight
            other_weight = blocks[not is_draft].shared_expert.down_proj.weight
            assert weight is not other_weight
            original, untouched = weight.clone(), other_weight.clone()
            name = next(n for n, p in model.named_parameters() if p is weight)
            start = (
                worker.start_draft_weight_update
                if is_draft
                else worker.start_weight_update
            )
            x = torch.ones(
                1, block.gate.input_size, dtype=torch.float16, device=worker.device
            )
            stream = torch.xpu.Stream(device=worker.device)
            stream.wait_stream(torch.xpu.current_stream())
            with torch.inference_mode(), set_forward_context(None, worker.vllm_config):
                with torch.xpu.stream(stream):
                    block(x)
                    block(x)
                torch.xpu.current_stream().wait_stream(stream)
                plan = block._xpu_moe_shared_fused_plan
                assert plan is not None
                weight_ptr = plan.shared_w2.data_ptr()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=stream):
                    output = block(x)
                graph.replay()
                torch.xpu.synchronize()
                before = output.clone()
                try:
                    start()
                    worker.update_weights(
                        {"weights": [(name, torch.zeros_like(weight))]}
                    )
                    worker.finish_weight_update()
                    torch.testing.assert_close(plan.shared_w2, weight, rtol=0, atol=0)
                    torch.testing.assert_close(other_weight, untouched, rtol=0, atol=0)
                    assert block._xpu_moe_shared_fused_plan is plan
                    assert plan.shared_w2.data_ptr() == weight_ptr
                    expected = Qwen3NextSparseMoeBlock.forward(block, x)
                    graph.replay()
                    torch.xpu.synchronize()
                    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
                    assert not torch.equal(output, before)
                finally:
                    start()
                    worker.update_weights({"weights": [(name, original)]})
                    worker.finish_weight_update()
                graph.replay()
                torch.xpu.synchronize()
                torch.testing.assert_close(output, before, rtol=0, atol=0)
    finally:
        worker.weight_transfer_engine = original_engine
    return blocks[0].tp_size


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
def test_xpu_qwen3_5_moe_reload_graph_tp2(vllm_runner, enable_pickle):
    """Real TP2 target/draft graphs observe reloads and weight-transfer updates."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig

    with vllm_runner(
        os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dtype="float16",
        quantization="fp8_per_tensor",
        tensor_parallel_size=2,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        distributed_executor_backend="mp",
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[2]
        ),
    ) as model:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids
        assert model.collective_rpc(_check_xpu_moe_reload) == [2, 2]
        assert model.collective_rpc(_check_xpu_moe_weight_updates) == [2, 2]
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids


def _check_xpu_moe_a8_shared_expert(worker):
    from tests.kernels.moe.test_moe import _check_xpu_qwen3_5_decode
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    model = worker.model_runner.get_model()
    block = next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
    for num_tokens in (1, 8):
        _check_xpu_qwen3_5_decode(
            (block, worker.vllm_config, True), num_tokens, should_fuse=False
        )
    return block.tp_size


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
def test_xpu_qwen3_5_moe_a8_shared_expert_fallback(vllm_runner, enable_pickle):
    """Real TP2 decode retains A8 shared projections, including graph replay."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig, KernelConfig

    with vllm_runner(
        os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dtype="float16",
        quantization="fp8_per_tensor",
        tensor_parallel_size=2,
        kernel_config=KernelConfig(linear_backend="xpu"),
        distributed_executor_backend="mp",
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 8]
        ),
    ) as model:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        outputs = model.llm.generate(["Hello"] * 8, params)
        assert all(len(output.outputs[0].token_ids) == 4 for output in outputs)
        assert model.collective_rpc(_check_xpu_moe_a8_shared_expert) == [2, 2]
