# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.platforms import current_platform


def test_qwen3_5_models_use_platform_specific_implementation():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP
    from vllm.platforms import current_platform

    # NVIDIA and XPU use eager models; ROCm and other platforms
    # keep the previous torch.compile implementation in
    # vllm/model_executor/models/qwen3_5{,_mtp}.py.
    if current_platform.is_cuda():
        model_module = "vllm.models.qwen3_5.nvidia.model"
        mtp_module = "vllm.models.qwen3_5.nvidia.mtp"
    elif current_platform.is_xpu():
        model_module = "vllm.models.qwen3_5.xpu.model"
        mtp_module = "vllm.models.qwen3_5.xpu.mtp"
    else:
        model_module = "vllm.model_executor.models.qwen3_5"
        mtp_module = "vllm.model_executor.models.qwen3_5_mtp"
    assert Qwen3_5ForCausalLM.__module__ == model_module
    assert Qwen3_5MTP.__module__ == mtp_module


def test_qwen3_5_nvidia_models_do_not_use_torch_compile_wrapper():
    # Import the NVIDIA modules directly so the assertion is independent of
    # the platform this test runs on.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.models.qwen3_5.nvidia import model as model_module
    from vllm.models.qwen3_5.nvidia import mtp as mtp_module

    assert not issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert not issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_legacy_models_keep_torch_compile_wrapper():
    # The legacy implementation in vllm/model_executor/models/ serves ROCm and
    # other non-NVIDIA platforms; it must keep the torch.compile wrapper.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.model_executor.models import qwen3_5 as model_module
    from vllm.model_executor.models import qwen3_5_mtp as mtp_module

    assert issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_xpu_models_do_not_use_torch_compile_wrapper():
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.models.qwen3_5.xpu import model, mtp

    assert not issubclass(model.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(
        mtp.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert not issubclass(mtp.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def _check_xpu_moe_reload(worker):
    """Reload through the worker and replay a graph holding derived weights."""
    import torch

    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    model = worker.model_runner.get_model()
    block = next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
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


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
def test_xpu_qwen3_5_moe_reload_graph_tp2(vllm_runner, enable_pickle):
    """Real weights: native TP2 graph replay observes in-place kernel reload."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig

    with vllm_runner(
        os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dtype="float16",
        quantization="fp8_per_tensor",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1]
        ),
    ) as model:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids
        assert model.collective_rpc(_check_xpu_moe_reload) == [2, 2]
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids
