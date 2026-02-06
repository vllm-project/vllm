# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from vllm.compilation.counter import compilation_counter
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.compilation import CompilationMode, PassConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _is_torch_equal_or_newer,
    is_torch_equal,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401


def test_version():
    # Test the version comparison logic using the private function
    assert _is_torch_equal_or_newer("2.8.0.dev20250624+cu128", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0a0+gitc82a174", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.1", "2.8.0.dev")
    assert not _is_torch_equal_or_newer("2.7.1", "2.8.0.dev")


def test_get_raw_stream_patch():
    """Test that get_raw_stream patch is applied only for torch 2.9.0 or 2.9.1."""
    import builtins

    # Check if get_raw_stream exists in builtins
    has_patch = hasattr(builtins, "get_raw_stream")

    # Import torch to get actual version

    is_torch_2_9 = is_torch_equal("2.9.0") or is_torch_equal("2.9.1")

    if is_torch_2_9:
        # For torch 2.9.x, the patch should be applied
        assert has_patch, "get_raw_stream should be patched for torch 2.9.x"
        # Verify it's callable (it should be the _cuda_getCurrentRawStream function)
        get_raw_stream = builtins.get_raw_stream  # type: ignore[attr-defined]
        assert callable(get_raw_stream)
        # Verify it's the correct function from torch._C
        from torch._C import _cuda_getCurrentRawStream

        assert get_raw_stream is _cuda_getCurrentRawStream


def test_copy_pass():
    vllm_config = VllmConfig()
    inductor_pass = FixFunctionalizationPass(vllm_config)
    copied_inductor_pass = copy.deepcopy(inductor_pass)
    assert (
        copied_inductor_pass.compilation_config.use_inductor_graph_partition
        == vllm_config.compilation_config.use_inductor_graph_partition
    )
    assert (
        copied_inductor_pass.compilation_config.splitting_ops
        == vllm_config.compilation_config.splitting_ops
    )


def test_custom_op():
    # proper syntax
    _ = CompilationConfig(custom_ops=["+quant_fp8", "-silu_and_mul"])

    with pytest.raises(ValueError, match="Invalid syntax '"):
        _ = CompilationConfig(custom_ops=["quant_fp8"])


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
# NB: We don't test VLLM_DISABLE_COMPILE_CACHE=0 because that depends
# on the state of the cache directory on the current machine, which
# may be influenced by other tests.
@pytest.mark.parametrize("val", ["1"])
def test_VLLM_DISABLE_COMPILE_CACHE(vllm_runner, monkeypatch, val):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", val)

    compilation_config = {
        "cudagraph_mode": CUDAGraphMode.NONE,  # speed things up a bit
    }
    with (
        compilation_counter.expect(
            num_cache_entries_updated=0, num_compiled_artifacts_saved=0
        ),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m",
            compilation_config=compilation_config,
            gpu_memory_utilization=0.4,
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.parametrize(
    "cudagraph_mode,num_cudagraph_captured",
    [
        (CUDAGraphMode.NONE, 0),
        (CUDAGraphMode.FULL_DECODE_ONLY, 1),
        (CUDAGraphMode.PIECEWISE, 13),
        (CUDAGraphMode.FULL_AND_PIECEWISE, 14),
    ],
)
def test_use_cudagraphs(
    vllm_runner, monkeypatch, cudagraph_mode, num_cudagraph_captured
):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    compilation_config = {
        "cudagraph_capture_sizes": [100],
        "cudagraph_mode": cudagraph_mode,
    }
    num_gpu_runner_capture_triggers = 1 if cudagraph_mode != CUDAGraphMode.NONE else 0
    with (
        compilation_counter.expect(
            num_graphs_seen=1,
            num_gpu_runner_capture_triggers=num_gpu_runner_capture_triggers,
            num_cudagraph_captured=num_cudagraph_captured,
        ),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m",
            compilation_config=compilation_config,
            gpu_memory_utilization=0.4,
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
def test_stock_torch_compile(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        compilation_counter.expect(stock_torch_compile_count=1),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m",
            compilation_config={"mode": CompilationMode.STOCK_TORCH_COMPILE},
            gpu_memory_utilization=0.4,
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
def test_no_compilation(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with (
        compilation_counter.expect(num_graphs_seen=0, stock_torch_compile_count=0),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m",
            compilation_config={"mode": CompilationMode.NONE},
            gpu_memory_utilization=0.4,
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
def test_enforce_eager(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        compilation_counter.expect(num_graphs_seen=0, stock_torch_compile_count=0),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.4
        ) as _,
    ):
        pass


def test_splitting_ops_dynamic():
    # Default config
    config = VllmConfig()
    # Default V1 config leaves cudagraph mode unset; splitting ops are only
    # populated when the engine decides to use piecewise compilation.
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    assert config.compilation_config.splitting_ops_contain_attention()

    # When use_inductor_graph_partition=True
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            splitting_ops=["vllm::unified_attention"],
        )
    )
    # with inductor partition we use splitting_ops directly for
    # partition rules
    assert config.compilation_config.splitting_ops == ["vllm::unified_attention"]

    # When attn_fusion pass enabled.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_attn_quant=True, eliminate_noops=True),
            custom_ops=["+quant_fp8"],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    assert config.compilation_config.splitting_ops == []
    # cudagraph mode also fall back to FULL
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL

    # splitting_ops can not contain attention ops when attn_fusion
    # pass enabled.
    with pytest.raises(ValidationError):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                pass_config=PassConfig(fuse_attn_quant=True, eliminate_noops=True),
                custom_ops=["+quant_fp8"],
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                # work around for accessing all attntion ops
                splitting_ops=CompilationConfig()._attention_ops,
            )
        )

    # When both use_inductor_graph_partition and attn_fusion pass enabled.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            pass_config=PassConfig(fuse_attn_quant=True, eliminate_noops=True),
            custom_ops=["+quant_fp8"],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    # With inductor graph partition, attn_fusion and splitting_ops
    # work together. Default splitting_ops include attention ops.
    assert config.compilation_config.splitting_ops_contain_attention()
    # fuse_attn_quant is directly supported under
    # use_inductor_graph_partition=True, and cudagraph_mode
    # is unchanged.
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_moe_splitting_ops_deepep_ht_inductor_partition():
    # Inductor partition case: user-provided splitting_ops should be
    # preserved and MoE ops should be appended for DeepEP HT with dp>1.
    config = VllmConfig(
        parallel_config=ParallelConfig(
            all2all_backend="deepep_high_throughput",
            data_parallel_size=8,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            splitting_ops=[
                "vllm::unified_attention",
                "vllm::moe_forward",
                "vllm::moe_forward_shared",
            ],
        ),
    )
    splitting_ops = config.compilation_config.splitting_ops
    assert splitting_ops == [
        "vllm::unified_attention",
        "vllm::moe_forward",
        "vllm::moe_forward_shared",
    ]


def test_should_split():
    import torch

    from vllm.compilation.partition_rules import should_split

    graph = torch.fx.Graph()
    node = torch.fx.Node(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.ops.aten.add.default,
        args=(),
        kwargs={},
    )

    # supports OpOverloadPacket
    splitting_ops = ["aten::add"]
    assert should_split(node, splitting_ops)

    # supports OpOverload
    splitting_ops = ["aten::add.default"]
    assert should_split(node, splitting_ops)

    # supports OpOverload
    splitting_ops = ["aten::add.Tensor"]
    assert not should_split(node, splitting_ops)

    q, k, v, out = [torch.randn(1)] * 4

    # supports custom ops as OpOverloadPacket
    node = torch.fx.Node(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.ops.silly.attention,
        args=(q, k, v, out),
        kwargs={},
    )

    splitting_ops = ["silly::attention"]
    assert should_split(node, splitting_ops)

    # supports custom ops as OpOverload
    node = torch.fx.Node(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.ops.silly.attention.default,
        args=(q, k, v, out),
        kwargs={},
    )

    splitting_ops = ["silly::attention"]
    assert should_split(node, splitting_ops)

    splitting_ops = ["silly::attention.default"]
    assert should_split(node, splitting_ops)


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if not cudagraph mode supported",
)
@pytest.mark.parametrize(
    (
        "cudagraph_capture_sizes",
        "max_cudagraph_capture_size",
        "tp_size",
        "enable_sp",
        "max_num_batched_tokens",
        "cudagraph_mode",
        "expected_max_size",
    ),
    [
        (None, None, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        ([1, 2, 4], 4, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 4),
        (
            [1, 2, 4],
            8,
            1,
            False,
            2048,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            ValidationError,
        ),
        ([1, 256], None, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        ([], None, 1, False, 2048, CUDAGraphMode.NONE, 0),
        (None, 0, 1, False, 2048, CUDAGraphMode.NONE, 0),
        # truncated to nearest multiple of 8 or 16
        (None, 257, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        # max from list
        ([1, 2, 4, 15], None, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 15),
        # filtered out 15 due to SP
        ([1, 2, 4, 15], None, 2, True, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 4),
        # limited by the max_tokens
        ([1, 2, 4, 15], None, 1, False, 8, CUDAGraphMode.FULL_AND_PIECEWISE, 4),
        # the list should contain at least 1 element when use cudagraph
        ([], None, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, ValidationError),
        # the max capturing size should be >= 1 when use cudagraph
        (None, 0, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, ValidationError),
    ],
)
def test_cudagraph_sizes_post_init(
    cudagraph_capture_sizes,
    max_cudagraph_capture_size,
    tp_size,
    enable_sp,
    max_num_batched_tokens,
    cudagraph_mode,
    expected_max_size,
):
    ctx = nullcontext()
    if expected_max_size == ValidationError:
        ctx = pytest.raises(expected_max_size)

    with (
        ctx,
        patch("vllm.config.parallel.cuda_device_count_stateless", return_value=tp_size),
    ):
        compilation_config = CompilationConfig(
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            max_cudagraph_capture_size=max_cudagraph_capture_size,
            pass_config=PassConfig(
                enable_sp=enable_sp,
                fuse_norm_quant=True,
                fuse_act_quant=True,
                eliminate_noops=True,
            ),
            cudagraph_mode=cudagraph_mode,
        )
        engine_args = EngineArgs(
            model="facebook/opt-125m",
            tensor_parallel_size=tp_size,
            max_num_seqs=min(max_num_batched_tokens, 128),
            max_num_batched_tokens=max_num_batched_tokens,
            compilation_config=compilation_config,
        )
        vllm_config = engine_args.create_engine_config()

        assert (
            vllm_config.compilation_config.max_cudagraph_capture_size
            == expected_max_size
        )


def test_cached_compilation_config(default_vllm_config):
    import torch
    from torch._inductor.utils import run_and_get_code

    from vllm.config import get_cached_compilation_config, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    batch_size, num_qo_heads, head_size = 8, 16, 128

    # access and cache default compilation config
    # default compilation config does not contain +quant_fp8 custom op. If this is
    # used, the generated code would use inductor-generated triton kernel instead
    # of the custom op `torch.ops._C.static_scaled_fp8_quant`.
    get_cached_compilation_config()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+quant_fp8"],
        )
    )

    # set_current_vllm_config should clear cached compilation config and
    # use the new compilation_config in vllm_config
    with set_current_vllm_config(vllm_config):
        query_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
        query_quant = torch.compile(query_quant)

        _q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        query = torch.randn(
            batch_size, num_qo_heads * head_size, dtype=dtype, device=device
        )

        _, code = run_and_get_code(query_quant, query, _q_scale)

    code = " ".join(code)
    assert "torch.ops._C.static_scaled_fp8_quant.default(" in code


def _create_vllm_config_for_validation(
    compilation_config: CompilationConfig,
) -> MagicMock:
    """Helper to create a mock VllmConfig for padding validation testing."""
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.compilation_config = compilation_config
    mock_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=8)
    mock_config.parallel_config = ParallelConfig()
    mock_config.speculative_config = None
    mock_config.lora_config = None
    return mock_config


def test_compile_sizes_padding_validation():
    """Test that compile_sizes with values that would be padded raises an error."""
    # cudagraph_capture_sizes=[1, 2, 4, 8] means:
    # - size 1 -> padded to 1
    # - size 2 -> padded to 2
    # - size 3 -> padded to 4
    # - size 4 -> padded to 4
    # - size 5 -> padded to 8
    # etc.
    # So compile_sizes=[3] should fail because 3 would be padded to 4

    with pytest.raises(ValueError, match="would be padded to"):
        config = CompilationConfig(
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_cudagraph_capture_size=8,
            compile_sizes=[3],
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        config.post_init_cudagraph_sizes()
        dispatcher = CudagraphDispatcher(_create_vllm_config_for_validation(config))
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)

    with pytest.raises(ValueError, match="would be padded to"):
        config = CompilationConfig(
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_cudagraph_capture_size=8,
            compile_sizes=[5],
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        config.post_init_cudagraph_sizes()
        dispatcher = CudagraphDispatcher(_create_vllm_config_for_validation(config))
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)

    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        compile_sizes=[1, 2, 4, 8],
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [1, 2, 4, 8]
    dispatcher = CudagraphDispatcher(_create_vllm_config_for_validation(config))
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)  # Should not raise

    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        compile_sizes=["cudagraph_capture_sizes"],
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [1, 2, 4, 8]

    # When cudagraphs are disabled (max_cudagraph_capture_size=0),
    # padding validation should be skipped
    config = CompilationConfig(
        cudagraph_capture_sizes=[],
        max_cudagraph_capture_size=0,
        compile_sizes=[3, 5, 7],  # would be invalid with cudagraphs
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [3, 5, 7]

    # When cudagraph_mode is NONE but capture_sizes is non-empty,
    # padding validation should still be skipped
    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        cudagraph_mode=CUDAGraphMode.NONE,
        compile_sizes=[3, 5, 7],  # would be invalid if cudagraphs were enabled
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [3, 5, 7]
    dispatcher = CudagraphDispatcher(_create_vllm_config_for_validation(config))
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)  # Should not raise
