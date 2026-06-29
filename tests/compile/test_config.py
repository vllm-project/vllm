# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
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

DEVICE_TYPE = current_platform.device_type


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
def test_stock_torch_compile_piecewise_graph_partition(vllm_runner, monkeypatch):
    # Stock + use_inductor_graph_partition recovers piecewise cudagraphs (step 2 of
    # the VllmBackend migration): Inductor partitions at the attention ops and the
    # external wrapper captures each partition (FULL_AND_PIECEWISE). opt-125m at one
    # capture size -> 13 piecewise + 1 full decode = 14 captured graphs (the same
    # count as the VllmBackend FaP path), proving the stock piecewise path actually
    # captures rather than running attention inside the cudagraph.
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("use_inductor_graph_partition requires torch>=2.9.0.dev")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with (
        compilation_counter.expect(
            stock_torch_compile_count=1,
            num_gpu_runner_capture_triggers=1,
            num_cudagraph_captured=14,
        ),
        vllm_runner(
            "facebook/opt-125m",
            compilation_config={
                "mode": CompilationMode.STOCK_TORCH_COMPILE,
                "use_inductor_graph_partition": True,
                "cudagraph_capture_sizes": [100],
            },
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


@pytest.mark.forked
def test_torch_compile_disable(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    with (
        compilation_counter.expect(num_graphs_seen=0, stock_torch_compile_count=0),
        vllm_runner(
            "facebook/opt-125m",
            gpu_memory_utilization=0.4,
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
            splitting_ops=["vllm::unified_attention_with_output"],
        )
    )
    # with inductor partition we use splitting_ops directly for
    # partition rules
    assert config.compilation_config.splitting_ops == [
        "vllm::unified_attention_with_output"
    ]

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
                "vllm::unified_attention_with_output",
                "vllm::moe_forward",
                "vllm::moe_forward_shared",
            ],
        ),
    )
    splitting_ops = config.compilation_config.splitting_ops
    assert splitting_ops == [
        "vllm::unified_attention_with_output",
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
        # max_num_batched_tokens <= max_cudagraph_capture_size should always be
        # captured even if not landing on a 16-stride step
        (None, 2048, 1, False, 257, CUDAGraphMode.FULL_AND_PIECEWISE, 257),
        # max from list
        ([1, 2, 4, 15], None, 1, False, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 15),
        # SP forces full-graph compilation, sizes are filtered by TP
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
        patch.object(current_platform, "device_count", return_value=tp_size),
    ):
        kwargs = {}
        if cudagraph_capture_sizes is not None:
            kwargs["cudagraph_capture_sizes"] = cudagraph_capture_sizes
        if max_cudagraph_capture_size is not None:
            kwargs["max_cudagraph_capture_size"] = max_cudagraph_capture_size
        compilation_config = CompilationConfig(
            pass_config=PassConfig(
                enable_sp=enable_sp,
                fuse_norm_quant=True,
                fuse_act_quant=True,
                eliminate_noops=True,
                sp_min_token_num=512 if enable_sp else None,
            ),
            cudagraph_mode=cudagraph_mode,
            **kwargs,
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


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if not cudagraph mode supported",
)
@pytest.mark.parametrize(
    (
        "cudagraph_mode",
        "use_inductor_graph_partition",
        "expected_enable_sp",
        "expected_cudagraph_mode",
        "expected_piecewise_compile",
        "expected_capture_sizes",
        "expected_max_size",
    ),
    [
        (CUDAGraphMode.PIECEWISE, False, True, CUDAGraphMode.FULL, False, [2, 4], 4),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            False,
            True,
            CUDAGraphMode.FULL_DECODE_ONLY,
            False,
            [2, 4],
            4,
        ),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            False,
            True,
            CUDAGraphMode.FULL,
            False,
            [2, 4],
            4,
        ),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            True,
            True,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            True,
            [2, 4],
            4,
        ),
    ],
)
def test_sequence_parallelism_requires_full_graph_compilation(
    cudagraph_mode: CUDAGraphMode,
    use_inductor_graph_partition: bool,
    expected_enable_sp: bool,
    expected_cudagraph_mode: CUDAGraphMode,
    expected_piecewise_compile: bool,
    expected_capture_sizes: list[int],
    expected_max_size: int,
):
    with patch.object(current_platform, "device_count", return_value=2):
        vllm_config = VllmConfig(
            parallel_config=ParallelConfig(tensor_parallel_size=2),
            scheduler_config=SchedulerConfig(
                max_num_seqs=128,
                max_num_batched_tokens=2048,
                max_model_len=2048,
                is_encoder_decoder=False,
            ),
        )
        vllm_config.model_config = MagicMock(
            dtype=torch.float16,
            enforce_eager=False,
            is_moe=False,
            disable_cascade_attn=False,
            get_hidden_size=MagicMock(return_value=4096),
        )
        vllm_config.compilation_config = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_capture_sizes=[1, 2, 4, 15],
            max_cudagraph_capture_size=None,
            compile_sizes=["cudagraph_capture_sizes"],
            use_inductor_graph_partition=use_inductor_graph_partition,
            pass_config=PassConfig(
                enable_sp=True,
                fuse_gemm_comms=True,
                fuse_norm_quant=True,
                fuse_act_quant=True,
                eliminate_noops=True,
                sp_min_token_num=512,
            ),
            cudagraph_mode=cudagraph_mode,
        )
        vllm_config.compilation_config.set_splitting_ops_for_v1(
            all2all_backend=vllm_config.parallel_config.all2all_backend,
            data_parallel_size=1,
        )
        vllm_config._set_compile_ranges()
        vllm_config._set_cudagraph_sizes()

    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        == use_inductor_graph_partition
    )
    assert (
        bool(vllm_config.compilation_config.splitting_ops) == expected_piecewise_compile
    )
    assert vllm_config.compilation_config.pass_config.enable_sp == expected_enable_sp
    assert (
        vllm_config.compilation_config.pass_config.fuse_gemm_comms == expected_enable_sp
    )
    assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode
    assert (
        vllm_config.compilation_config.cudagraph_capture_sizes == expected_capture_sizes
    )
    assert (
        vllm_config.compilation_config.max_cudagraph_capture_size == expected_max_size
    )
    assert (
        511 in vllm_config.compilation_config.compile_ranges_endpoints
    ) == expected_enable_sp


def test_cached_compilation_config(default_vllm_config):
    import torch
    from torch._inductor.utils import run_and_get_code

    from vllm.config import get_cached_compilation_config, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    dtype = torch.bfloat16
    device = torch.device(f"{DEVICE_TYPE}:0")
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

        _q_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE_TYPE)
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


def test_inductor_asserts_default_disabled(monkeypatch):
    """Test that inductor runtime asserts are disabled by default
    (INFO logging level) on torch < 2.12."""
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")

    import importlib

    import vllm.envs

    importlib.reload(vllm.envs)

    config = CompilationConfig()
    if not _is_torch_equal_or_newer(torch.__version__, "2.12.0.dev"):
        assert config.inductor_compile_config.get("size_asserts") is False
        assert config.inductor_compile_config.get("alignment_asserts") is False
        assert config.inductor_compile_config.get("scalar_asserts") is False


def test_inductor_asserts_enabled_in_debug(monkeypatch):
    """Test that VLLM_LOGGING_LEVEL=DEBUG enables inductor runtime asserts
    on torch < 2.12."""
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")

    import importlib

    import vllm.envs

    importlib.reload(vllm.envs)

    config = CompilationConfig()
    if not _is_torch_equal_or_newer(torch.__version__, "2.12.0.dev"):
        assert config.inductor_compile_config.get("size_asserts") is True
        assert config.inductor_compile_config.get("alignment_asserts") is True
        assert config.inductor_compile_config.get("scalar_asserts") is True


def test_get_inductor_factors_includes_configs():
    """Changing inductor or functorch config must change the cache key factors."""
    from torch._functorch import config as functorch_config
    from torch._inductor import config as inductor_config

    from vllm.compilation.compiler_interface import get_inductor_factors

    baseline = get_inductor_factors()

    with inductor_config.patch("max_autotune", not inductor_config.max_autotune):
        patched = get_inductor_factors()
    assert baseline != patched, "inductor config change was not reflected"

    with functorch_config.patch("donated_buffer", not functorch_config.donated_buffer):
        patched = get_inductor_factors()
    assert baseline != patched, "functorch config change was not reflected"


def test_inductor_asserts_user_override(monkeypatch):
    """Test that explicit inductor_compile_config overrides the
    debug-logging default."""
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")

    import importlib

    import vllm.envs

    importlib.reload(vllm.envs)

    config = CompilationConfig(
        inductor_compile_config={"size_asserts": True},
    )
    assert config.inductor_compile_config.get("size_asserts") is True
    if not _is_torch_equal_or_newer(torch.__version__, "2.12.0.dev"):
        assert config.inductor_compile_config.get("alignment_asserts") is False


# --- STOCK_TORCH_COMPILE migration (SupportsStockCompile) config resolution ---
# These build VllmConfig directly (no model load / GPU memory) and assert the
# resolved compile mode / cudagraph mode. The cudagraph-mode assertions need a
# platform that supports static graph mode (FDO is otherwise forced to NONE).


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
@pytest.mark.parametrize(
    "user_cudagraph_mode,expected",
    [
        # Unset, partition explicitly off -> FULL_DECODE_ONLY (whole-graph stock,
        # decode-only external cudagraph). The partition-on default is covered by
        # test_stock_defaults_to_graph_partition_piecewise.
        (None, CUDAGraphMode.FULL_DECODE_ONLY),
        # Explicit NONE is the user's deliberate choice and must be preserved.
        (CUDAGraphMode.NONE, CUDAGraphMode.NONE),
        # Piecewise needs graph partition (off here); the stock path downgrades it to
        # NONE (not silently re-promoted to FULL_DECODE_ONLY).
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE),
        (CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.NONE),
        # Explicit full modes are honored as-is.
        (CUDAGraphMode.FULL, CUDAGraphMode.FULL),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_stock_cudagraph_mode_resolution(user_cudagraph_mode, expected):
    # Pin partition off so this isolates the no-partition resolution regardless of
    # the torch version (the default auto-enables partition on torch>=2.9).
    cc_kwargs = {
        "mode": CompilationMode.STOCK_TORCH_COMPILE,
        "use_inductor_graph_partition": False,
    }
    if user_cudagraph_mode is not None:
        cc_kwargs["cudagraph_mode"] = user_cudagraph_mode
    config = VllmConfig(compilation_config=CompilationConfig(**cc_kwargs))
    assert config.compilation_config.cudagraph_mode == expected


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_defaults_to_graph_partition_piecewise():
    # A stock-compile model with nothing set defaults to graph-partition piecewise
    # (FULL_AND_PIECEWISE) on torch>=2.9, auto-enabling use_inductor_graph_partition;
    # on older torch it falls back to FULL_DECODE_ONLY. The flag is folded into the
    # SupportsStockCompile default rather than being opt-in.
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    cc = config.compilation_config
    if is_torch_equal_or_newer("2.9.0.dev"):
        assert cc.use_inductor_graph_partition is True
        assert cc.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    else:
        assert cc.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
@pytest.mark.parametrize(
    "supported,user_mode,expected_mode,expected_cudagraph",
    [
        (
            True,
            None,
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        (
            False,
            None,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        # An explicit mode always wins over the allowlist auto-selection.
        (
            True,
            CompilationMode.VLLM_COMPILE,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
    ],
)
def test_stock_mode_auto_selection(
    supported, user_mode, expected_mode, expected_cudagraph
):
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    cc_kwargs = {} if user_mode is None else {"mode": user_mode}
    with patch.object(VllmConfig, "_stock_compile_supported", lambda self: supported):
        config = VllmConfig(compilation_config=CompilationConfig(**cc_kwargs))
    assert config.compilation_config.mode == expected_mode
    # The stock default is graph-partition piecewise only on torch>=2.9; otherwise
    # it falls back to the decode-only external cudagraph.
    if (
        expected_mode == CompilationMode.STOCK_TORCH_COMPILE
        and not is_torch_equal_or_newer("2.9.0.dev")
    ):
        expected_cudagraph = CUDAGraphMode.FULL_DECODE_ONLY
    assert config.compilation_config.cudagraph_mode == expected_cudagraph


@pytest.mark.parametrize(
    "mode,cudagraph_mode,expected",
    [
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.NONE,
        ),
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.NONE,
        ),
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        # VLLM_COMPILE keeps piecewise (it has the FX splitting to honor it).
        (
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.PIECEWISE,
        ),
    ],
)
def test_downgrade_piecewise_cudagraph_helper(mode, cudagraph_mode, expected):
    # Pin partition off so the STOCK rows test the clamp (the default auto-enables
    # partition on torch>=2.9, which is the exempt-from-clamp path).
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=mode, use_inductor_graph_partition=False
        )
    )
    config.compilation_config.cudagraph_mode = cudagraph_mode
    config._downgrade_piecewise_cudagraph_for_non_vllm_compile()
    assert config.compilation_config.cudagraph_mode == expected


# --- step 2: piecewise cudagraphs on the stock path via Inductor graph partition ---


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
@pytest.mark.parametrize(
    "user_cudagraph_mode,expected",
    [
        # Unset + graph partition -> prefill can be piecewise-captured, so default to
        # FULL_AND_PIECEWISE rather than FULL_DECODE_ONLY.
        (None, CUDAGraphMode.FULL_AND_PIECEWISE),
        # Explicit piecewise modes are now honored on the stock path (Inductor graph
        # partition provides them) instead of being clamped to NONE.
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE),
        (CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE),
        # Explicit NONE is still the user's deliberate choice.
        (CUDAGraphMode.NONE, CUDAGraphMode.NONE),
    ],
)
def test_stock_graph_partition_cudagraph_mode(user_cudagraph_mode, expected):
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("use_inductor_graph_partition requires torch>=2.9.0.dev")
    cc_kwargs = {
        "mode": CompilationMode.STOCK_TORCH_COMPILE,
        "use_inductor_graph_partition": True,
    }
    if user_cudagraph_mode is not None:
        cc_kwargs["cudagraph_mode"] = user_cudagraph_mode
    config = VllmConfig(compilation_config=CompilationConfig(**cc_kwargs))
    assert config.compilation_config.cudagraph_mode == expected


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_graph_partition_populates_splitting_ops():
    # With graph partition, splitting_ops must name the attention ops so Inductor
    # partitions there; without it the stock path is whole-graph (empty splitting_ops).
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("use_inductor_graph_partition requires torch>=2.9.0.dev")

    partition = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=True,
        )
    )
    assert partition.compilation_config.splitting_ops == list(
        CompilationConfig._attention_ops
    )

    # An explicit empty list with partition on must still be populated (else
    # FULL_AND_PIECEWISE would have zero piecewise partitions, silently capturing
    # attention inside the cudagraph).
    explicit_empty = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=True,
            splitting_ops=[],
        )
    )
    assert explicit_empty.compilation_config.splitting_ops == list(
        CompilationConfig._attention_ops
    )

    # Partition explicitly off -> whole-graph stock, no splitting ops (the default
    # now auto-enables partition, so this must pin it off to test the FDO path).
    no_partition = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=False,
        )
    )
    assert no_partition.compilation_config.splitting_ops == []


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="enable_qk_norm_rope_fusion is forced off on non-CUDA platforms",
)
def test_stock_inductor_options_fusion_gate():
    # The fusion gate is derived from the configured PostGradPassManager rather
    # than a hand-maintained flag list, so it cannot drift: dense bf16 (only the
    # always-on noop pass) registers nothing, while a fusion the old list omitted
    # (enable_qk_norm_rope_fusion) still registers the pass manager.
    from types import SimpleNamespace

    import vllm.compilation.passes.inductor_pass as inductor_pass_mod
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def options_for(**pass_flags):
        vc = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.STOCK_TORCH_COMPILE,
                pass_config=PassConfig(**pass_flags),
            )
        )
        vc.model_config = MagicMock(dtype=torch.bfloat16)
        stub = SimpleNamespace(compilation_config=vc.compilation_config, vllm_config=vc)
        return GPUModelRunner._stock_inductor_options(stub)

    assert options_for(eliminate_noops=True) is None

    # The fusion-positive dict must actually wire the pre/post-grad passes through
    # Inductor's custom-pass hooks, not merely be non-None.
    from vllm.compilation.passes.ir.inplace_functionalization import (
        VllmIRInplaceFunctionalizationPass,
    )

    # The fusion path installs a process-global pass context (set_pass_context has
    # no teardown by design); restore it so this unit test leaves no global state.
    prev_pass_context = inductor_pass_mod._pass_context
    try:
        opts = options_for(eliminate_noops=True, enable_qk_norm_rope_fusion=True)
        assert opts is not None
        assert isinstance(
            opts["pre_grad_custom_pass"], VllmIRInplaceFunctionalizationPass
        )
        assert current_platform.pass_key in opts
        assert "pre_grad_custom_pass" in opts["_cache_config_ignore_prefix"]
    finally:
        inductor_pass_mod._pass_context = prev_pass_context


def test_stock_inert_flag_warning():
    # VllmBackend-only flags on a stock model emit one warning that names the
    # working escape hatch and never the broken -O.mode=3 form.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE, compile_sizes=[1, 2]
        )
    )
    config.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        config._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 1
    fmt, *fmt_args = mock_warn.call_args.args
    assert "--compilation-config" in fmt
    assert "-O.mode=3" not in fmt
    assert "compile_sizes" in " ".join(str(a) for a in fmt_args)

    # cudagraph_copy_inputs and splitting_ops (without graph partition) are also
    # VllmBackend-only on the stock path and must each surface in the warning args.
    copy_cfg = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE, cudagraph_copy_inputs=True
        )
    )
    copy_cfg.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        copy_cfg._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 1
    assert "cudagraph_copy_inputs" in " ".join(
        str(a) for a in mock_warn.call_args.args
    )

    split_cfg = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            splitting_ops=["vllm::unified_attention"],
            use_inductor_graph_partition=False,
        )
    )
    split_cfg.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        split_cfg._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 1
    assert "splitting_ops" in " ".join(str(a) for a in mock_warn.call_args.args)

    # VLLM_COMPILE honors these flags, so no warning.
    other = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, compile_sizes=[1, 2]
        )
    )
    other.model_config = MagicMock(architectures=["LlamaForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        other._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 0


def test_stock_graph_partition_no_piecewise_override_warning():
    # With use_inductor_graph_partition the stock path provides piecewise cudagraphs,
    # so resolving to FULL_AND_PIECEWISE must NOT warn about an unsupported/overridden
    # cudagraph_mode (regression: the inert-flag warning used to fire here).
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("use_inductor_graph_partition requires torch>=2.9.0.dev")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=True,
        )
    )
    config.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        config._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 0


def test_stock_compile_cache_flags_warn(monkeypatch):
    # vLLM's own compile cache is inert on the stock path (torch.compile uses its
    # own cache); setting any of its knobs must warn, not be silently ignored.
    monkeypatch.setenv("VLLM_COMPILE_CACHE_SAVE_FORMAT", "binary")

    def warn_parts(**cc):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.STOCK_TORCH_COMPILE, **cc
            )
        )
        config.model_config = MagicMock(architectures=["GptOssForCausalLM"])
        with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
            config._warn_ignored_vllm_backend_only_flags()
        assert mock_warn.call_count == 1
        return " ".join(str(a) for a in mock_warn.call_args.args)

    assert "cache_dir" in warn_parts(cache_dir="/tmp/vllm-cache")
    assert "compile_cache_save_format" in warn_parts(
        compile_cache_save_format="unpacked"
    )

    # VLLM_DISABLE_COMPILE_CACHE is an env var, not a field.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    config.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        config._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 1
    assert "VLLM_DISABLE_COMPILE_CACHE" in " ".join(
        str(a) for a in mock_warn.call_args.args
    )


def test_stock_warn_no_false_positive_on_autopopulated_fields():
    # compile_ranges_endpoints is auto-populated later in __post_init__ (and
    # __post_init__ may re-run on a reused compilation_config), so a resolved
    # non-empty value must NOT trigger a warning when the user set no inert flags.
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    # simulate the resolved/auto-populated state seen on a real run / re-run
    config.compilation_config.compile_ranges_endpoints = [16384]
    config.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        config._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 0


def test_stock_warning_handles_missing_model_config():
    # mode==STOCK with no model_config must not raise (the warn helper dereferences
    # model_config.architectures only after a None guard).
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE, compile_sizes=[1, 2]
        )
    )
    config.model_config = None
    config._warn_ignored_vllm_backend_only_flags()


def test_stock_compile_supported_safe_defaults():
    # The real _stock_compile_supported (not patched out) must return False for
    # both the no-model case and any registry resolution error, so a broken/unknown
    # model never gets auto-selected onto the stock path.
    no_model = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    no_model.model_config = None
    assert no_model._stock_compile_supported() is False

    raising = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    registry = MagicMock()
    registry.is_stock_compile_supported_model.side_effect = RuntimeError("boom")
    raising.model_config = MagicMock(
        architectures=["GptOssForCausalLM"], registry=registry
    )
    assert raising._stock_compile_supported() is False


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_dynamic_sd_keeps_cudagraph_none():
    # The auto-promotion from NONE to FULL_DECODE_ONLY is guarded by uses_dynamic_sd:
    # dynamic spec decode varies the verification length, so external full-decode
    # capture must NOT kick in. An algorithmic (ngram_gpu) drafter keeps the stock
    # path, so this isolates the dynamic-SD guard on STOCK.
    from types import SimpleNamespace

    spec = SimpleNamespace(
        method="ngram_gpu",
        disable_padded_drafter_batch=False,
        num_speculative_tokens=3,
        max_num_new_slots_for_drafting=3,
        parallel_drafting=False,
        uses_dynamic_speculative_decoding=lambda: True,
        use_eagle=lambda: False,
        compute_hash=lambda: "spec",
    )
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    # Assign the stub after construction (pydantic's dataclass validator rejects a
    # SimpleNamespace at construction time) and re-run resolution to exercise the
    # dynamic-SD guard in __post_init__.
    config.speculative_config = spec
    config.compilation_config.cudagraph_mode = None
    # Reset the auto-resolved partition flag too: the spec-less first construction
    # auto-enabled it, and a fresh resolution (the real single-construction path with
    # a dynamic drafter) would not, since the dynamic-SD guard skips that block.
    config.compilation_config.use_inductor_graph_partition = None
    config.__post_init__()
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    "method",
    ["eagle", "eagle3", "mtp", "dflash", "draft_model", "medusa"],
)
def test_stock_falls_back_to_vllm_compile_for_model_backed_drafter(method):
    # A model-backed spec-decode drafter is compiled by its own decorator under
    # VLLM_COMPILE but never on the stock path, so a stock target with such a
    # drafter must fall back to VLLM_COMPILE (overriding even an explicit STOCK).
    from types import SimpleNamespace

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    config.speculative_config = SimpleNamespace(method=method)
    config._maybe_fallback_stock_to_vllm_compile_for_drafter()
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu", "suffix"])
def test_stock_preserved_for_algorithmic_drafter(method):
    # Algorithmic drafters have no model to compile, so the stock path is kept.
    from types import SimpleNamespace

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    config.speculative_config = SimpleNamespace(method=method)
    config._maybe_fallback_stock_to_vllm_compile_for_drafter()
    assert config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE


def test_stock_drafter_fallback_noop_cases():
    from types import SimpleNamespace

    # No speculative config -> stock preserved (the validated no-spec sweep).
    cfg = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    cfg.speculative_config = None
    cfg._maybe_fallback_stock_to_vllm_compile_for_drafter()
    assert cfg.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE

    # Non-stock mode is left untouched even with a model-backed drafter.
    cfg = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    cfg.speculative_config = SimpleNamespace(method="eagle")
    cfg._maybe_fallback_stock_to_vllm_compile_for_drafter()
    assert cfg.compilation_config.mode == CompilationMode.VLLM_COMPILE


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_torch_compile_captures_full_cudagraph(vllm_runner, monkeypatch):
    # The headline parity path: a stock-compiled model must still attach vLLM's
    # external FULL cudagraph wrapper and actually capture a FULL_DECODE_ONLY graph.
    # opt-125m is not SupportsStockCompile, so mode=STOCK is forced explicitly here;
    # the auto-FDO promotion is allowlist-independent, so this isolates the cudagraph
    # fall-through (not the SupportsStockCompile selection).
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with (
        compilation_counter.expect(
            stock_torch_compile_count=1,
            num_gpu_runner_capture_triggers=1,
            num_cudagraph_captured=1,
        ),
        vllm_runner(
            "facebook/opt-125m",
            compilation_config={
                "mode": CompilationMode.STOCK_TORCH_COMPILE,
                # Pin partition off: this test isolates the FULL_DECODE_ONLY fall-
                # through, while the default now auto-enables graph-partition
                # piecewise (covered by test_stock_torch_compile_piecewise_*).
                "use_inductor_graph_partition": False,
                "cudagraph_capture_sizes": [100],
            },
            gpu_memory_utilization=0.4,
        ) as _,
    ):
        pass
