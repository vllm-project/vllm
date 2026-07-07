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
def test_stock_uses_decoupled_cudagraph_wrapper(vllm_runner, monkeypatch):
    # The stock path must capture via its own StockTorchCompileCUDAGraphWrapper,
    # never the shared CUDAGraphWrapper (which is coupled to vLLM's
    # non-torch.compile full-cudagraph path). Forked so the wrapper-class
    # instance registries start empty.
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.stock_cudagraph import StockTorchCompileCUDAGraphWrapper
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("use_inductor_graph_partition requires torch>=2.9.0.dev")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(
        "facebook/opt-125m",
        compilation_config={
            "mode": CompilationMode.STOCK_TORCH_COMPILE,
            "use_inductor_graph_partition": True,
            "cudagraph_capture_sizes": [100],
        },
        gpu_memory_utilization=0.4,
    ) as _:
        assert len(StockTorchCompileCUDAGraphWrapper._all_instances) > 0
        assert len(CUDAGraphWrapper._all_instances) == 0


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
    "user_cudagraph_mode,expected_mode,expected_cudagraph",
    [
        # Unset, partition explicitly off -> the opt-level default is a piecewise
        # mode the stock path can't honor without partition, so fall back to the
        # legacy VllmBackend (keeping the piecewise default) rather than silently
        # degrading to a different cudagraph strategy. The partition-on default is
        # covered by test_stock_defaults_to_graph_partition_piecewise.
        (None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE),
        # Explicit NONE is the user's deliberate choice and must be preserved.
        (CUDAGraphMode.NONE, CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.NONE),
        # An explicit piecewise mode needs graph partition (off here); the stock path
        # can't provide it, so fall back to the legacy VllmBackend (which has its own
        # FX splitting) rather than silently dropping cudagraphs.
        (
            CUDAGraphMode.PIECEWISE,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.PIECEWISE,
        ),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        # Explicit full modes need no piecewise capture, so they stay on stock as-is.
        (CUDAGraphMode.FULL, CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.FULL),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
    ],
)
def test_stock_cudagraph_mode_resolution(
    user_cudagraph_mode, expected_mode, expected_cudagraph
):
    # Pin partition off so this isolates the no-partition resolution regardless of
    # the torch version (the default auto-enables partition on torch>=2.9).
    cc_kwargs = {
        "mode": CompilationMode.STOCK_TORCH_COMPILE,
        "use_inductor_graph_partition": False,
    }
    if user_cudagraph_mode is not None:
        cc_kwargs["cudagraph_mode"] = user_cudagraph_mode
    config = VllmConfig(compilation_config=CompilationConfig(**cc_kwargs))
    assert config.compilation_config.mode == expected_mode
    assert config.compilation_config.cudagraph_mode == expected_cudagraph
    # A fallen-back stock config must be indistinguishable from a native
    # VLLM_COMPILE one: ir_enable_torch_wrap follows the FINAL mode, not the
    # transient STOCK mode it started in.
    assert config.compilation_config.ir_enable_torch_wrap is (
        expected_mode == CompilationMode.VLLM_COMPILE
    )


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_late_fallback_resolves_torch_wrap():
    # A piecewise cudagraph_mode re-introduced LATE (here via the dynamic
    # speculative-decoding (SD) override, but also reachable via pooler /
    # KV-connector / platform check_and_update_config) forces a
    # stock->VLLM_COMPILE fallback AFTER the
    # point mode is normally settled. ir_enable_torch_wrap must still resolve to
    # True so custom IR torch-wrap/lowering stays enabled, matching a config that
    # started as VLLM_COMPILE. Regression guard for resolving it too early.
    def _force_piecewise(self):
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    with patch.object(
        VllmConfig, "_maybe_override_dynamic_sd_cudagraph_mode", _force_piecewise
    ):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.STOCK_TORCH_COMPILE,
                use_inductor_graph_partition=False,
            )
        )
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config.compilation_config.ir_enable_torch_wrap is True


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_inert_flag_warning_emitted_after_final_mode():
    # End-to-end ordering guard for the warn-before-fallback fix: the inert-flag
    # warning must be evaluated only after every late mode fallback in __post_init__.
    # We force a late STOCK->VLLM_COMPILE fallback (the dynamic-SD override turns the
    # mode piecewise on a no-partition stock config) and capture the mode the warning
    # method observes; moving the call back up next to where mode is first settled
    # would make it observe STOCK here and re-introduce the false "flags are inert"
    # warning.
    def _force_piecewise(self):
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    observed = {}
    orig = VllmConfig._warn_ignored_vllm_backend_only_flags

    def spy(self):
        observed["mode"] = self.compilation_config.mode
        return orig(self)

    with (
        patch.object(
            VllmConfig, "_maybe_override_dynamic_sd_cudagraph_mode", _force_piecewise
        ),
        patch.object(VllmConfig, "_warn_ignored_vllm_backend_only_flags", spy),
    ):
        VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.STOCK_TORCH_COMPILE,
                use_inductor_graph_partition=False,
                cudagraph_copy_inputs=True,
            )
        )
    # By the time the warning is evaluated the mode has settled to VLLM_COMPILE
    # (which honors cudagraph_copy_inputs), so the mode gate suppresses the false
    # "flags are inert" diagnostic.
    assert observed["mode"] == CompilationMode.VLLM_COMPILE


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_dynamic_sd_reenables_graph_partition():
    # Guards the re-resolve of _resolve_stock_default_cudagraph_mode after the
    # dynamic-SD override: a stock config whose mode only becomes piecewise LATE
    # (dynamic-SD turning an explicit FULL into PIECEWISE) must auto-enable Inductor
    # graph partition and STAY on stock, rather than needlessly falling back to
    # VLLM_COMPILE. Reverting the re-resolve flips both asserts below (mode becomes
    # VLLM_COMPILE, partition stays False). As a downstream consequence the user-set
    # splitting_ops is then the honored partition boundary; that it is not reported
    # inert (and the emission-ordering guarding it) is covered by
    # test_stock_inert_flag_warning_emitted_after_final_mode.
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("use_inductor_graph_partition requires torch>=2.9.0.dev")

    def _force_piecewise(self):
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    with patch.object(
        VllmConfig, "_maybe_override_dynamic_sd_cudagraph_mode", _force_piecewise
    ):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.STOCK_TORCH_COMPILE,
                cudagraph_mode=CUDAGraphMode.FULL,
                splitting_ops=["vllm::unified_attention"],
            )
        )
    assert config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE
    assert config.compilation_config.use_inductor_graph_partition is True


def test_stock_falls_back_to_vllm_compile_for_v2_model_runner():
    # The V2 model runner does not support stock torch.compile. A stock-eligible
    # arch force-run on V2 (VLLM_USE_V2_MODEL_RUNNER=1 -> use_v2_model_runner True)
    # must fall back to VLLM_COMPILE instead of hard-failing _validate_v2_model_runner.
    # Regression guard for the GPT-OSS default-to-stock migration.
    forced = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    with patch.object(VllmConfig, "use_v2_model_runner", property(lambda self: True)):
        forced._maybe_fallback_stock_for_v2_model_runner(user_set_graph_partition=False)
    assert forced.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert forced.compilation_config.use_inductor_graph_partition is False

    # An explicit user use_inductor_graph_partition=True is preserved across the V2
    # fallback: VLLM_COMPILE (which V2 supports) honors graph partition, so the
    # deliberate request must not be silently reset to False.
    explicit = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    explicit.compilation_config.use_inductor_graph_partition = True
    with patch.object(VllmConfig, "use_v2_model_runner", property(lambda self: True)):
        explicit._maybe_fallback_stock_for_v2_model_runner(
            user_set_graph_partition=True
        )
    assert explicit.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert explicit.compilation_config.use_inductor_graph_partition is True

    # V2 not in use -> stock is preserved (the auto path instead keeps stock on the
    # V1 runner via _get_v2_model_runner_unsupported_features).
    keep = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    with patch.object(VllmConfig, "use_v2_model_runner", property(lambda self: False)):
        keep._maybe_fallback_stock_for_v2_model_runner(user_set_graph_partition=False)
    assert keep.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_defaults_to_graph_partition_piecewise():
    # A stock-compile model with nothing set defaults to graph-partition piecewise
    # (FULL_AND_PIECEWISE) on torch>=2.9, auto-enabling use_inductor_graph_partition;
    # on older torch the piecewise default can't be honored, so it falls back to the
    # legacy VllmBackend (preserving the cudagraph mode). The flag is folded into the
    # SupportsStockCompile default rather than being opt-in.
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    cc = config.compilation_config
    if is_torch_equal_or_newer("2.9.0.dev"):
        assert cc.mode == CompilationMode.STOCK_TORCH_COMPILE
        assert cc.use_inductor_graph_partition is True
        assert cc.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    else:
        assert cc.mode == CompilationMode.VLLM_COMPILE
        assert cc.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


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
    # The stock default is graph-partition piecewise only on torch>=2.9; on older
    # torch the piecewise default can't be honored, so it falls back to VllmBackend
    # (the cudagraph mode is preserved across the fallback).
    if (
        expected_mode == CompilationMode.STOCK_TORCH_COMPILE
        and not is_torch_equal_or_newer("2.9.0.dev")
    ):
        expected_mode = CompilationMode.VLLM_COMPILE
    assert config.compilation_config.mode == expected_mode
    assert config.compilation_config.cudagraph_mode == expected_cudagraph


@pytest.mark.parametrize(
    "mode,cudagraph_mode,partition,expected_mode,expected_cudagraph",
    [
        # STOCK + a piecewise mode without graph partition can't be honored, so the
        # engine falls back to the legacy VllmBackend (which has its own FX splitting),
        # leaving cudagraph_mode intact for that backend to consume.
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.PIECEWISE,
            False,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.PIECEWISE,
        ),
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            False,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        # STOCK + piecewise WITH graph partition is supported, so it stays on stock.
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.PIECEWISE,
            True,
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.PIECEWISE,
        ),
        # A non-piecewise mode never needs fixing.
        (
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_DECODE_ONLY,
            False,
            CompilationMode.STOCK_TORCH_COMPILE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        # VLLM_COMPILE honors piecewise itself, so it is left untouched.
        (
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.PIECEWISE,
            False,
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.PIECEWISE,
        ),
        # NONE / DYNAMO_TRACE_ONCE have no compiled graph to capture piecewise, so the
        # piecewise cudagraph_mode drops to NONE (no fallback -- there is nothing to
        # fall back to for these explicit modes).
        (
            CompilationMode.NONE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            False,
            CompilationMode.NONE,
            CUDAGraphMode.NONE,
        ),
        (
            CompilationMode.DYNAMO_TRACE_ONCE,
            CUDAGraphMode.PIECEWISE,
            False,
            CompilationMode.DYNAMO_TRACE_ONCE,
            CUDAGraphMode.NONE,
        ),
    ],
)
def test_fix_unsupported_piecewise_cudagraph_mode(
    mode, cudagraph_mode, partition, expected_mode, expected_cudagraph
):
    # Construct with a benign non-piecewise cudagraph_mode so the config settles in
    # the requested `mode`: an unset (piecewise) default would itself trip the
    # stock->VLLM_COMPILE fallback during __post_init__. Then set the mode under test
    # and invoke the helper in isolation.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=mode,
            use_inductor_graph_partition=partition,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
    )
    config.compilation_config.cudagraph_mode = cudagraph_mode
    config._fix_unsupported_piecewise_cudagraph_mode()
    assert config.compilation_config.mode == expected_mode
    assert config.compilation_config.cudagraph_mode == expected_cudagraph


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
    # partitions there; without it the stock path cannot provide piecewise cudagraphs
    # and falls back to the legacy VllmBackend.
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

    # Partition explicitly off on the stock path can't provide piecewise cudagraphs,
    # so it falls back to the legacy VllmBackend rather than silently degrading to a
    # whole-graph stock path.
    no_partition = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=False,
        )
    )
    assert no_partition.compilation_config.mode == CompilationMode.VLLM_COMPILE


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="stock inductor-options pass wiring is exercised on CUDA",
)
def test_stock_inductor_options_merges_user_post_grad_pass():
    # A user-supplied post_grad_custom_post_pass must be merged into the stock path's
    # PostGradPassManager (mirroring VllmBackend.configure_post_pass), not clobbered by
    # the manager registration. Regression guard: the stock path used to overwrite the
    # pass_key unconditionally, silently dropping the user's post-grad pass.
    from types import SimpleNamespace

    import vllm.compilation.passes.inductor_pass as inductor_pass_mod
    from vllm.compilation.passes.inductor_pass import InductorPass
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    class _MarkerPass(InductorPass):
        def __call__(self, graph):
            pass

    marker = _MarkerPass()
    # Pin the whole-graph stock path (partition off, decode-only cudagraphs) with a real
    # fusion pass active (enable_qk_norm_rope_fusion) so _stock_inductor_options
    # registers the pass manager and reaches the merge branch instead of returning the
    # base options early.
    vc = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=False,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            pass_config=PassConfig(
                eliminate_noops=True, enable_qk_norm_rope_fusion=True
            ),
            inductor_compile_config={current_platform.pass_key: marker},
        )
    )
    vc.model_config = MagicMock(dtype=torch.bfloat16)
    # max_num_tokens is the upper bound of the pass-context Range the stock options
    # install; mirror the runner's self.max_num_tokens (scheduler
    # max_num_batched_tokens).
    stub = SimpleNamespace(
        compilation_config=vc.compilation_config,
        vllm_config=vc,
        max_num_tokens=vc.scheduler_config.max_num_batched_tokens,
    )

    # set_pass_context installs process-global state with no teardown by design;
    # restore it so this unit test leaves no global leak for other tests.
    prev_pass_context = inductor_pass_mod._pass_context
    try:
        opts, has_fusion = GPUModelRunner._stock_inductor_options(stub)
        # The installed pass context must be the finite [0, max_num_tokens] range,
        # not an unbounded end: end-bounded fusions (allreduce+rmsnorm, rope+kvcache)
        # gate on compile_range.end <= max_token_num, so an unbounded end would
        # silently drop every one of them. Regression guard for that range fix.
        installed_range = inductor_pass_mod._pass_context.compile_range
        assert installed_range.start == 0
        assert installed_range.end == vc.scheduler_config.max_num_batched_tokens
    finally:
        inductor_pass_mod._pass_context = prev_pass_context

    assert has_fusion
    manager = opts[current_platform.pass_key]
    # The manager replaces the raw pass under pass_key, and the user's pass is merged
    # into the manager's pass list rather than dropped.
    assert manager is not marker
    assert marker in manager.passes


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
    assert "cudagraph_copy_inputs" in " ".join(str(a) for a in mock_warn.call_args.args)

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


def test_stock_inert_flag_warning_suppressed_after_fallback():
    # Regression guard for the warn-before-fallback ordering bug: the warning is
    # gated on the FINAL mode. A config that LATE-falls-back to VLLM_COMPILE (dynamic
    # SD / pooler / KV connector / platform check_and_update_config re-introducing a
    # piecewise mode) must NOT be warned that its VllmBackend-only flags are inert --
    # VLLM_COMPILE honors them.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE, cudagraph_copy_inputs=True
        )
    )
    config.model_config = MagicMock(architectures=["GptOssForCausalLM"])
    # Control: while still on the stock path the flag DOES warn.
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        config._warn_ignored_vllm_backend_only_flags()
    assert mock_warn.call_count == 1
    # Simulate the late fallback to the legacy backend: the same flag is now honored.
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        config._warn_ignored_vllm_backend_only_flags()
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
    # A swallowed resolution error must still return the safe default AND be
    # surfaced at a visible level so a stock-eligible arch silently falling
    # back to VllmBackend is diagnosable.
    with patch("vllm.config.vllm.logger.warning_once") as mock_warn:
        assert raising._stock_compile_supported() is False
    assert mock_warn.call_count == 1
    warn_text = " ".join(str(a) for a in mock_warn.call_args.args)
    assert "GptOssForCausalLM" in warn_text
    assert "boom" in warn_text


@pytest.mark.parametrize("decorate_outer", [True, False])
def test_stock_support_torch_compile_routes_to_runner_compile(decorate_outer):
    # Spec-decode drafters run under the engine-global STOCK mode with their
    # @support_torch_compile decorator not using vLLM's custom backend
    # (do_not_use_custom_torch_compile_backend), and the runner
    # compiles the drafter model via nn.Module.compile().
    # - decorate_outer=True (MTP shape): the compiled module IS the decorated class.
    #   Because the decorator overrides nn.Module.__call__, .compile() only takes
    #   effect if that __call__ routes to _compiled_call_impl -- with the pre-fix
    #   decorator the backend is never invoked (calls == 0). This case guards the fix.
    # - decorate_outer=False (eagle-head shape): an undecorated outer wraps the
    #   decorated inner; the outer's .compile() works regardless of the fix. This is a
    #   regression guard that the eagle shape still compiles under stock.
    import torch
    import torch.nn as nn

    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import set_current_vllm_config

    @support_torch_compile
    class Decorated(nn.Module):
        def __init__(self, *, vllm_config=None, prefix=""):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor):
            return self.lin(x)

    cfg = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            use_inductor_graph_partition=False,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
    )
    with set_current_vllm_config(cfg):
        decorated = Decorated(vllm_config=cfg)

    # decorate_outer=True compiles the decorated module directly (the MTP-drafter
    # shape); False wraps it in an undecorated outer (the eagle-head shape).
    module = decorated if decorate_outer else nn.Sequential(decorated)

    calls = {"n": 0}

    def counting_backend(gm, example_inputs):
        calls["n"] += 1
        return gm.forward

    module.compile(fullgraph=True, backend=counting_backend)
    module(torch.randn(4, 8))
    assert calls["n"] >= 1, (
        "nn.Module.compile() did not take effect under STOCK_TORCH_COMPILE; the "
        "drafter would run eager"
    )


def test_support_torch_compile_none_mode_runs_eager():
    # The __call__ routing that sends do_not_use_custom_torch_compile_backend modules
    # to _compiled_call_impl is STOCK-only: under CompilationMode.NONE the module has
    # do_not_use_custom_torch_compile_backend set and nothing calls nn.Module.compile()
    # on it, so _compiled_call_impl stays None and it runs eager (guards the shared
    # decorator path against a regression).
    import torch
    import torch.nn as nn

    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import set_current_vllm_config

    @support_torch_compile
    class Decorated(nn.Module):
        def __init__(self, *, vllm_config=None, prefix=""):
            super().__init__()
            self.ran_eager = False

        def forward(self, x: torch.Tensor):
            self.ran_eager = True
            return x

    cfg = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(cfg):
        m = Decorated(vllm_config=cfg)
    assert getattr(m, "_compiled_call_impl", None) is None
    m(torch.randn(2, 2))
    assert m.ran_eager


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="glue-kernel resolution comparison is CUDA-specific",
)
def test_stock_uses_default_glue_kernels_like_vllm_backend():
    # Stock must NOT force the vLLM custom RMSNorm/RoPE glue kernels. A TP1-TP8 sweep
    # found the custom glue (+rotary_embedding, ir_op_priority.rms_norm=vllm_c) is ~10%
    # slower at batch-1 decode than the Inductor glue VllmBackend actually uses
    # (custom_ops=none), with no throughput or prefill benefit (PR #46423). Stock must
    # resolve to the same glue as VllmBackend so they run identical kernels.
    stock = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    vllm = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    # No forced custom RoPE kernel, and RMSNorm priority is not forced to vllm_c.
    assert "+rotary_embedding" not in stock.compilation_config.custom_ops
    assert stock.kernel_config.ir_op_priority.rms_norm[0] != "vllm_c"
    assert stock.kernel_config.ir_op_priority.fused_add_rms_norm[0] != "vllm_c"
    # Stock resolves to the same glue kernels as VllmBackend.
    assert (
        stock.kernel_config.ir_op_priority.rms_norm
        == vllm.kernel_config.ir_op_priority.rms_norm
    )
    assert (
        stock.kernel_config.ir_op_priority.fused_add_rms_norm
        == vllm.kernel_config.ir_op_priority.fused_add_rms_norm
    )


@pytest.mark.parametrize(
    "method,expected_mode",
    [
        # Validated drafter methods keep the stock path (runner compiles the drafter).
        ("eagle", CompilationMode.STOCK_TORCH_COMPILE),
        ("eagle3", CompilationMode.STOCK_TORCH_COMPILE),
        # Algorithmic (model-less) drafters have nothing to compile, so stay stock.
        ("ngram", CompilationMode.STOCK_TORCH_COMPILE),
        ("ngram_gpu", CompilationMode.STOCK_TORCH_COMPILE),
        ("suffix", CompilationMode.STOCK_TORCH_COMPILE),
        # Unvalidated model-backed drafters fall back so they compile as before
        # (fullgraph-compiling them on stock is unchecked / would graph-break). mtp is
        # not yet e2e-validated on stock, so it falls back too.
        ("mtp", CompilationMode.VLLM_COMPILE),
        ("draft_model", CompilationMode.VLLM_COMPILE),
        ("medusa", CompilationMode.VLLM_COMPILE),
        ("dflash", CompilationMode.VLLM_COMPILE),
        ("extract_hidden_states", CompilationMode.VLLM_COMPILE),
    ],
)
def test_stock_drafter_method_fallback(method, expected_mode):
    from types import SimpleNamespace

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    config.speculative_config = SimpleNamespace(method=method)
    config._maybe_fallback_stock_for_unsupported_drafter()
    assert config.compilation_config.mode == expected_mode


def test_stock_drafter_fallback_noop_cases():
    from types import SimpleNamespace

    # No speculative config -> stock preserved.
    cfg = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)
    )
    cfg.speculative_config = None
    cfg._maybe_fallback_stock_for_unsupported_drafter()
    assert cfg.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE

    # Non-stock mode is left untouched even with an unvalidated model-backed drafter.
    cfg = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    cfg.speculative_config = SimpleNamespace(method="medusa")
    cfg._maybe_fallback_stock_for_unsupported_drafter()
    assert cfg.compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_algorithmic_drafter_methods_stay_stock():
    from types import SimpleNamespace

    # Iterate the shared constant (not a hardcoded list) so the single source of
    # truth used by _maybe_fallback_stock_for_unsupported_drafter is what gets
    # exercised: any model-less method added to _ALGORITHMIC_DRAFTER_METHODS is
    # covered automatically, and the disjointness check guards against a method
    # being listed as both algorithmic and validated-model-backed.
    assert VllmConfig._ALGORITHMIC_DRAFTER_METHODS
    assert not (
        set(VllmConfig._ALGORITHMIC_DRAFTER_METHODS)
        & set(VllmConfig._STOCK_SUPPORTED_DRAFTER_METHODS)
    )
    for method in VllmConfig._ALGORITHMIC_DRAFTER_METHODS:
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.STOCK_TORCH_COMPILE
            )
        )
        config.speculative_config = SimpleNamespace(method=method)
        config._maybe_fallback_stock_for_unsupported_drafter()
        assert config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE, (
            method
        )


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if cudagraph mode not supported",
)
def test_stock_torch_compile_compiles_spec_decode_drafter(vllm_runner, monkeypatch):
    # A model-backed spec-decode drafter must be compiled on the stock path (no
    # fallback to VLLM_COMPILE): the runner compiles both the target and the eagle
    # drafter model via stock torch.compile, so stock_torch_compile_count == 2. The
    # drafter's own @support_torch_compile decorator is disabled under STOCK, and its
    # piecewise cudagraphs come from the same Inductor graph partition as the target.
    # Llama-3.2-1B is not SupportsStockCompile, so mode=STOCK is forced here.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with (
        compilation_counter.expect(stock_torch_compile_count=2),
        vllm_runner(
            "meta-llama/Llama-3.2-1B-Instruct",
            speculative_config={
                "method": "eagle3",
                "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
                "num_speculative_tokens": 3,
            },
            compilation_config={"mode": CompilationMode.STOCK_TORCH_COMPILE},
            max_model_len=2048,
            gpu_memory_utilization=0.55,
        ) as _,
    ):
        pass


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
