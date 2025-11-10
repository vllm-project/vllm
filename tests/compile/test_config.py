# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from contextlib import nullcontext

import pytest

from vllm.compilation.counter import compilation_counter
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _is_torch_equal_or_newer, is_torch_equal_or_newer


def test_version():
    # Test the version comparison logic using the private function
    assert _is_torch_equal_or_newer("2.8.0.dev20250624+cu128", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0a0+gitc82a174", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.1", "2.8.0.dev")
    assert not _is_torch_equal_or_newer("2.7.1", "2.8.0.dev")


def test_use_cudagraphs_dynamic():
    vllm_config = VllmConfig()
    # Default V1 configuration now starts without cudagraphs enabled; the
    # engine decides when to capture based on runtime settings instead of a
    # blanket default.
    assert vllm_config.compilation_config.use_cudagraph


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
        "use_cudagraph": False,  # speed things up a bit
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
@pytest.mark.parametrize("enabled", [True, False])
def test_use_cudagraphs(vllm_runner, monkeypatch, enabled):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    compilation_config = {
        "cudagraph_capture_sizes": [100],
        "use_cudagraph": enabled,
    }
    with (
        compilation_counter.expect(
            num_graphs_seen=1,
            num_gpu_runner_capture_triggers=1 if enabled else 0,
            num_cudagraph_captured=13 if enabled else 0,
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
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert not config.compilation_config.splitting_ops_contain_attention()

    # When use_inductor_graph_partition=True
    if is_torch_equal_or_newer("2.9.0.dev"):
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

    # When attn_fusion pass enabled, splitting_ops now default to attention ops.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config={"enable_attn_fusion": True, "enable_noop": True},
            custom_ops=["+quant_fp8"],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    # With the new simplified logic, attention fusion works with splitting_ops
    assert config.compilation_config.splitting_ops_contain_attention()
    # cudagraph mode remains PIECEWISE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE

    # When both use_inductor_graph_partition and attn_fusion pass enabled.
    if is_torch_equal_or_newer("2.9.0.dev"):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                use_inductor_graph_partition=True,
                pass_config={"enable_attn_fusion": True, "enable_noop": True},
                custom_ops=["+quant_fp8"],
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
            )
        )
        # With inductor graph partition, attn_fusion and splitting_ops
        # work together. Default splitting_ops include attention ops.
        assert config.compilation_config.splitting_ops_contain_attention()
        # enable_attn_fusion is directly supported under
        # use_inductor_graph_partition=True, and cudagraph_mode
        # is unchanged.
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


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

    @torch.library.custom_op(
        "silly::attention",
        mutates_args=["out"],
    )
    def attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        out.copy_(q + k + v)

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
        "enable_sequence_parallelism",
        "max_num_batched_tokens",
        "use_cudagraph",
        "expected_max_size",
    ),
    [
        (None, None, 1, False, 2048, True, 512),
        ([1, 2, 4], 4, 1, False, 2048, True, 4),
        ([1, 2, 4], 8, 1, False, 2048, True, RuntimeError),
        ([1, 256], None, 1, False, 2048, 256),
        ([], None, 1, False, 2048, False, 0),
        (None, 0, 1, False, 2048, False, 0),
        # truncated to nearest multiple of 8 or 16
        (None, 257, 1, False, 2048, True, 256),
        ([1, 2, 4, 15], None, 1, False, 2048, True, 15),  # max from list
        ([1, 2, 4, 15], None, 2, True, 2048, True, 4),  # filtered out 15 due to SP
        ([1, 2, 4, 15], None, 1, False, 8, True, 4),  # limited by the max_tokens
        # the list should contain at least 1 element when use cudagraph
        ([], None, 1, False, 2048, True, RuntimeError),
        # the max capturing size should be >= 1 when use cudagraph
        (None, 0, 1, False, 2048, True, RuntimeError),
    ],
)
def test_cudagraph_sizes_post_init(
    cudagraph_capture_sizes,
    max_cudagraph_capture_size,
    tp_size,
    enable_sequence_parallelism,
    max_num_batched_tokens,
    use_cudagraph,
    expected_max_size,
):
    ctx = nullcontext()
    if isinstance(expected_max_size, Exception):
        ctx = pytest.raises(expected_max_size)

    cudagraph_mode = CUDAGraphMode.PIECEWISE if use_cudagraph else CUDAGraphMode.NONE
    with ctx:
        compilation_config = CompilationConfig(
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            max_cudagraph_capture_size=max_cudagraph_capture_size,
            pass_config={
                "enable_sequence_parallelism": enable_sequence_parallelism,
                "enable_fusion": True,
                "enable_noop": True,
            },
            cudagraph_mode=cudagraph_mode,
        )
        engine_args = EngineArgs(
            model="facebook/opt-125m",
            tensor_parallel_size=tp_size,
            max_num_batched_tokens=max_num_batched_tokens,
            compilation_config=compilation_config,
        )
        vllm_config = engine_args.create_engine_config()

    assert (
        vllm_config.compilation_config.max_cudagraph_capture_size == expected_max_size
    )
