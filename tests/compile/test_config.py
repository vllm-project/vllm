# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.utils import _is_torch_equal_or_newer


def test_version():
    assert _is_torch_equal_or_newer("2.8.0.dev20250624+cu128", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0a0+gitc82a174", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.1", "2.8.0.dev")
    assert not _is_torch_equal_or_newer("2.7.1", "2.8.0.dev")


def test_use_cudagraphs_dynamic():
    vllm_config = VllmConfig()
    assert vllm_config.compilation_config.use_cudagraph


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
def test_dynamo_as_is(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        compilation_counter.expect(dynamo_as_is_count=1),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m",
            compilation_config={"level": 1},
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
        compilation_counter.expect(num_graphs_seen=0, dynamo_as_is_count=0),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m",
            compilation_config={"level": 0},
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
        compilation_counter.expect(num_graphs_seen=0, dynamo_as_is_count=0),
        # loading the model causes compilation (if enabled) to happen
        vllm_runner(
            "facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.4
        ) as _,
    ):
        pass


def test_splitting_ops_dynamic():
    # Default config
    config = VllmConfig()
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    assert config.compilation_config.splitting_ops_contain_attention()

    # When use_inductor_graph_partition=True
    if _is_torch_equal_or_newer("2.9.0.dev"):
        # inductor graph partition is only available in PyTorch 2.9+.
        # this is a fast config check so we are not using pytest.skip.
        config = VllmConfig(
            compilation_config=CompilationConfig(
                use_inductor_graph_partition=True, splitting_ops=["silly_attention"]
            )
        )
        # should ignore splitting_ops
        assert config.compilation_config.splitting_ops == []

    # When attn_fusion pass enabled.
    config = VllmConfig(
        compilation_config=CompilationConfig(
            pass_config={"enable_attn_fusion": True, "enable_noop": True},
            custom_ops=["+quant_fp8"],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    assert config.compilation_config.splitting_ops == []
    # cudagraph mode also fall back to FULL
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL

    # splitting_ops can not contain attention ops when attn_fusion
    # pass enabled.
    with pytest.raises(AssertionError):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                pass_config={"enable_attn_fusion": True, "enable_noop": True},
                custom_ops=["+quant_fp8"],
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                # work around for accessing all attntion ops
                splitting_ops=CompilationConfig()._attention_ops,
            )
        )

    # When both use_inductor_graph_partition and attn_fusion pass enabled.
    if _is_torch_equal_or_newer("2.9.0.dev"):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                use_inductor_graph_partition=True,
                pass_config={"enable_attn_fusion": True, "enable_noop": True},
                custom_ops=["+quant_fp8"],
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
            )
        )
        assert config.compilation_config.splitting_ops == []
        # enable_attn_fusion is directly support under
        # use_inductor_graph_partition=True, and cudagraph_mode
        # is unchanged.
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
