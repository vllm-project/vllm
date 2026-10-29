# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm.compilation.compiler_interface import InductorStandaloneAdaptor


@pytest.mark.parametrize("save_format", ["binary", "unpacked"])
def test_inductor_standalone_load_uses_current_cache_dir(
    tmp_path: Path,
    save_format: str,
):
    old_cache_dir = tmp_path / "old"
    new_cache_dir = tmp_path / "new"
    key = "artifact_shape_None_subgraph_0"

    adaptor = InductorStandaloneAdaptor(use_aot_compile=True, save_format=save_format)
    adaptor.initialize_cache(str(new_cache_dir))

    # Simulate a handle persisted before the cache directory was relocated.
    handle = (key, str(old_cache_dir / key))

    with (
        patch(
            "torch._inductor.CompiledArtifact.load",
            return_value=MagicMock(),
        ) as load_mock,
        patch(
            "torch._inductor.compile_fx.graph_returns_tuple",
            return_value=True,
        ),
    ):
        adaptor.load(
            handle=handle,
            graph=MagicMock(),
            example_inputs=[],
            graph_index=0,
            compile_range=MagicMock(),
        )

    load_mock.assert_called_once_with(
        path=str(new_cache_dir / key),
        format=save_format,
    )


@pytest.mark.parametrize("raises", [False, True])
def test_opt_out_functorch_config_is_scoped(monkeypatch, raises):
    from contextlib import nullcontext

    import torch

    from vllm.compilation.compiler_interface import (
        get_inductor_factors,
        set_functorch_config,
    )

    config = torch._functorch.config
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
    from vllm.envs import disable_envs_cache

    disable_envs_cache()
    with config.patch(bundled_autograd_cache=True):
        error = pytest.raises(RuntimeError, match="probe") if raises else nullcontext()
        with error, set_functorch_config(use_aot_compile=False):
            assert config.bundled_autograd_cache is False
            if raises:
                raise RuntimeError("probe")
        assert config.bundled_autograd_cache is True
        original = type(config).save_config_portable
        observed = []

        def observe(module):
            if module is config:
                observed.append(config.bundled_autograd_cache)
            return original(module)

        with patch.object(type(config), "save_config_portable", new=observe):
            get_inductor_factors(use_aot_compile=False)
        assert observed == [False]
        assert config.bundled_autograd_cache is True


def test_opt_out_ignores_existing_direct_cache_index(tmp_path, monkeypatch):
    import builtins

    from vllm.compilation.backends import CompilerManager
    from vllm.config import CompilationConfig, CompilationMode
    from vllm.config.utils import Range
    from vllm.envs import disable_envs_cache

    monkeypatch.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
    disable_envs_cache()
    cache = tmp_path / "rank_0_0" / "decoder"
    cache.mkdir(parents=True)
    index = cache / "vllm_compile_cache.py"
    index.write_text("old index must not be read")
    manager = CompilerManager(
        CompilationConfig(mode=CompilationMode.VLLM_COMPILE), use_aot_compile=False
    )
    real_open = builtins.open
    reads = []

    def check_open(path, *args, **kwargs):
        if str(path) == str(index):
            reads.append(path)
            raise AssertionError("read old index")
        return real_open(path, *args, **kwargs)

    with (
        patch.object(manager.compiler, "initialize_cache"),
        patch("builtins.open", side_effect=check_open),
    ):
        manager.initialize_cache(str(cache), prefix="decoder")
    assert Path(manager.cache_file_path) == index
    assert reads == []
    compile_range = Range(1, 8)
    manager.cache[(compile_range, 0, manager.compiler.name)] = {
        "graph_handle": ("old", "unused"),
        "cache_key": "old",
    }
    with patch.object(
        manager.compiler, "load", side_effect=AssertionError("direct load")
    ) as load:
        assert manager.load(MagicMock(), [], 0, compile_range) is None
        load.assert_not_called()
    assert index.read_text() == "old index must not be read"
