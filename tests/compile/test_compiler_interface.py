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

    adaptor = InductorStandaloneAdaptor(save_format=save_format)
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
