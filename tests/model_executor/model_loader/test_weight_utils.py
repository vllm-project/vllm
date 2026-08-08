# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import (
    DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
    DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
)
from vllm.model_executor.model_loader import weight_utils


@pytest.mark.parametrize(
    ("fs_type", "strategy", "available_ram_multiplier", "should_prefetch"),
    [
        pytest.param("virtiofs", None, 2, True, id="virtiofs"),
        pytest.param("virtiofs", None, 1, False, id="virtiofs-low-ram"),
        pytest.param("virtiofs", "lazy", 2, False, id="virtiofs-explicit-lazy"),
        pytest.param("ext4", None, 2, False, id="ext4"),
    ],
)
def test_auto_prefetch_selection(
    tmp_path,
    monkeypatch,
    fs_type,
    strategy,
    available_ram_multiplier,
    should_prefetch,
):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2)}, checkpoint)

    monkeypatch.setattr(weight_utils, "_get_fs_type", lambda _: fs_type)
    monkeypatch.setattr(
        weight_utils,
        "_get_available_ram_bytes",
        lambda: checkpoint.stat().st_size * available_ram_multiplier,
    )
    prefetch_calls = []
    monkeypatch.setattr(
        weight_utils,
        "_prefetch_all_checkpoints",
        lambda *args, **kwargs: prefetch_calls.append((args, kwargs)),
    )

    weights = dict(
        weight_utils.safetensors_weights_iterator(
            [str(checkpoint)],
            use_tqdm_on_load=False,
            safetensors_load_strategy=strategy,
        )
    )

    assert torch.equal(weights["weight"], torch.ones(2))
    assert bool(prefetch_calls) is should_prefetch
    if should_prefetch:
        assert prefetch_calls == [
            (
                ([str(checkpoint)],),
                {
                    "num_prefetch_threads": (DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS),
                    "block_size": DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
                },
            )
        ]
