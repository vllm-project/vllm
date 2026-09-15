# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.model_executor.layers.attention.direct_kv import validate_direct_final
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _config():
    return NS(
        parallel_config=NS(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            num_ubatches=0,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        model_config=NS(enable_sleep_mode=False),
        attention_config=NS(
            hisparse_config=None, resolve_indexer_kv_dtype=lambda _: "fp8"
        ),
        cache_config=NS(cache_dtype="fp8"),
        compilation_config=NS(
            static_forward_context={"layer": NS(use_pcp=True, pcp_vmm_domain=None)}
        ),
    )


@pytest.mark.parametrize("case", ["backing", "dcp", "pcp", "dtype", "layer", "sleep"])
def test_requested_direct_mode_rejects_unsupported(monkeypatch, case):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    config = _config()
    validate_direct_final(config, True)
    if case == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif case == "pcp":
        config.parallel_config.prefill_context_parallel_size = 1
    elif case == "dtype":
        config.cache_config.cache_dtype = "nvfp4"
    elif case == "layer":
        config.compilation_config.static_forward_context["layer"] = NS(use_pcp=True)
    elif case == "sleep":
        config.model_config.enable_sleep_mode = True
    with pytest.raises(ValueError, match="VLLM_USE_PCP_DIRECT_KV=1"):
        validate_direct_final(config, case != "backing")


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_direct_slots_use_stable_local_prefix(rank):
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=rank,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=16,
        use_local_kv_slot_mappings=True,
    )
    manager._gathered_kv_slot_mappings = torch.empty((1, 32), dtype=torch.int64)
    base = manager._gathered_kv_slot_mappings.data_ptr()
    for rows in (1, 4, 2):
        manager._padded_gather_idx = torch.arange(rows).repeat(4)
        manager._gathered_kv_write_mask = torch.zeros(4 * rows, dtype=torch.bool)
        # Replicated decode: only rank zero publishes each cache row.
        manager._gathered_kv_write_mask[:rows] = True
        result = manager._convert_to_gathered_slot_mappings(
            torch.arange(10, 10 + rows).unsqueeze(0)
        )
        assert result.data_ptr() == base
        assert result.shape == (1, rows)
        expected = torch.arange(10, 10 + rows) if rank == 0 else torch.full((rows,), -1)
        torch.testing.assert_close(result[0], expected)
        assert manager.get_dummy_slot_mappings(rows).data_ptr() == base
