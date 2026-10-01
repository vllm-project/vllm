# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.warmup import replayssm_warmup as warmup
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FlashInfer ReplaySSM warmup tests require CUDA and FlashInfer",
)

PREFILL_KWARGS = {
    "num_tokens": 128,
    "skip_eplb": True,
    "is_profile": True,
    "randomize_inputs": True,
}


def _autotune_runner(
    *,
    use_v2_model_runner: bool = False,
    query_len: int = 6,
    max_num_seqs: int = 32,
    num_blocks: int = 17,
    max_num_tokens: int = 100,
    use_replayssm: bool = True,
    backend: MambaBackendEnum = MambaBackendEnum.FLASHINFER,
) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=use_replayssm),
            mamba_config=SimpleNamespace(backend=backend),
            use_v2_model_runner=use_v2_model_runner,
        ),
        uniform_decode_query_len=query_len,
        decode_query_len=query_len,
        max_num_tokens=max_num_tokens,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_cache_config=SimpleNamespace(num_blocks=num_blocks),
    )


@pytest.mark.parametrize(
    ("runner_kwargs", "expected_num_reqs"),
    [
        # max_num_seqs (32) vs max_num_tokens // query_len (16) vs blocks-1 (16).
        (dict(query_len=6, use_v2_model_runner=False), 16),
        (dict(query_len=6, use_v2_model_runner=True), 16),
        # num_blocks - 1 is the binding constraint.
        (dict(query_len=1, max_num_tokens=128, max_num_seqs=64, num_blocks=5), 4),
    ],
    ids=["v1", "v2", "clamped_to_state_capacity"],
)
def test_replayssm_autotune_decode_kwargs(runner_kwargs, expected_num_reqs):
    query_len = runner_kwargs["query_len"]
    with patch.object(
        warmup, "flashinfer_replayssm_autotune_supported", return_value=True
    ):
        result = warmup._replayssm_autotune_kwargs(
            _autotune_runner(**runner_kwargs), PREFILL_KWARGS
        )

    assert result is not None
    max_num_reqs, decode_kwargs = result
    assert max_num_reqs == expected_num_reqs
    expected_kwargs = {
        **PREFILL_KWARGS,
        "num_tokens": expected_num_reqs * query_len,
        "uniform_decode": True,
    }
    if runner_kwargs.get("use_v2_model_runner"):
        expected_kwargs["valid_dummy_state_slots"] = True
    else:
        expected_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=query_len + 1,
        )
    assert decode_kwargs == expected_kwargs


@pytest.mark.parametrize(
    ("runner_kwargs", "flashinfer_supported"),
    [
        (dict(use_replayssm=False), True),
        (dict(backend=MambaBackendEnum.TRITON), True),
        ({}, False),
        (dict(num_blocks=1), True),
    ],
    ids=[
        "replayssm_disabled",
        "non_flashinfer_backend",
        "kernel_unavailable",
        "zero_non_padding_slots",
    ],
)
def test_replayssm_autotune_kwargs_skipped(runner_kwargs, flashinfer_supported):
    with patch.object(
        warmup,
        "flashinfer_replayssm_autotune_supported",
        return_value=flashinfer_supported,
    ):
        result = warmup._replayssm_autotune_kwargs(
            _autotune_runner(**runner_kwargs), PREFILL_KWARGS
        )
    assert result is None


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("fail_warmup", [False, True])
def test_replayssm_autotune_slots_restore_state_and_trackers(use_v2, fail_warmup):
    mixer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.use_replayssm = True
    mixer.replayssm_buffer_len = 16
    mixer.kv_cache = (
        torch.full((4, 2), 3.0),
        torch.full((4, 2), 3.0),
    )
    mixer.replayssm_cache = tuple(torch.full((4, 2, 17), 3.0) for _ in range(3))
    mixer._replayssm_ring_start = torch.full((4,), 3, dtype=torch.int32)
    mixer._replayssm_prev_num_accepted = torch.full((4,), 3, dtype=torch.int32)
    tracked = (
        *mixer.kv_cache,
        *mixer.replayssm_cache,
        mixer._replayssm_ring_start,
        mixer._replayssm_prev_num_accepted,
    )

    block_ids = np.arange(10, 14, dtype=np.int32).reshape(4, 1)
    original_block_ids = block_ids.copy()
    block_table = SimpleNamespace(block_table=SimpleNamespace(np=block_ids))
    multi_group_block_table = SimpleNamespace(
        block_tables=[block_table], commit_block_table=Mock()
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=use_v2),
        input_batch=SimpleNamespace(block_table=multi_group_block_table),
        block_tables=SimpleNamespace(get_dummy_block_tables=Mock()),
        get_model=lambda: SimpleNamespace(modules=lambda: (mixer,)),
    )

    error = (
        pytest.raises(RuntimeError, match="warmup failed")
        if fail_warmup
        else nullcontext()
    )
    with error, warmup._temporary_replayssm_autotune_state(runner, 2):
        if not use_v2:
            assert block_ids[:2, 0].tolist() == [1, 2]
        for tensor in tracked:
            tensor[1:3].fill_(9)
        if fail_warmup:
            raise RuntimeError("warmup failed")

    assert np.array_equal(block_ids, original_block_ids)
    if use_v2:
        runner.block_tables.get_dummy_block_tables.assert_called_once_with(2)
    else:
        assert multi_group_block_table.commit_block_table.call_args_list == [
            ((2,), {}),
            ((2,), {}),
        ]
    for tensor in (*mixer.kv_cache, *mixer.replayssm_cache):
        assert torch.count_nonzero(tensor[1:3]) == 0
        assert torch.all(tensor[0] == 3)
        assert torch.all(tensor[3] == 3)
    for tensor in (mixer._replayssm_ring_start, mixer._replayssm_prev_num_accepted):
        assert torch.count_nonzero(tensor) == 0


def test_prepare_replayssm_runtime_binds_every_mixer():
    mixers = []
    for _ in range(2):
        mixer = MambaMixer2.__new__(MambaMixer2)
        torch.nn.Module.__init__(mixer)
        mixer.prepare_replayssm_runtime_kernel = Mock()
        mixers.append(mixer)
    runner = SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=lambda: (*mixers, object()))
    )

    with patch.object(warmup, "_replayssm_autotune_kwargs", return_value=(37, {})):
        warmup.prepare_replayssm_runtime(runner)

    for mixer in mixers:
        mixer.prepare_replayssm_runtime_kernel.assert_called_once_with(37)


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32, torch.int8])
def test_mixer_prepares_exact_replayssm_runtime_specialization(state_dtype):
    mixer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.use_flashinfer_replayssm = True
    mixer.num_spec = 3
    mixer._replayssm_runtime_kernel = None
    mixer._replayssm_runtime_policies = ()
    mixer.kv_cache = (
        torch.empty(0),
        torch.empty(2, 32, 64, 128, dtype=state_dtype),
    )
    mixer.replayssm_cache = (
        torch.empty(2, 32, 20, 64, dtype=torch.bfloat16),
        torch.empty(2, 32, 20, dtype=torch.float32),
        torch.empty(2, 2, 20, 128, dtype=torch.bfloat16),
    )
    mixer.D = torch.nn.Parameter(torch.empty(32, dtype=torch.bfloat16))
    mixer.A = torch.nn.Parameter(torch.empty(32, dtype=torch.float32))
    mixer.mamba_config = SimpleNamespace(
        enable_stochastic_rounding=True,
        stochastic_rounding_philox_rounds=5,
    )
    kernel = Mock()
    policies = ((False, 1, 0, 0, 0),) * 38

    with patch(
        "vllm.model_executor.layers.mamba.mamba_mixer2."
        "prepare_flashinfer_replayssm_runtime",
        return_value=(kernel, policies),
    ) as prepare:
        mixer.prepare_replayssm_runtime_kernel(37)

    prepare.assert_called_once_with(
        state_dtype=state_dtype,
        input_dtype=torch.bfloat16,
        dt_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        matrix_a_dtype=torch.float32,
        state_index_dtype=torch.int32,
        dim=64,
        dstate=128,
        num_predicted_tokens=4,
        max_window=16,
        heads_per_group=16,
        num_groups=2,
        max_batch_size=37,
        device=mixer.kv_cache[1].device,
        use_rand_seed=True,
        philox_rounds=5,
    )
    assert mixer._replayssm_runtime_kernel is kernel
    assert mixer._replayssm_runtime_policies is policies
