# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import pp_utils


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_activation_transport_rejects_non_bf16_schema() -> None:
    schema = IntermediateTensors({"hidden_states": torch.empty(8, 16)})
    with pytest.raises(ValueError, match="Expected BF16"):
        pp_utils.PPActivationTransport(
            schema,
            chunk_tokens=8,
            ring_size=2,
            device=torch.device("cpu"),
        )


def test_activation_transport_full_chunk_and_override_gate() -> None:
    transport = pp_utils.PPActivationTransport.__new__(pp_utils.PPActivationTransport)
    transport.chunk_tokens = 8
    transport.keys = ("hidden_states", "residual")

    assert transport.can_transfer(8, {})
    assert not transport.can_transfer(7, {})
    assert not transport.can_transfer(8, {"residual": False})


def test_prepare_pp_intermediate_tensors_reuses_receive_target() -> None:
    hidden = torch.zeros(8, 16, dtype=torch.bfloat16)
    residual = torch.ones(8, 16, dtype=torch.bfloat16)
    persistent = IntermediateTensors(
        {
            "hidden_states": hidden,
            "residual": residual,
        }
    )

    prepared = pp_utils.prepare_pp_intermediate_tensors(
        persistent,
        persistent,
        num_tokens=4,
        dummy_run=False,
    )

    assert prepared["hidden_states"].data_ptr() == hidden.data_ptr()
    assert prepared["residual"].data_ptr() == residual.data_ptr()


def test_prepare_pp_intermediate_tensors_copies_generic_receive() -> None:
    persistent = IntermediateTensors(
        {
            "hidden_states": torch.zeros(8, 16, dtype=torch.bfloat16),
            "residual": torch.zeros(8, 16, dtype=torch.bfloat16),
        }
    )
    received = IntermediateTensors(
        {
            "hidden_states": torch.ones(4, 16, dtype=torch.bfloat16),
            "residual": torch.full((4, 16), 2, dtype=torch.bfloat16),
        }
    )

    prepared = pp_utils.prepare_pp_intermediate_tensors(
        persistent,
        received,
        num_tokens=4,
        dummy_run=False,
    )

    torch.testing.assert_close(prepared["hidden_states"], received["hidden_states"])
    torch.testing.assert_close(prepared["residual"], received["residual"])


def test_prepare_pp_intermediate_tensors_accepts_sp_local_rows() -> None:
    persistent = IntermediateTensors(
        {
            "hidden_states": torch.zeros(8, 16, dtype=torch.bfloat16),
        }
    )
    received = IntermediateTensors(
        {
            "hidden_states": torch.ones(2, 16, dtype=torch.bfloat16),
        }
    )

    prepared = pp_utils.prepare_pp_intermediate_tensors(
        persistent,
        received,
        num_tokens=8,
        dummy_run=False,
    )

    assert prepared["hidden_states"].shape == (2, 16)
    torch.testing.assert_close(prepared["hidden_states"], received["hidden_states"])


def test_activation_transport_falls_back_on_cross_rank_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = Mock(world_size=2, cpu_group=object())
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: group)

    def all_gather_object(descriptors, descriptor, group):
        descriptors[:] = [descriptor, (("residual", (8, 16), torch.bfloat16, True),)]

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        all_gather_object,
    )
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.device = torch.device("cpu")
    handler.activation_transport = None
    schema = IntermediateTensors(
        {
            "hidden_states": torch.empty(8, 16, dtype=torch.bfloat16),
        }
    )

    assert not handler.init_activation_transport(schema, chunk_tokens=8)
    assert handler.activation_transport is None


def test_activation_transport_drain_and_reap() -> None:
    transport = pp_utils.PPActivationTransport.__new__(pp_utils.PPActivationTransport)
    ring_work = Mock()
    metadata_work = Mock()
    generic_work = Mock()
    generic_work.is_completed.return_value = True
    transport.send_work = [[ring_work], None]
    transport.generic_send_work = [[metadata_work, generic_work]]
    transport.group = Mock()

    transport._reap_generic_send_work()
    ring_work.wait.assert_not_called()
    metadata_work.wait.assert_called_once_with()
    generic_work.wait.assert_called_once_with()

    transport.drain()
    ring_work.wait.assert_called_once_with()
    transport.group.drain_pending_isends.assert_called_once_with()
    assert transport.send_work == [None, None]
    assert transport.generic_send_work == []


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]
