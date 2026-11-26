# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import model


@pytest.mark.parametrize("last,mtp", [(True, False), (True, True), (False, False)])
def test_sequence_parallel_head_gathers_only_required_streams(monkeypatch, last, mtp):
    # Rank zero has three rows for a padded five-token input.
    row = torch.arange(12).reshape(1, 4, 3).float()
    inputs = row.repeat(5, 1, 1)
    gather = Mock(side_effect=lambda x: torch.cat((x, x)))
    monkeypatch.setattr(model, "sp_shard", lambda x: x[:3])
    monkeypatch.setattr(model, "sp_all_gather", gather)
    monkeypatch.setattr(
        model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=last),
    )
    monkeypatch.setattr(model, "mhc_post_tilelang", lambda x, *args: x)
    monkeypatch.setattr(
        model, "hc_head_fused_kernel_tilelang", lambda x, *args: x.sum(1)
    )
    buffer = torch.empty(5, 12) if mtp else None
    norm = Mock(side_effect=lambda x: x)
    stub = SimpleNamespace(
        use_native_mega_moe=False,
        use_sequence_parallel=True,
        use_mega_gate=False,
        start_layer=0,
        end_layer=1,
        layers=[lambda x, *args, **kwargs: (x, None, None, None)],
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=buffer,
        norm=norm,
        hc_head_fn=None,
        hc_head_base=None,
        hc_head_scale=None,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        pack_local_aux_hidden_states=lambda _: {},
    )
    ids = torch.arange(5)
    out = model.DeepseekV4Model.forward(stub, ids, ids, None, inputs)
    shape = (3, 3) if last and not mtp else (3, 4, 3)
    assert gather.call_count == 1
    assert gather.call_args.args[0].shape == shape
    if last:
        # Token-dependent RMSNorm implementations need the original batch size.
        assert norm.call_args.args[0].shape == (5, 3)
        torch.testing.assert_close(out, inputs.sum(1))
    else:
        torch.testing.assert_close(out["hidden_states"], inputs)
    if mtp:
        torch.testing.assert_close(buffer, inputs.flatten(1))
