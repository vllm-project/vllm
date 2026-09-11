# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

from vllm.v1.worker.gpu.sample import gumbel


def test_gumbel_warmup_covers_runtime_signatures() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_vocab_size=lambda: 8192,
            head_dtype=torch.bfloat16,
        ),
    )

    cases = list(
        gumbel._gumbel_sample_kernel._provider_cases(
            gumbel._gumbel_sample_kernel._warmup_inputs_fn,
            config,
        )
    )

    signatures = {
        (
            case["logits_ptr"].dtype,
            case["local_max_ptr"].dtype,
            case["logits_cache_ptr"] is not None,
            case["IS_DRAFTING"],
            case["APPLY_TEMPERATURE"],
            case["PER_TOKEN_COL"],
            None
            if case["logits_cache_col_ptr"] is None
            else case["logits_cache_col_ptr"].aligned,
        )
        for case in cases
    }
    expected_modes = {
        (False, False, False, False, None),
        (True, True, True, False, True),
        (True, True, True, False, False),
        (True, True, True, True, True),
    }

    assert signatures == {
        (logits_dtype, local_max_dtype, *mode)
        for logits_dtype in (torch.float32, torch.bfloat16)
        for local_max_dtype in (torch.float32, torch.float64)
        for mode in expected_modes
    }


def test_sampling_states_registers_gumbel_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker.gpu.sample import states

    registrations: list[str] = []

    class FakeUvaBackedTensor:
        def __init__(self, size: int, **_kwargs) -> None:
            self.np = np.zeros(size)

        def copy_to_uva(self) -> None:
            pass

    monkeypatch.setattr(states, "UvaBackedTensor", FakeUvaBackedTensor)
    monkeypatch.setattr(
        states,
        "register_top_k_top_p_warmups",
        lambda: registrations.append("topk_topp"),
    )
    monkeypatch.setattr(
        states, "register_gumbel_warmup", lambda: registrations.append("gumbel")
    )

    states.SamplingStates(max_num_reqs=2, vocab_size=8192)
    assert registrations == ["topk_topp", "gumbel"]
