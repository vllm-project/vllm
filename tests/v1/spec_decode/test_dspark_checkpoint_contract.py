# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _validate_qwen3_vl_weight_contract,
)

pytestmark = pytest.mark.skip_global_cleanup


class _DraftConfig:
    architectures = ["Qwen3VLDSparkModel"]

    def __init__(self, input_vocab_size: int, output_vocab_size: int):
        self.input_vocab_size = input_vocab_size
        self.hf_config = SimpleNamespace(draft_vocab_size=output_vocab_size)

    def get_vocab_size(self) -> int:
        return self.input_vocab_size


def test_qwen3_vl_full_vocab_checkpoint_can_share_weights() -> None:
    draft = SimpleNamespace()
    config = _DraftConfig(input_vocab_size=128, output_vocab_size=128)

    _validate_qwen3_vl_weight_contract(draft, config, target_vocab_size=128)


@pytest.mark.parametrize(
    ("draft", "config", "error"),
    [
        (
            SimpleNamespace(),
            _DraftConfig(input_vocab_size=129, output_vocab_size=128),
            "must include embed_tokens weights",
        ),
        (
            SimpleNamespace(),
            _DraftConfig(input_vocab_size=128, output_vocab_size=64),
            "must include lm_head weights",
        ),
        (
            SimpleNamespace(has_own_lm_head=True),
            _DraftConfig(input_vocab_size=128, output_vocab_size=64),
            "must include a d2t token mapping",
        ),
    ],
)
def test_qwen3_vl_checkpoint_requires_non_shared_weights(draft, config, error) -> None:
    with pytest.raises(ValueError, match=error):
        _validate_qwen3_vl_weight_contract(draft, config, target_vocab_size=128)


def test_qwen3_vl_reduced_vocab_checkpoint_with_own_weights_is_valid() -> None:
    draft = SimpleNamespace(
        has_own_lm_head=True,
        has_own_draft_id_mapping=True,
    )
    config = _DraftConfig(input_vocab_size=128, output_vocab_size=64)

    _validate_qwen3_vl_weight_contract(draft, config, target_vocab_size=128)
