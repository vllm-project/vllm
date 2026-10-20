# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for loading custom logits processors by FQCN.

Kept separate from `test_custom_offline.py`, which runs end-to-end against a
downloaded model; these exercise the loader alone and need no model.
"""

import pytest

from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns


@pytest.mark.parametrize(
    "fqcn",
    [
        "invalid_fqcn_without_colon",
        "invalid:fqcn:with:multiple:colons",
    ],
    ids=["missing-colon", "too-many-colons"],
)
def test_invalid_fqcn_rejected(fqcn: str):
    with pytest.raises(ValueError, match="Invalid logits processor FQCN"):
        _load_logitsprocs_by_fqcns([fqcn])
