# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
)


def test_disable_mtp_completeness_check_is_scoped():
    assert is_mtp_completeness_check_enabled()

    with pytest.raises(RuntimeError), disable_mtp_completeness_check():
        assert not is_mtp_completeness_check_enabled()
        raise RuntimeError

    assert is_mtp_completeness_check_enabled()
