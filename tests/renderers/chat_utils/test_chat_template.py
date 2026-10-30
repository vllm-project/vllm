# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils import validate_chat_template


def test_validate_chat_template_rejects_invalid_type():
    """A non-str/Path chat_template is invalid user input."""
    with pytest.raises(
        VLLMValidationError, match="not a valid chat template type"
    ) as exc_info:
        validate_chat_template(123)
    assert exc_info.value.parameter == "chat_template"
