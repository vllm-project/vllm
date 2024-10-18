import pytest

from vllm import LLM

from ...utils import error_on_warning

MODEL_NAME = "facebook/opt-125m"


def test_pos_args_deprecated():
    with error_on_warning(DeprecationWarning):
        LLM(model=MODEL_NAME, tokenizer=MODEL_NAME)

    with error_on_warning(DeprecationWarning):
        LLM(MODEL_NAME, tokenizer=MODEL_NAME)

    with pytest.warns(DeprecationWarning, match="'tokenizer'"):
        LLM(MODEL_NAME, MODEL_NAME)

    with pytest.warns(DeprecationWarning,
                      match="'tokenizer', 'tokenizer_mode'"):
        LLM(MODEL_NAME, MODEL_NAME, "auto")
