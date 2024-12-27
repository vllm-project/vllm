from pathlib import Path

import pytest
import transformers

from ..utils import VLLM_PATH

EXAMPLES_DIR = VLLM_PATH / "examples"

jinja_paths = [
    pytest.param(path, id=path.stem)
    for path in sorted(EXAMPLES_DIR.glob("*.jinja"))
]


@pytest.mark.parametrize("path", jinja_paths)
@pytest.mark.parametrize("num_messages", [1, 3])
def test_bos(path: Path, num_messages: int) -> None:
    with path.open("r", encoding="utf-8") as f:
        chat_template = f.read()
    # We might guess an appropriate tokenizer model from the file name but we
    # don't maintain such list.
    # Use arbitrary BOS for testing. It doesn't have to match the str in the
    # correct tokenizer.
    bos_token = "=BOS="
    tokenizer = transformers.PreTrainedTokenizerBase(
        chat_template=chat_template, bos_token=bos_token, eos_token="=EOS=")
    conversation = [
        {
            "role": "user",
            "content": "1"
        },
        {
            "role": "assistant",
            "content": "2"
        },
        {
            "role": "user",
            "content": "3"
        },
    ][:num_messages]
    try:
        prompt: str = tokenizer.apply_chat_template(conversation=conversation,
                                                    tokenize=False)
    except Exception as e:
        if str(e
               ) == "Embedding models should only embed one message at a time":
            pytest.skip(reason=str(e))
        raise
    assert prompt.startswith(bos_token)
    assert prompt.count(bos_token) == 1
