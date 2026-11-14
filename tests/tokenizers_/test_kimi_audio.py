# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pybase64

from vllm.tokenizers.kimi_audio import KimiAudioTokenizer

SPECIAL_TOKENS = {"<|im_end|>": 100, "<|im_user_msg_start|>": 101}


def test_config_special_tokens_are_consistent(tmp_path: Path):
    """Config special tokens behave the same in decode, encode and conversion."""
    vocab_file = tmp_path / "tiktoken.model"
    vocab_file.write_text(
        "\n".join(
            f"{pybase64.b64encode(token.encode()).decode()} {rank}"
            for rank, token in enumerate(["hello", " world"])
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    str(token_id): {"content": token, "special": True}
                    for token, token_id in SPECIAL_TOKENS.items()
                }
            }
        ),
        encoding="utf-8",
    )
    tokenizer = KimiAudioTokenizer(
        vocab_file=vocab_file, name_or_path=str(tmp_path), truncation_side="left"
    )

    # `TokenizerLike.all_special_tokens` is a `list[str]`: `_build_drop_info`
    # and the detokenizer compare its entries against token strings.
    assert all(isinstance(token, str) for token in tokenizer.all_special_tokens)
    # `convert_ids_to_tokens` must skip exactly what `decode` skips, including
    # config-declared tokens that never enter `added_tokens_decoder`.
    ids = list(SPECIAL_TOKENS.values())
    assert tokenizer.decode(ids, skip_special_tokens=True) == ""
    assert tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) == []
    # Config-declared special tokens must also be encodable from prompt text.
    assert tokenizer.encode("hello<|im_end|>")[-1] == SPECIAL_TOKENS["<|im_end|>"]
