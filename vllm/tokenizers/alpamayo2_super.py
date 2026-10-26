# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer for Alpamayo 2 Super.

The checkpoint ships the base Qwen3-VL tokenizer. The reference implementation
(NVlabs/alpamayo2, ``build_alpamayo2_super_tokenizer``) appends the discrete
trajectory tokens and the model's special tokens in code; the model's
embeddings and ``traj_ids`` assume those ids. Append them the same way, so
prompts with ego-trajectory tokens tokenize as in training and generated
special tokens such as ``<|traj_future_start|>`` detokenize.
"""

from pathlib import Path

from transformers import AutoTokenizer

from vllm.tokenizers.hf import HfTokenizer, get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike
from vllm.transformers_utils.configs.alpamayo2_super import Alpamayo2SuperConfig

# The order defines the token ids that follow the trajectory tokens.
SPECIAL_TOKENS = tuple(
    f"<|{name}|>"
    for name in (
        "prompt_start",
        "prompt_end",
        "image_start",
        "image_pre_tkn",
        "image_end",
        "traj_history_start",
        "traj_history_pre_tkn",
        "traj_history_end",
        "cot_start",
        "cot_end",
        "meta_action_start",
        "meta_action_end",
        "traj_future_start",
        "traj_future_pre_tkn",
        "traj_future_end",
        "traj_history",
        "traj_future",
        "image_pad",  # already in the Qwen3-VL vocabulary; keeps its id
        "vectorized_wm",
        "vectorized_wm_start",
        "vectorized_wm_end",
        "vectorized_wm_pre_tkn",
        "route_start",
        "route_pad",
        "route_end",
        "question_start",
        "question_end",
        "answer_start",
        "answer_end",
    )
)

# config.traj_ids key -> the token it must resolve to
_TRAJ_ID_TOKENS = {
    "history_start": "<|traj_history_start|>",
    "history_end": "<|traj_history_end|>",
    "history_pad": "<|traj_history|>",
    "future_start": "<|traj_future_start|>",
    "future_end": "<|traj_future_end|>",
    "future_pad": "<|traj_future|>",
}


class Alpamayo2SuperTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
            **kwargs,
        )
        config = Alpamayo2SuperConfig.from_pretrained(
            path_or_repo_id, revision=revision or "main", cache_dir=download_dir
        )
        # Tokens that are already present (e.g. in a pre-extended tokenizer)
        # keep their ids.
        tokenizer.add_tokens([f"<i{i}>" for i in range(config.traj_vocab_size)])
        tokenizer.add_tokens(list(SPECIAL_TOKENS), special_tokens=True)

        expected = {
            "history_id0": "<i0>",
            "future_id0": f"<i{config.history_vocab_size}>",
            **_TRAJ_ID_TOKENS,
        }
        for key, token in expected.items():
            if key in config.traj_ids:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != config.traj_ids[key]:
                    raise ValueError(
                        f"Alpamayo 2 Super tokenizer maps {token} to {token_id}, "
                        f"but the config expects {config.traj_ids[key]} "
                        f"(traj_ids[{key!r}])."
                    )
        return get_cached_tokenizer(tokenizer)
