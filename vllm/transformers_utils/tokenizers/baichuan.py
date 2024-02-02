# Adapted from
# https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/99fe8c0fb8bfb8e821d281c4302eb98999e50761/generation_utils.py

from typing import List
import types

from transformers import AutoTokenizer

BAI_CHUAN_CONFIG = {
    "user_token_id": 195,
    "assistant_token_id": 196,
}


def BaichuanTokenizer(*args, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)

    def build_chat_input(self, messages: List[dict]):

        def _parse_messages(messages, split_role="user"):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    assert i == 0
                    system = message["content"]
                    continue
                if message["role"] == split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds

        system, rounds = _parse_messages(messages, split_role="user")
        system_tokens = self.encode(system)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(BAI_CHUAN_CONFIG["user_token_id"])
                else:
                    round_tokens.append(BAI_CHUAN_CONFIG["assistant_token_id"])
                round_tokens.extend(self.encode(message["content"]))
            history_tokens = round_tokens + history_tokens  # concat left

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(BAI_CHUAN_CONFIG["assistant_token_id"])
        return input_tokens

    tokenizer.build_chat_input = types.MethodType(build_chat_input, tokenizer)
    return tokenizer
