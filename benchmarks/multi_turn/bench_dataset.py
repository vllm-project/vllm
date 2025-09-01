# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from statistics import mean
from typing import Any, NamedTuple, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from bench_utils import (
    TEXT_SEPARATOR,
    Color,
    logger,
)
from transformers import AutoTokenizer  # type: ignore

# Conversation ID is a string (e.g: "UzTK34D")
ConvId = str

# A list of dicts (dicts with keys "id" and "messages")
ShareGptConversations = list[dict[str, Any]]

# A list of dicts (dicts with keys "role" and "content")
MessagesList = list[dict[str, str]]

# Map conversation ID to conversation messages
ConversationsMap = list[ConvId, MessagesList]


class Distribution(ABC):
    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        pass


class UniformDistribution(Distribution):
    def __init__(
        self,
        min_val: Union[int, float],
        max_val: Union[int, float],
        is_integer: bool = True,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer

    def sample(self, size: int = 1) -> np.ndarray:
        if self.is_integer:
            return np.random.randint(
                int(self.min_val), int(self.max_val + 1), size=size
            )
        else:
            return np.random.uniform(self.min_val, self.max_val, size=size)

    def __repr__(self) -> str:
        return f"UniformDistribution[{self.min_val}, {self.max_val}]"


class ConstantDistribution(Distribution):
    def __init__(self, value: Union[int, float]) -> None:
        self.value = value
        self.max_val = value

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value)

    def __repr__(self) -> str:
        return f"Constant[{self.value}]"


class ZipfDistribution(Distribution):
    def __init__(self, alpha: float, max_val: Optional[int] = None) -> None:
        self.alpha = alpha
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.zipf(self.alpha, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"ZipfDistribution[{self.alpha}]"


class PoissonDistribution(Distribution):
    def __init__(self, alpha: float, max_val: Optional[int] = None) -> None:
        self.alpha = alpha
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.poisson(self.alpha, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"PoissonDistribution[{self.alpha}]"


class LognormalDistribution(Distribution):
    def __init__(
        self, mean: float, sigma: float, max_val: Optional[int] = None
    ) -> None:
        self.mean = mean
        self.sigma = sigma
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.lognormal(mean=self.mean, sigma=self.sigma, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)

        return np.round(samples).astype(int)

    def __repr__(self) -> str:
        return f"LognormalDistribution[{self.mean}, {self.sigma}]"


class GenConvArgs(NamedTuple):
    num_conversations: int
    text_files: list[str]
    input_num_turns: Distribution
    input_common_prefix_num_tokens: Distribution
    input_prefix_num_tokens: Distribution
    input_num_tokens: Distribution
    output_num_tokens: Distribution
    print_stats: bool


def verify_field_exists(
    conf: dict, field_name: str, section: str, subsection: str
) -> None:
    if field_name not in conf:
        raise ValueError(
            f"Missing field '{field_name}' in {section=} and {subsection=}"
        )


def get_random_distribution(
    conf: dict, section: str, subsection: str, optional: bool = False
) -> Distribution:
    # section can be "prompt_input" or "prompt_output" (both required)
    conf = conf[section]

    if optional and subsection not in conf:
        # Optional subsection, if not found assume the value is always 0
        return ConstantDistribution(0)

    # subsection can be "num_turns", "num_tokens" or "prefix_num_tokens"
    if subsection not in conf:
        raise ValueError(f"Missing subsection {subsection} in section {section}")

    conf = conf[subsection]

    distribution = conf.get("distribution")
    if distribution is None:
        raise ValueError(
            f"Missing field 'distribution' in {section=} and {subsection=}"
        )

    if distribution == "constant":
        verify_field_exists(conf, "value", section, subsection)
        return ConstantDistribution(conf["value"])

    elif distribution == "zipf":
        verify_field_exists(conf, "alpha", section, subsection)
        max_val = conf.get("max", None)
        return ZipfDistribution(conf["alpha"], max_val=max_val)

    elif distribution == "poisson":
        verify_field_exists(conf, "alpha", section, subsection)
        max_val = conf.get("max", None)
        return PoissonDistribution(conf["alpha"], max_val=max_val)

    elif distribution == "lognormal":
        verify_field_exists(conf, "mean", section, subsection)
        verify_field_exists(conf, "sigma", section, subsection)
        max_val = conf.get("max", None)
        return LognormalDistribution(conf["mean"], conf["sigma"], max_val=max_val)

    elif distribution == "uniform":
        verify_field_exists(conf, "min", section, subsection)
        verify_field_exists(conf, "max", section, subsection)

        min_value = conf["min"]
        max_value = conf["max"]

        assert min_value > 0
        assert min_value <= max_value

        is_integer = isinstance(min_value, int) and isinstance(max_value, int)
        return UniformDistribution(min_value, max_value, is_integer)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def parse_input_json_file(conf: dict) -> GenConvArgs:
    # Validate the input file
    assert isinstance(conf, dict)
    required_fields = [
        "filetype",
        "num_conversations",
        "text_files",
        "prompt_input",
        "prompt_output",
    ]
    for field in required_fields:
        assert field in conf, f"Missing field {field} in input {conf}"

    assert conf["filetype"] == "generate_conversations"

    assert conf["num_conversations"] > 0, "num_conversations should be larger than zero"

    text_files = conf["text_files"]

    assert isinstance(text_files, list), "Field 'text_files' should be a list"
    assert len(text_files) > 0, (
        "Field 'text_files' should be a list with at least one file"
    )

    # Parse the parameters for the prompt input/output workload
    input_num_turns = get_random_distribution(conf, "prompt_input", "num_turns")
    input_num_tokens = get_random_distribution(conf, "prompt_input", "num_tokens")
    input_common_prefix_num_tokens = get_random_distribution(
        conf, "prompt_input", "common_prefix_num_tokens", optional=True
    )
    input_prefix_num_tokens = get_random_distribution(
        conf, "prompt_input", "prefix_num_tokens"
    )
    output_num_tokens = get_random_distribution(conf, "prompt_output", "num_tokens")

    print_stats: bool = conf.get("print_stats", False)
    assert isinstance(print_stats, bool), (
        "Field 'print_stats' should be either 'true' or 'false'"
    )

    args = GenConvArgs(
        num_conversations=conf["num_conversations"],
        text_files=text_files,
        input_num_turns=input_num_turns,
        input_common_prefix_num_tokens=input_common_prefix_num_tokens,
        input_prefix_num_tokens=input_prefix_num_tokens,
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        print_stats=print_stats,
    )
    return args


def print_conv_stats(conversations: ConversationsMap, tokenizer: AutoTokenizer) -> None:
    # Collect statistics
    conv_stats: list[dict[Any, Any]] = []
    req_stats: list[int] = []

    print("\nCollecting statistics...")
    for messages in conversations.values():
        # messages is a list of dicts
        user_tokens: list[int] = []
        assistant_tokens: list[int] = []
        request_tokens: list[int] = []

        req_tokens = 0
        for m in messages:
            content = m["content"]
            num_tokens = len(tokenizer(content).input_ids)

            if m["role"] == "user":
                user_tokens.append(num_tokens)
                # New user prompt including all chat history
                req_tokens += num_tokens
                request_tokens.append(req_tokens)

            elif m["role"] == "assistant":
                assistant_tokens.append(num_tokens)
                # Update assistant answer
                # (will be part of chat history for the next user prompt)
                req_tokens += num_tokens

        item_stats = {
            "conversation_turns": len(messages),
            "user_tokens": mean(user_tokens),
            "assistant_tokens": mean(assistant_tokens),
        }

        conv_stats.append(item_stats)
        req_stats.extend(request_tokens)

    # Print statistics
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Conversations statistics:{Color.RESET}")
    print(TEXT_SEPARATOR)
    df = pd.DataFrame(conv_stats)
    print(df.describe(percentiles=percentiles).transpose())
    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Request statistics:{Color.RESET}")
    print(TEXT_SEPARATOR)
    df = pd.DataFrame(req_stats, columns=["request_tokens"])
    print(df.describe(percentiles=percentiles).transpose())
    print(TEXT_SEPARATOR)


def generate_conversations(
    args: GenConvArgs, tokenizer: AutoTokenizer
) -> ConversationsMap:
    # Text for all user prompts
    # (text from the input text files will be appended to this line)
    base_prompt_text = "Please rewrite the following text and add more content: "
    base_prompt_token_count = len(
        tokenizer.encode(base_prompt_text, add_special_tokens=False)
    )

    logger.info(f"{Color.PURPLE}Generating conversations...{Color.RESET}")
    logger.info(args)

    list_of_tokens = []

    for filename in args.text_files:
        # Load text file that will be used to generate prompts
        with open(filename) as file:
            data = file.read()
            tokens_in_file = tokenizer.encode(data, add_special_tokens=False)
            list_of_tokens.extend(tokens_in_file)

    conversations: ConversationsMap = {}
    conv_id = 0

    # Generate number of turns for every conversation
    turn_count: np.ndarray = args.input_num_turns.sample(args.num_conversations)

    # Turn count should be at least 2 (one user prompt and one assistant answer)
    turn_count = np.maximum(turn_count, 2)

    # Round up to an even number (every user prompt should have an answer)
    turn_count = turn_count + (turn_count % 2)

    # Generate number of prefix tokens for every conversation
    conv_prefix_tokens: np.ndarray = args.input_prefix_num_tokens.sample(
        args.num_conversations
    )

    # Used to reduce shared text between conversations
    # (jump/skip over text sections between conversations)
    base_offset = 0

    # Common prefix size for all conversations (only 1 sample required)
    common_prefix_text = ""
    common_prefix_tokens: int = args.input_common_prefix_num_tokens.sample(1)[0]
    if common_prefix_tokens > 0:
        # Using "." at the end to separate sentences
        common_prefix_text = (
            tokenizer.decode(list_of_tokens[: common_prefix_tokens - 2]) + "."
        )
        base_offset += common_prefix_tokens

    for conv_id in range(args.num_conversations):
        # Generate a single conversation
        messages: MessagesList = []

        nturns = turn_count[conv_id]

        # User prompt token count per turn (with lower limit)
        input_token_count: np.ndarray = args.input_num_tokens.sample(nturns)
        input_token_count = np.maximum(input_token_count, base_prompt_token_count)

        # Assistant answer token count per turn (with lower limit)
        output_token_count: np.ndarray = args.output_num_tokens.sample(nturns)
        output_token_count = np.maximum(output_token_count, 1)

        user_turn = True
        for turn_id in range(nturns):
            if user_turn:
                role = "user"
                num_tokens = input_token_count[turn_id]

                # Generate the user prompt,
                # use a unique prefix (the conv_id) for each conversation
                # (to avoid shared prefix between conversations)
                content = f"{conv_id} is a nice number... "

                if len(common_prefix_text) > 0 and turn_id == 0:
                    content = common_prefix_text + content

                # Update the number of tokens left for the content
                num_tokens -= len(tokenizer.encode(content, add_special_tokens=False))

                if turn_id == 0:
                    prefix_num_tokens = conv_prefix_tokens[conv_id]
                    if prefix_num_tokens > 0:
                        # Add prefix text (context) to the first turn
                        start_offset = base_offset
                        end_offset = start_offset + prefix_num_tokens
                        assert len(list_of_tokens) > end_offset, (
                            "Not enough input text to generate "
                            f"{prefix_num_tokens} tokens for the "
                            f"prefix text ({start_offset=}, {end_offset=})"
                        )

                        content += f"{conv_id}, " + tokenizer.decode(
                            list_of_tokens[start_offset:end_offset]
                        )
                        base_offset += prefix_num_tokens

                # Add the actual user prompt/question after the prefix text
                content += base_prompt_text
                num_tokens -= base_prompt_token_count

                if num_tokens > 0:
                    # Add text from the input file (to reach the desired token count)
                    start_offset = base_offset + turn_id * input_token_count.max()
                    end_offset = start_offset + num_tokens
                    assert len(list_of_tokens) > end_offset, (
                        f"Not enough input text to generate {num_tokens} tokens "
                        f"for the prompt ({start_offset=}, {end_offset=})"
                    )

                    # Convert tokens back to text
                    content += tokenizer.decode(list_of_tokens[start_offset:end_offset])
            else:
                role = "assistant"
                # This content will not be used as input to the LLM server
                # (actual answers will be used instead).
                # Content is only required to determine the min_tokens/max_tokens
                # (inputs to the LLM server).
                num_tokens = output_token_count[turn_id]
                assert len(list_of_tokens) > num_tokens, (
                    f"Not enough input text to generate {num_tokens} "
                    "tokens for assistant content"
                )
                content = tokenizer.decode(list_of_tokens[:num_tokens])

            # Append the user/assistant message to the list of messages
            messages.append({"role": role, "content": content})
            user_turn = not user_turn

        # Add the new conversation
        conversations[f"CONV_ID_{conv_id}"] = messages

        # Increase base offset for the next conversation
        base_offset += nturns

    if args.print_stats:
        print_conv_stats(conversations, tokenizer)

    return conversations


def conversations_list_to_dict(input_list: ShareGptConversations) -> ConversationsMap:
    conversations: ConversationsMap = {}

    for item in input_list:
        conv_id: str = item["id"]
        assert isinstance(conv_id, str)

        assert conv_id not in conversations, (
            f"Conversation ID {conv_id} found more than once in the input"
        )

        messages: MessagesList = item["messages"]
        assert isinstance(messages, list), (
            f"Conversation messages should be a list (ID: {conv_id})"
        )
        assert len(messages) > 0, f"Conversation with no messages (ID: {conv_id})"

        conversations[conv_id] = messages

    logger.info(f"Using {len(conversations)} unique conversations (IDs)")
    assert len(conversations) == len(input_list)

    # Print statistics about the selected conversations
    stats: list[dict[str, Any]] = []
    for conv_data in conversations.values():
        stats.append({"num_turns": len(conv_data)})

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Conversations statistics:{Color.RESET}")
    print(TEXT_SEPARATOR)
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    conv_stats = pd.DataFrame(stats).describe(percentiles=percentiles)
    print(conv_stats.transpose())
    print(TEXT_SEPARATOR)

    return conversations


def conversations_dict_to_list(input_dict: ConversationsMap) -> ShareGptConversations:
    output: ShareGptConversations = []
    for conv_id, conv_data in input_dict.items():
        new_item = {"id": conv_id, "messages": conv_data}
        output.append(new_item)

    return output
