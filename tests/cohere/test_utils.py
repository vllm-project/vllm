# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import random
from enum import Enum
from typing import Any

import numpy as np
import tiktoken
import xgrammar as xgr
from jsonschema import validate
from PIL import Image

from vllm import SamplingParams
from vllm.cohere.guided_decoding.cohere_constants import MODEL_TO_PREFIX_POSTFIX
from vllm.cohere.guided_decoding.convert_to_structural_tag_format import (  # noqa: E501
    convert_schema_to_structural_tags,
)
from vllm.cohere.guided_decoding.tool_grammar import (
    COMMAND_R_TOOLS_TAG,
)
from vllm.cohere.utils import get_text_model_name
from vllm.config.reasoning import ReasoningConfig
from vllm.reasoning.cohere_command_reasoning_parser import (
    COMMAND_A_TOOLS_TAG,
    MODEL_TO_TAG_STYLE,
)
from vllm.sampling_params import StructuredOutputsParams

CURRENT_DIR = os.getcwd()
INPUT_FILE = os.path.join(CURRENT_DIR, "long_prompt.txt")

JSON_SCHEMA = {
    "type": "object",
    "required": ["title", "authors", "key_findings"],
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "key_findings": {"type": "array", "items": {"type": "string"}},
    },
}

TEST_LENGTHS = [2000, 5000]


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def get_paper_text(target_length: int) -> str:
    try:
        with open(INPUT_FILE) as file:
            full_text = file.read()

        words = full_text.split()
        current_text = ""
        for word in words:
            if count_tokens(current_text) >= target_length:
                break
            current_text += word + " "
        return current_text
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}") from err


def get_long_context(input_length: int):
    input_text = get_paper_text(input_length)
    prompt = f"Generate a JSON summary of this research paper text: {input_text}"
    return prompt


def get_input_text():
    long_context_text = []
    for length in TEST_LENGTHS:
        input_text = get_long_context(length)
        long_context_text.append(input_text)
    return long_context_text, JSON_SCHEMA


def validate_output(engine_output, schema_list, model_architecture):
    invalid_json = []
    invalid_json_schema = []
    for request_id, output in enumerate(engine_output):
        print("output: ", output)
        tokens = MODEL_TO_PREFIX_POSTFIX[model_architecture]
        prefix_token = tokens[0]
        postfix_token = tokens[1]
        output = find_text_between(output, prefix_token, postfix_token)
        try:
            json_obj = json.loads(output)

        except Exception:
            invalid_json.append(request_id)

        if len(schema_list) != 0:
            schema = schema_list[request_id]
            try:
                validate(json_obj, schema)

            except Exception:
                print("output: ", output)
                invalid_json_schema.append(request_id)
    return invalid_json, invalid_json_schema


def _create_reasoning_config():
    """Create a ReasoningConfig instance with proper initialization."""
    return ReasoningConfig(
        reasoning_start_str="<|START_THINKING|>",
        reasoning_end_str="<|END_THINKING|>",
    )


def add_system_prompt(example_input):
    prompt = (
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble\nYou are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful outcomes.\n\nYour information cutoff date is June 2024.\n\nYou have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.\n\n# Default Preamble\nThe following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.\n- Your name is Command.\n- You are a large language model built by Cohere.\n- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.\n- If the input is ambiguous, ask clarifying follow-up questions.\n- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).\n- Use LaTeX to generate mathematical notation for complex equations.\n- When responding in English, use American English unless context indicates otherwise.\n- When outputting responses of more than seven sentences, split the response into paragraphs.\n- Prefer the active voice.\n- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.\n- Use gender-neutral pronouns for unspecified persons.\n- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.\n- Use the third person when asked to write a summary.\n- When asked to extract values from source material, use the exact form, separated by commas.\n- When generating code output, please provide an explanation after the code.\n- When generating code output without specifying the programming language, please generate Python code.\n- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        + example_input
        + "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    )
    return prompt


def add_reasoning_prompt(example_input, continue_thinking=False):
    prompt = (
        r"""<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble\nYou are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful outcomes.\n\nYour information cutoff date is June 2024.\n\nYou have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.\n\n## Reasoning\nStart your response by writing <|START_THINKING|>. Then slowly and carefully reason through the problem. If you notice that you've made a mistake, you can correct it. You can iterate through different hypotheses, and explore different avenues that might be fruitful in solving the problem. Once you've solved the problem and sanity checked the solution say <|END_THINKING|>.\nWhen you are ready to respond write <|START_RESPONSE|>. Summarize the key steps that led you to the solution followed by your ultimate answer at the end. Once you are done, end your response with <|END_RESPONSE|>.\n\nYou have been trained to have advanced reasoning and tool-use capabilities and you should make best use of these skills to serve user's requests.\n\n## Tool Use\nThink about how you can make best use of the provided tools to help with the task and come up with a high level plan that you will execute first.\n\n0. Start by writing <|START_THINKING|> followed by a detailed step by step plan of how you will solve the problem. For each step explain your thinking fully and give details of required tool calls (if needed). Unless specified otherwise, you write your plan in natural language. When you finish, close it out with <|END_THINKING|>.\n\nThen carry out your plan by repeatedly executing the following steps.\n1. Action: write <|START_ACTION|> followed by a list of JSON-formatted tool calls, with each one containing \"tool_name\" and \"parameters\" fields.\n    When there are multiple tool calls which are completely independent of each other (i.e. they can be executed in parallel), you should list them out all together in one step. When you finish, close it out with <|END_ACTION|>.\n2. Observation: you will then receive results of those tool calls in JSON format in the very next turn, wrapped around by <|START_TOOL_RESULT|> and <|END_TOOL_RESULT|>. Carefully observe those results and think about what to do next. Note that these results will be provided to you in a separate turn. NEVER hallucinate results.\n    Every tool call produces a list of results (when a tool call produces no result or a single result, it'll still get wrapped inside a list). Each result is clearly linked to its originating tool call via its \"tool_call_id\".\n3. Reflection: start the next turn by writing <|START_THINKING|> followed by what you've figured out so far, any changes you need to make to your plan, and what you will do next. When you finish, close it out with <|END_THINKING|>.\n\nYou can repeat the above 3 steps multiple times (could be 0 times too if no suitable tool calls are available or needed), until you decide it's time to finally respond to the user.\n\n4. Response: then break out of the loop and write <|START_RESPONSE|> followed by a piece of text which serves as a response to the user's last request. Use all previous tool calls and results to help you when formulating your response. When you finish, close it out with <|END_RESPONSE|>.\n\n## Grounding\nImportantly, note that \"Reflection\" and \"Response\" above can be grounded.\nGrounding means you associate pieces of texts (called \"spans\") with those specific tool results that support them (called \"sources\"). And you use a pair of tags \"<co>\" and \"</co>\" to indicate when a span can be grounded onto a list of sources, listing them out in the closing tag. Sources from the same tool call are grouped together and listed as \"{tool_call_id}:[{list of result indices}]\", before they are joined together by \",\". E.g., \"<co>span</co: 0:[1,2],1:[0]>\" means that \"span\" is supported by result 1 and 2 from \"tool_call_id=0\" as well as result 0 from \"tool_call_id=1\".\n\n## Available Tools\nHere is the list of tools that you have available to you.\nYou can ONLY use the tools listed here. When a tool is not listed below, it is NOT available and you should NEVER attempt to use it.\nEach tool is represented as a JSON object with fields like \"name\", \"description\", \"parameters\" (per JSON Schema), and optionally, \"responses\" (per JSON Schema).\n\n```json\n[\n\n]\n```\n\n# Default Preamble\nThe following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.\n- Your name is Command.\n- You are a large language model built by Cohere.\n- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.\n- If the input is ambiguous, ask clarifying follow-up questions.\n- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).\n- Use LaTeX to generate mathematical notation for complex equations.\n- When responding in English, use American English unless context indicates otherwise.\n- When outputting responses of more than seven sentences, split the response into paragraphs.\n- Prefer the active voice.\n- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.\n- Use gender-neutral pronouns for unspecified persons.\n- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.\n- Use the third person when asked to write a summary.\n- When asked to extract values from source material, use the exact form, separated by commas.\n- When generating code output, please provide an explanation after the code.\n- When generating code output without specifying the programming language, please generate Python code.\n- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"""
        + example_input
        + r"""<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""
    )
    if continue_thinking:
        # prompt += r"""<|START_THINKING|>Okay, so I"""
        prompt += r"""<|START_THINKING|>"""
    return prompt


def generate_tool_grammar(response_schema):
    with open("tool_grammar_C3.txt", "a") as f:
        f.write(repr(response_schema))


def find_text_between(response, prefix, postfix):
    if prefix in response and postfix in response:
        # Both present — extract text between
        start = response.find(prefix) + len(prefix)
        end = response.find(postfix, start)
        return response[start:end].strip()
    elif prefix in response:
        # Only prefix present — remove it
        return response.replace(prefix, "").strip()
    elif postfix in response:
        # Only suffix present — remove it
        return response.replace(postfix, "").strip()
    else:
        # Neither present — return as is
        return response.strip()


def get_tool_schema(model_architecture, tool_schema):
    """Per-tool JSON Schema strings aligned with ``collect_tool_schema_v2``."""
    tool_style = MODEL_TO_TAG_STYLE[model_architecture].tools
    schema_list = []
    for tool in tool_schema:
        tool_name = tool["name"]
        tool_parameters = json.dumps(tool["parameters"])
        if tool_style == COMMAND_A_TOOLS_TAG:
            json_schema = f"""{{
                            "type": "object",
                            "properties": {{
                                "tool_call_id": {{
                                    "type": "string",
                                    "pattern": "^[0-9]+$"
                                }},
                                "tool_name": {{
                                    "type": "string",
                                    "const": "{tool_name}"
                                }},
                                "parameters": {tool_parameters}
                                }}
                                }}"""
        elif tool_style == COMMAND_R_TOOLS_TAG:
            json_schema = f"""{{
                            "type": "object",
                            "properties": {{
                                "tool_name": {{
                                    "type": "string",
                                    "const": "{tool_name}"
                                }},
                                "parameters": {tool_parameters}
                                }}
                                }}"""
        else:
            raise ValueError(
                f"Unsupported tool tag style for {model_architecture!r}: {tool_style!r}"
            )
        schema_list.append(json_schema)
    return schema_list


def generate_random_image(width, height, channels=3, method="numpy"):
    """
    Generate an image with random pixel values.

    Args:
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels
        channels (int): Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)
        method (str): Method to use ('numpy' or 'random')

    Returns:
        PIL.Image: Generated image with random pixels
    """

    if method == "numpy":
        # Using numpy for faster generation
        if channels == 1:
            # Grayscale
            pixel_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            image = Image.fromarray(pixel_data, mode="L")
        elif channels == 3:
            # RGB
            pixel_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(pixel_data, mode="RGB")
        elif channels == 4:
            # RGBA
            pixel_data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
            image = Image.fromarray(pixel_data, mode="RGBA")
        else:
            raise ValueError("channels must be 1, 3, or 4")

    elif method == "random":
        # Using Python's random module (slower but no numpy dependency)
        if channels == 1:
            # Grayscale
            gray_pixels = [random.randint(0, 255) for _ in range(width * height)]
            image = Image.new("L", (width, height))
            image.putdata(gray_pixels)
        elif channels == 3:
            # RGB
            rgb_pixels = [
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in range(width * height)
            ]
            image = Image.new("RGB", (width, height))
            image.putdata(rgb_pixels)
        elif channels == 4:
            # RGBA
            rgba_pixels = [
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                for _ in range(width * height)
            ]
            image = Image.new("RGBA", (width, height))
            image.putdata(rgba_pixels)
        else:
            raise ValueError("channels must be 1, 3, or 4")

    else:
        raise ValueError("method must be 'numpy' or 'random'")

    return image


def _build_prompt_string_with_chat_template(tokenizer, engine, prompt_text: str) -> str:
    """
    Build the rendered prompt string with the checkpoint Jinja chat template, matching
    ``test_guided_generation_vision_spec_async`` (``apply_chat_template(..., tokenize=False,
    add_generation_prompt=True)``). vLLM tokenizes this the same way as when
    ``multi_modal_data`` is present. Falls back to ``add_system_prompt`` if needed.
    """
    model_ref = getattr(engine.model_config, "model", None) or getattr(
        tokenizer, "name_or_path", None
    )
    if not model_ref:
        return add_system_prompt(prompt_text)

    trust = getattr(engine.model_config, "trust_remote_code", False)
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=trust)
    except Exception:
        return add_system_prompt(prompt_text)

    # Same user message shapes as test_guided_generation_vision_spec_async (list content, then string).
    message_variants = [
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        [{"role": "user", "content": prompt_text}],
    ]
    prompt_str: str | None = None
    for messages in message_variants:
        try:
            prompt_str = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            break
        except Exception:
            continue
    if not prompt_str:
        return add_system_prompt(prompt_text)
    return prompt_str


async def generate_guided_output(
    engine,
    prompt_text,
    request_id,
    response_schema,
    tokenizer,
    *,
    thinking_token_budget: int | None = None,
):
    """Generate output for one long-context prompt."""
    sampling_kwargs: dict[str, Any] = {"temperature": 0.3, "top_p": 0.75, "top_k": -1}
    if thinking_token_budget is not None:
        sampling_kwargs["thinking_token_budget"] = thinking_token_budget
    sampling = SamplingParams(**sampling_kwargs)

    model_arch = get_text_model_name(engine.model_config)
    # Enforce max tokens to be 8192 for Command A
    # this also ensure that we test for recursion depth
    # of 64000.
    sampling.max_tokens = (
        8192
        if model_arch == "Cohere2ForCausalLM"
        or model_arch == "Cohere2MoeForCausalLM"
        or model_arch == "Cohere2VisionForConditionalGeneration"
        else 4096
    )
    if response_schema:
        structural_tag = convert_schema_to_structural_tags(
            schema=response_schema, engine=engine
        )
        if structural_tag is not None:
            sampling.structured_outputs = StructuredOutputsParams(
                structural_tag=structural_tag, backend="xgrammar"
            )
        else:
            grammar = str(xgr.Grammar.from_json_schema(json.dumps(response_schema)))
            sampling.structured_outputs = StructuredOutputsParams(
                grammar=grammar, backend="xgrammar"
            )
    else:
        schema = {"type": "object"}
        structural_tag = convert_schema_to_structural_tags(schema=schema, engine=engine)
        if structural_tag is not None:
            sampling.structured_outputs = StructuredOutputsParams(
                structural_tag=structural_tag, backend="xgrammar"
            )
        else:
            sampling.structured_outputs = StructuredOutputsParams(
                json_object=True, backend="xgrammar"
            )

    prompt_str = _build_prompt_string_with_chat_template(tokenizer, engine, prompt_text)
    final_output = None
    async for out in engine.generate(
        {"prompt": prompt_str},
        sampling_params=sampling,
        request_id=str(request_id),
    ):
        final_output = out

    if not final_output or not final_output.outputs:
        return None

    out = final_output.outputs[0]
    return (
        tokenizer.decode(
            out.token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        if getattr(out, "token_ids", None)
        else out.text
    )


class RunMode(str, Enum):
    NON_SPECULATIVE = "non-speculative"
    SPECULATIVE = "speculative"
    BOTH = "both"


def make_speculative_config(args) -> dict[str, Any] | None:
    if args.mode == RunMode.NON_SPECULATIVE:
        return None
    if not args.draft_model:
        raise ValueError("--draft_model is required for speculative mode")
    config: dict[str, Any] = {
        "method": args.method,
        "model": args.draft_model,
        "num_speculative_tokens": args.num_spec_tokens,
        "draft_tensor_parallel_size": args.draft_tp,
        "max_model_len": args.max_model_len,
    }
    target_model = getattr(args, "model", None)
    if target_model:
        from vllm.cohere.auto_config import apply_profile_draft_attention_backend

        apply_profile_draft_attention_backend(
            config,
            target_model,
            trust_remote_code=True,
        )
    return config
