# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test guided generation via vLLM serve: Chat Completions and Responses.

- Schema mode: Chat uses ``response_format`` (``json_schema``); Responses uses
  ``structured_outputs`` with ``json`` (not ``text.format``).
- JSON object mode (no field schema): Chat uses ``response_format`` type
  ``json_object``; Responses uses ``structured_outputs`` with ``json_object``.

Within each phase, all ``synth_prompts`` requests are sent concurrently
(``asyncio.gather``) so the server schedules overlapping sequences like production.

CI (``run_guided_generation`` in ``run_tests.sh``) also runs this script
against the ``mhl_v2`` checkpoint, non-speculative, TP 4,
after syncing to ``gs://cohere-model-efficiency-ci/engines/``.

Use ``--debug-engine-request`` to log API sampling fields, the user prompt, and the
post-chat-template engine prompt (Chat: ``prompt_token_ids``; Responses: ``input_messages``).
"""

import argparse
import asyncio
import copy
import json
import os
import pprint
import sys
from typing import Any

# Add repo root so tests.utils and vllm are importable when run from tests/cohere
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Cohere test data and validation (same as engine-based test)
import openai  # noqa: E402
from jsonschema import validate as jsonschema_validate  # noqa: E402
from test_schema_data import synth_prompts  # noqa: E402
from test_utils import (  # noqa: E402
    RunMode,
    make_speculative_config,
)
from transformers import AutoTokenizer  # noqa: E402

from tests.utils import RemoteOpenAIServer  # noqa: E402

# Ad hoc schema for request 1 only: valid JSON Schema, unrelated to that prompt's
# canonical schema in schema_map, so guided output vs validator can diverge.


def validate_api_output(output_texts: list[str | None], schema_map: dict[int, dict]):
    """
    Validate output texts: each should be valid JSON and, when a schema exists,
    conform to the schema in schema_map (canonical prompts, not the guided schema).
    Returns (invalid_json, invalid_schema).
    """
    invalid_json = []
    invalid_schema = []
    for request_id, output in enumerate(output_texts):
        if output is None or not output.strip():
            invalid_json.append(request_id)
            continue
        print(f"Request {request_id} output: {output}")
        try:
            json_obj = json.loads(output)
        except Exception:
            invalid_json.append(request_id)
            continue
        if request_id in schema_map:
            try:
                jsonschema_validate(json_obj, schema_map[request_id])
            except Exception:
                invalid_schema.append(request_id)
    return invalid_json, invalid_schema


def _responses_output_text(response: object) -> str | None:
    """Best-effort assistant text from a Responses API result."""
    out_text = getattr(response, "output_text", None)
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()
    output = getattr(response, "output", None) or []
    for item in reversed(output):
        if getattr(item, "type", None) != "message":
            continue
        chunks: list[str] = []
        for block in getattr(item, "content", None) or []:
            t = getattr(block, "text", None)
            if t:
                chunks.append(t)
        if chunks:
            joined = "".join(chunks).strip()
            return joined or None
    return None


def _server_args(args: argparse.Namespace) -> list[str]:
    server_args = [
        "--tensor-parallel-size",
        "4",
        "--enable-prefix-caching",
        "--max-num-seqs",
        "32",
        "--quantization",
        "compressed-tensors",
        "--async-scheduling",
        "--gpu_memory_utilization",
        "0.9",
        "--reasoning-config",
        '{"reasoning_start_str": "<|START_THINKING|>", '
        '"reasoning_end_str": "<|END_THINKING|>"}',
        "--structured-outputs-config",
        '{"backend": "xgrammar", "enable_in_reasoning":true}',
        "--reasoning-parser",
        "cohere_command4",
    ]
    if args.tensor_parallel_size is not None:
        server_args.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    speculative_config = make_speculative_config(args)
    if speculative_config is not None:
        server_args.extend(["--speculative_config", json.dumps(speculative_config)])
    return server_args


def _assert_schema_checks(phase: str, bad_json: list, bad_schema: list) -> None:
    assert len(bad_json) <= 5, f"{phase}: too many invalid JSON objects: {bad_json}"
    assert len(bad_schema) <= 5, f"{phase}: too many invalid schema JSON: {bad_schema}"


def validate_json_object_mode_outputs(output_texts: list[str | None]) -> list[int]:
    """
    Indices where output is missing, non-JSON, or not a top-level JSON object.
    Does not check a field schema (JSON object mode only).
    """
    bad: list[int] = []
    for i, output in enumerate(output_texts):
        if output is None or not output.strip():
            bad.append(i)
            continue
        try:
            obj = json.loads(output.strip())
        except Exception:
            bad.append(i)
            continue
        if not isinstance(obj, dict):
            bad.append(i)
    return bad


def _assert_json_object_mode_checks(phase: str, bad: list[int]) -> None:
    assert len(bad) <= 5, f"{phase}: too many invalid JSON objects: {bad}"


# Chat / Responses generation knobs (must match the values passed to the APIs below).
CHAT_TEMPERATURE = 0.3
CHAT_TOP_P = 0.75
CHAT_MAX_TOKENS = 32000
RESPONSES_MAX_OUTPUT_TOKENS = 32000


def _response_format_for_log(response_format: dict | None) -> dict | str | None:
    if response_format is None:
        return None
    rf = copy.deepcopy(response_format)
    if rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict):
        js = rf["json_schema"]
        rf["json_schema"] = {
            "name": js.get("name"),
            "schema": f"<{len(json.dumps(js.get('schema', {})))} chars>",
        }
    return rf


def _print_chat_engine_debug(
    phase: str,
    request_id: int,
    user_text: str,
    response_json: dict,
    tokenizer: AutoTokenizer,
    request_kwargs: dict,
) -> None:
    print(f"\n{'=' * 16} ENGINE DEBUG {phase} (request {request_id}) {'=' * 16}")
    print("--- user text (single user message ``content``; before chat template) ---")
    print(user_text)
    print(
        "\n--- OpenAI Chat Completions request fields "
        "(what this script sends → vLLM; engine merges defaults + structured output) ---"
    )
    log_kw = {
        k: v
        for k, v in request_kwargs.items()
        if k not in ("extra_body", "response_format")
    }
    log_kw["response_format"] = _response_format_for_log(
        request_kwargs.get("response_format")
    )
    log_kw["extra_body"] = request_kwargs.get("extra_body")
    pprint.pprint(log_kw, width=120, sort_dicts=False)
    pt = response_json.get("prompt_token_ids")
    if pt:
        rendered = tokenizer.decode(
            pt,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        print(
            "\n--- engine prompt string "
            "(``decode(response['prompt_token_ids'])``; after chat template, "
            "what the model conditions on) ---"
        )
        print(rendered)
    else:
        print(
            "\n(no ``prompt_token_ids`` in response; use ``--debug-engine-request`` "
            "which sets ``return_token_ids: true``)"
        )


def _print_responses_engine_debug(
    phase: str,
    request_id: int,
    user_text: str,
    response_json: dict,
    request_kwargs: dict,
) -> None:
    print(f"\n{'=' * 16} ENGINE DEBUG {phase} (request {request_id}) {'=' * 16}")
    print("--- user text (input user ``content``; before Responses render) ---")
    print(user_text)
    print(
        "\n--- OpenAI Responses ``create`` kwargs "
        "(script → vLLM; engine adds guided decoding sampling fields server-side) ---"
    )
    eb = request_kwargs.get("extra_body") or {}
    log_kw = {k: v for k, v in request_kwargs.items() if k != "extra_body"}
    log_kw["extra_body"] = {
        k: (f"<{len(json.dumps(v))} chars>" if k == "structured_outputs" else v)
        for k, v in eb.items()
    }
    pprint.pprint(log_kw, width=120, sort_dicts=False)
    in_msgs = response_json.get("input_messages")
    if in_msgs:
        print(
            "\n--- ``input_messages`` from server "
            "(``enable_response_messages``; includes tokenized / templated input) ---"
        )
        pprint.pprint(in_msgs, width=120)
    else:
        print(
            "\n(no ``input_messages`` in response; ensure ``enable_response_messages`` "
            "is true in the request — set automatically with ``--debug-engine-request``)"
        )


async def _chat_completion_create(
    client: openai.AsyncOpenAI,
    args: argparse.Namespace,
    *,
    phase: str,
    request_id: int,
    user_prompt: str,
    messages: list[dict],
    response_format: dict | None,
    debug_tokenizer: AutoTokenizer | None,
) -> Any:
    kwargs: dict = {
        "model": args.model,
        "messages": messages,
        "temperature": CHAT_TEMPERATURE,
        "top_p": CHAT_TOP_P,
        "max_tokens": CHAT_MAX_TOKENS,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if not getattr(args, "debug_engine_request", False):
        return await client.chat.completions.create(**kwargs)
    assert debug_tokenizer is not None
    kwargs["extra_body"] = {"return_token_ids": True}
    raw = await client.with_raw_response.chat.completions.create(**kwargs)
    body = json.loads(raw.http_response.text)
    _print_chat_engine_debug(
        phase, request_id, user_prompt, body, debug_tokenizer, kwargs
    )
    return raw.parse()


async def _responses_create(
    client: openai.AsyncOpenAI,
    args: argparse.Namespace,
    *,
    phase: str,
    request_id: int,
    user_prompt: str,
    input_items: list[dict],
    extra_body: dict,
) -> Any:
    kwargs: dict = {
        "model": args.model,
        "input": input_items,
        "temperature": CHAT_TEMPERATURE,
        "top_p": CHAT_TOP_P,
        "max_output_tokens": RESPONSES_MAX_OUTPUT_TOKENS,
        "extra_body": dict(extra_body),
    }
    if not getattr(args, "debug_engine_request", False):
        return await client.responses.create(**kwargs)
    kwargs["extra_body"]["enable_response_messages"] = True
    raw = await client.with_raw_response.responses.create(**kwargs)
    body = json.loads(raw.http_response.text)
    _print_responses_engine_debug(phase, request_id, user_prompt, body, kwargs)
    return raw.parse()


async def run_guided_generation_tests(args: argparse.Namespace) -> None:
    """Chat Completions (response_format) then Responses (structured_outputs).

    Each phase issues all ``synth_prompts`` requests concurrently (``asyncio.gather``)
    so the engine sees overlapping work like production, not strictly sequential awaits.
    """
    server_args = _server_args(args)
    with RemoteOpenAIServer(
        args.model,
        server_args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1"},
        max_wait_seconds=getattr(args, "server_wait_seconds", 300),
    ) as server:
        base_url = f"http://{server.host}:{server.port}/v1"
        async with openai.AsyncOpenAI(
            base_url=base_url,
            api_key=RemoteOpenAIServer.DUMMY_API_KEY,
        ) as client:
            schema_map = {i: p["schema"] for i, p in enumerate(synth_prompts)}
            debug_tokenizer: AutoTokenizer | None = None
            if getattr(args, "debug_engine_request", False):
                debug_tokenizer = AutoTokenizer.from_pretrained(
                    args.model, use_fast=True
                )

            async def chat_json_schema_one(i: int, p: dict) -> str | None:
                try:
                    response = await _chat_completion_create(
                        client,
                        args,
                        phase="Chat Completions json_schema",
                        request_id=i,
                        user_prompt=p["prompt"],
                        messages=[{"role": "user", "content": p["prompt"]}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response",
                                "schema": p["schema"],
                            },
                        },
                        debug_tokenizer=debug_tokenizer,
                    )
                    choice = response.choices[0] if response.choices else None
                    text = (
                        (choice.message.content or "").strip()
                        if choice and choice.message
                        else None
                    )
                    return text or None
                except Exception as e:
                    print(f"Chat request {i} failed: {e}")
                    return None

            chat_outputs: list[str | None] = list(
                await asyncio.gather(
                    *[chat_json_schema_one(i, p) for i, p in enumerate(synth_prompts)]
                )
            )

            bad_json, bad_schema = validate_api_output(chat_outputs, schema_map)
            _assert_schema_checks("Chat Completions", bad_json, bad_schema)
            print(
                "✅ JSON schema validation via Chat Completions API passed "
                f"({args.mode.value})"
            )

            # --- Responses: structured_outputs.json (concurrent) ---
            async def responses_json_schema_one(i: int, p: dict) -> str | None:
                try:
                    response = await _responses_create(
                        client,
                        args,
                        phase="Responses structured_outputs json",
                        request_id=i,
                        user_prompt=p["prompt"],
                        input_items=[{"role": "user", "content": p["prompt"]}],
                        extra_body={"structured_outputs": {"json": p["schema"]}},
                    )
                    return _responses_output_text(response)
                except Exception as e:
                    print(f"Responses request {i} failed: {e}")
                    return None

            responses_outputs: list[str | None] = list(
                await asyncio.gather(
                    *[
                        responses_json_schema_one(i, p)
                        for i, p in enumerate(synth_prompts)
                    ]
                )
            )

            bad_json_r, bad_schema_r = validate_api_output(
                responses_outputs, schema_map
            )
            _assert_schema_checks("Responses API", bad_json_r, bad_schema_r)
            print(
                "✅ JSON schema validation via Responses API (structured_outputs) "
                f"passed ({args.mode.value})"
            )

            # JSON object mode (no schema): Chat + Responses; same synth_prompts.
            async def chat_json_object_one(i: int, p: dict) -> str | None:
                try:
                    response = await _chat_completion_create(
                        client,
                        args,
                        phase="Chat Completions json_object",
                        request_id=i,
                        user_prompt=p["prompt"],
                        messages=[{"role": "user", "content": p["prompt"]}],
                        response_format={"type": "json_object"},
                        debug_tokenizer=debug_tokenizer,
                    )
                    choice = response.choices[0] if response.choices else None
                    text = (
                        (choice.message.content or "").strip()
                        if choice and choice.message
                        else None
                    )
                    return text or None
                except Exception as e:
                    print(f"Chat json_object request {i} failed: {e}")
                    return None

            chat_json_outputs: list[str | None] = list(
                await asyncio.gather(
                    *[chat_json_object_one(i, p) for i, p in enumerate(synth_prompts)]
                )
            )

            bad_chat_jm = validate_json_object_mode_outputs(chat_json_outputs)
            _assert_json_object_mode_checks("Chat Completions json_object", bad_chat_jm)
            print(
                "✅ JSON object mode (no schema) via Chat Completions passed "
                f"({args.mode.value})"
            )

            async def responses_json_object_one(i: int, p: dict) -> str | None:
                try:
                    response = await _responses_create(
                        client,
                        args,
                        phase="Responses structured_outputs json_object",
                        request_id=i,
                        user_prompt=p["prompt"],
                        input_items=[{"role": "user", "content": p["prompt"]}],
                        extra_body={"structured_outputs": {"json_object": True}},
                    )
                    return _responses_output_text(response)
                except Exception as e:
                    print(f"Responses json_object request {i} failed: {e}")
                    return None

            responses_json_outputs: list[str | None] = list(
                await asyncio.gather(
                    *[
                        responses_json_object_one(i, p)
                        for i, p in enumerate(synth_prompts)
                    ]
                )
            )

            bad_resp_jm = validate_json_object_mode_outputs(responses_json_outputs)
            _assert_json_object_mode_checks(
                "Responses structured_outputs json_object", bad_resp_jm
            )
            print(
                "✅ JSON object mode (no schema) via Responses passed "
                f"({args.mode.value})"
            )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Guided JSON schema via vLLM serve: Chat Completions and Responses API"
        )
    )
    p.add_argument("--model", type=str, default="CohereForAI/c4ai-command-r-v01")
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument(
        "--mode", type=RunMode, choices=list(RunMode), default=RunMode.NON_SPECULATIVE
    )
    p.add_argument("--method", type=str, default="eagle")
    p.add_argument("--draft_model", type=str, default=None)
    p.add_argument("--num_spec_tokens", type=int, default=4)
    p.add_argument("--draft_tp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32000)
    p.add_argument("--server_wait_seconds", type=float, default=4000)
    p.add_argument(
        "--debug-engine-request",
        action="store_true",
        help=(
            "Per request: print sampling fields this script sends, the user text, "
            "and how the engine sees the prompt after the chat template "
            "(Chat: decode ``prompt_token_ids`` via ``return_token_ids``; "
            "Responses: ``input_messages`` via ``enable_response_messages``)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_guided_generation_tests(args))


if __name__ == "__main__":
    main()
