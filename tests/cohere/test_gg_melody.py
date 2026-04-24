# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test guided generation via vLLM serve: Chat Completions and Responses.

- Schema mode: Chat uses ``response_format`` (``json_schema``); Responses uses
  ``structured_outputs`` with ``json`` (not ``text.format``).
- JSON object mode (no field schema): Chat uses ``response_format`` type
  ``json_object``; Responses uses ``structured_outputs`` with ``json_object``.
"""

import argparse
import asyncio
import json
import os
import sys

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
        '{"think_start_str": "<|START_THINKING|>", '
        '"think_end_str": "<|END_THINKING|>"}',
        "--structured-outputs-config",
        '{"backend": "xgrammar"}',
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


async def run_guided_generation_tests(args: argparse.Namespace) -> None:
    """Chat Completions (response_format) then Responses (structured_outputs)."""
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

            # --- Chat Completions: response_format json_schema ---
            chat_outputs: list[str | None] = []
            for i, p in enumerate(synth_prompts):
                try:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": p["prompt"]}],
                        temperature=0.6,
                        top_p=0.95,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response",
                                "schema": p["schema"],
                            },
                        },
                        max_tokens=32000,
                    )
                    choice = response.choices[0] if response.choices else None
                    text = (
                        (choice.message.content or "").strip()
                        if choice and choice.message
                        else None
                    )
                    chat_outputs.append(text or None)
                except Exception as e:
                    print(f"Chat request {i} failed: {e}")
                    chat_outputs.append(None)

            bad_json, bad_schema = validate_api_output(chat_outputs, schema_map)
            _assert_schema_checks("Chat Completions", bad_json, bad_schema)
            print(
                "✅ JSON schema validation via Chat Completions API passed "
                f"({args.mode.value})"
            )

            # --- Responses: structured_outputs.json (not text.format) ---
            responses_outputs: list[str | None] = []
            for i, p in enumerate(synth_prompts):
                response = await client.responses.create(
                    model=args.model,
                    input=[{"role": "user", "content": p["prompt"]}],
                    temperature=0.6,
                    top_p=0.95,
                    max_output_tokens=32000,
                    extra_body={
                        "structured_outputs": {"json": p["schema"]},
                    },
                )
                responses_outputs.append(_responses_output_text(response))

            bad_json_r, bad_schema_r = validate_api_output(
                responses_outputs, schema_map
            )
            _assert_schema_checks("Responses API", bad_json_r, bad_schema_r)
            print(
                "✅ JSON schema validation via Responses API (structured_outputs) "
                f"passed ({args.mode.value})"
            )

            # JSON object mode (no schema): Chat + Responses; same synth_prompts.
            chat_json_outputs: list[str | None] = []
            for i, p in enumerate(synth_prompts):
                try:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": p["prompt"]}],
                        temperature=0.6,
                        top_p=0.95,
                        response_format={"type": "json_object"},
                        max_tokens=32000,
                    )
                    choice = response.choices[0] if response.choices else None
                    text = (
                        (choice.message.content or "").strip()
                        if choice and choice.message
                        else None
                    )
                    chat_json_outputs.append(text or None)
                except Exception as e:
                    print(f"Chat json_object request {i} failed: {e}")
                    chat_json_outputs.append(None)

            bad_chat_jm = validate_json_object_mode_outputs(chat_json_outputs)
            _assert_json_object_mode_checks("Chat Completions json_object", bad_chat_jm)
            print(
                "✅ JSON object mode (no schema) via Chat Completions passed "
                f"({args.mode.value})"
            )

            responses_json_outputs: list[str | None] = []
            for i, p in enumerate(synth_prompts):
                try:
                    response = await client.responses.create(
                        model=args.model,
                        input=[{"role": "user", "content": p["prompt"]}],
                        temperature=0.6,
                        top_p=0.95,
                        max_output_tokens=32000,
                        extra_body={"structured_outputs": {"json_object": True}},
                    )
                    responses_json_outputs.append(_responses_output_text(response))
                except Exception as e:
                    print(f"Responses json_object request {i} failed: {e}")
                    responses_json_outputs.append(None)

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
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_guided_generation_tests(args))


if __name__ == "__main__":
    main()
