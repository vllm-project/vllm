# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demonstrate compact code-edit programs with vLLM structured outputs.

This example uses vLLM's OpenAI-compatible structured outputs to make the model
emit a small JSON edit program instead of repeating full source files. A client
side renderer then validates and applies the edit program deterministically.
"""

import argparse
import asyncio
import os
from enum import Enum

import openai
from pydantic import BaseModel, Field, ValidationError


class EditOperation(str, Enum):
    prepend_many = "prepend_many"


class SharedPrependEdit(BaseModel):
    operation: EditOperation = Field(description="The shared edit operation to apply.")
    paths: list[str] = Field(
        min_length=1,
        description="Repository-relative paths that receive the same prefix.",
    )
    text: str = Field(
        min_length=1,
        description="The exact text to prepend to every listed file.",
    )


SOURCE_FILES = {
    "src/api/users.py": """def list_users(request):
    return repository.fetch_users(request.account_id)
""",
    "src/api/teams.py": """def list_teams(request):
    return repository.fetch_teams(request.account_id)
""",
    "src/jobs/sync.py": """def run_sync_job(account_id):
    queue.enqueue("sync", account_id=account_id)
""",
    "tests/test_routes.py": """def test_users_route(client):
    assert client.get("/users").status_code == 200
""",
}

EXPECTED_PREFIX = "# CHIMERA_EDIT_MARKER: add request tracing\n"

SYSTEM_PROMPT = """You are editing a code repository.

Return only a JSON edit program that matches the provided schema. Do not return
full files. Use the prepend_many operation when identical text should be added
to several files.
"""


def build_user_prompt() -> str:
    files = "\n\n".join(
        f"Path: {path}\n```python\n{content}```"
        for path, content in SOURCE_FILES.items()
    )
    return f"""Add this exact first line to every file:
{EXPECTED_PREFIX!r}

Emit the smallest valid edit program. Include every listed path exactly once.

Files:
{files}
"""


def validate_edit_program(program: SharedPrependEdit) -> None:
    expected_paths = set(SOURCE_FILES)
    actual_paths = set(program.paths)
    if len(program.paths) != len(actual_paths):
        raise ValueError("edit program contains duplicate paths")
    if actual_paths != expected_paths:
        raise ValueError(
            "edit program paths must match fixture paths exactly; "
            f"expected={sorted(expected_paths)!r}, actual={sorted(actual_paths)!r}"
        )
    if program.text != EXPECTED_PREFIX:
        raise ValueError(
            "edit program text does not match the requested shared prefix; "
            f"expected={EXPECTED_PREFIX!r}, actual={program.text!r}"
        )


def render_edit_program(program: SharedPrependEdit) -> dict[str, str]:
    validate_edit_program(program)
    return {path: program.text + SOURCE_FILES[path] for path in program.paths}


def estimate_token_count(text: str) -> int:
    # A rough, dependency-free estimate that is sufficient for example output.
    return max(1, len(text) // 4)


def print_report(program: SharedPrependEdit, rendered_files: dict[str, str]) -> None:
    program_json = program.model_dump_json(indent=2)
    rendered_payload = "\n\n".join(
        f"Path: {path}\n{content}" for path, content in rendered_files.items()
    )
    full_file_payload = "\n\n".join(
        f"Path: {path}\n{EXPECTED_PREFIX}{content}"
        for path, content in SOURCE_FILES.items()
    )

    print("\nChimera-style edit program:")
    print(program_json)
    print("\nRendered preview:")
    preview_path = next(iter(rendered_files))
    print(f"Path: {preview_path}\n{rendered_files[preview_path]}")
    print("\nToken estimate:")
    print(f"  Edit program: {estimate_token_count(program_json)} tokens")
    print(f"  Rendered files: {estimate_token_count(rendered_payload)} tokens")
    print(
        f"  Full-file model output baseline: "
        f"{estimate_token_count(full_file_payload)} tokens"
    )


async def call_vllm(args: argparse.Namespace) -> SharedPrependEdit:
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    model = args.model or (await client.models.list()).data[0].id

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt()},
        ],
        max_tokens=args.max_tokens,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "shared-prepend-edit",
                "schema": SharedPrependEdit.model_json_schema(),
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    try:
        return SharedPrependEdit.model_validate_json(content)
    except ValidationError:
        print("Raw model output:")
        print(content)
        raise


async def cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Chimera-style code edit program example with vLLM "
            "structured outputs."
        ),
    )
    _ = parser.add_argument(
        "--model",
        default=None,
        help="Model name to request. Defaults to the first model served by vLLM.",
    )
    _ = parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for the structured edit program.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the vLLM request and render the built-in expected program.",
    )
    args = parser.parse_args()

    if args.dry_run:
        program = SharedPrependEdit(
            operation=EditOperation.prepend_many,
            paths=list(SOURCE_FILES),
            text=EXPECTED_PREFIX,
        )
    else:
        program = await call_vllm(args)

    rendered_files = render_edit_program(program)
    print_report(program, rendered_files)


def main() -> None:
    asyncio.run(cli())


if __name__ == "__main__":
    main()
