# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import Any

import pytest


@pytest.fixture
def vllm_runner_factory(vllm_runner) -> Iterator[Callable[..., Any]]:
    """Create context-managed runners that are closed after each test.

    Some entrypoint tests need a runner for the entire test body, or several
    live runners at once. Tracking them in one ``ExitStack`` keeps those tests
    concise while still guaranteeing the complete ``VllmRunner.__exit__``
    cleanup path when a test fails partway through.
    """
    runners: list[Any] = []
    defer_rocm_memory_wait = False
    with ExitStack() as stack:

        def settle_after_runners() -> None:
            # This callback is registered before the runner exits, so the
            # ExitStack invokes it last. Drop the runner wrappers before
            # waiting as their ``llm`` attributes have already been deleted.
            runners.clear()
            if defer_rocm_memory_wait:
                from tests.utils import wait_for_rocm_memory_to_settle

                wait_for_rocm_memory_to_settle()

        stack.callback(settle_after_runners)

        def create_runner(*args: Any, **kwargs: Any) -> Any:
            nonlocal defer_rocm_memory_wait
            if runners:
                # Every runner created by this fixture remains alive until
                # fixture teardown. Once runners coexist, no individual
                # runner can wait for baseline VRAM while the others are
                # still using it; defer one wait until all have exited.
                for active_runner in runners:
                    active_runner.wait_for_rocm_memory = False
                kwargs["wait_for_rocm_memory"] = False
                defer_rocm_memory_wait = True

            runner = vllm_runner(*args, **kwargs)
            defer_rocm_memory_wait |= not runner.wait_for_rocm_memory
            runners.append(runner)
            return stack.enter_context(runner)

        yield create_runner


@pytest.fixture
def sample_prompts():
    return [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]


@pytest.fixture
def sample_token_ids():
    return [
        [0],
        [0, 1],
        [0, 2, 1],
        [0, 3, 1, 2],
    ]


@pytest.fixture
def sample_regex():
    return (
        r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
        r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"
    )


@pytest.fixture
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string", "maxLength": 10},
                "minItems": 3,
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "duration": {"type": "number"},
                        "position": {"type": "string"},
                    },
                    "required": ["company", "position"],
                },
            },
        },
        "required": ["name", "age", "skills", "work_history"],
    }


@pytest.fixture
def sample_complex_json_schema():
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,  # Numeric range
            },
            "grade": {
                "type": "string",
                "pattern": "^[A-D]$",  # Regex pattern
            },
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    # Combining length and pattern restrictions
                    "pattern": "^[a-z]{1,10}$",
                },
            },
        },
        "required": ["score", "grade", "email", "tags"],
    }


@pytest.fixture
def sample_definition_json_schema():
    return {
        "$defs": {
            "Step": {
                "properties": {
                    "explanation": {"title": "Explanation", "type": "string"},
                    "output": {"title": "Output", "type": "string"},
                },
                "required": ["explanation", "output"],
                "title": "Step",
                "type": "object",
            }
        },
        "properties": {
            "steps": {
                "items": {"$ref": "#/$defs/Step"},
                "title": "Steps",
                "type": "array",
            },
            "final_answer": {"title": "Final Answer", "type": "string"},
        },
        "required": ["steps", "final_answer"],
        "title": "MathReasoning",
        "type": "object",
    }


@pytest.fixture
def sample_enum_json_schema():
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "pending"],  # Literal values using enum
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
            },
            "category": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["bug", "feature", "improvement"],
                    },
                    "severity": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],  # Enum can also contain numbers
                    },
                },
                "required": ["type", "severity"],
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["urgent", "blocked", "needs_review", "approved"],
                },
            },
        },
        "required": ["status", "priority", "category", "flags"],
    }


@pytest.fixture
def sample_structured_outputs_choices():
    return [
        "Python",
        "Java",
        "JavaScript",
        "C++",
        "C#",
        "PHP",
        "TypeScript",
        "Ruby",
        "Swift",
        "Kotlin",
    ]


@pytest.fixture
def sample_sql_statements():
    return """
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
"""


@pytest.fixture(scope="session")
def qwen3_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id="charent/self_cognition_Alice")


@pytest.fixture(scope="session")
def qwen3_meowing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="session")
def qwen3_woofing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


@pytest.fixture(scope="session")
def opt125_lora_files() -> str:
    """Download opt-125m LoRA files once per test session."""
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(
        repo_id="peft-internal-testing/opt-125m-dummy-lora"
    )
