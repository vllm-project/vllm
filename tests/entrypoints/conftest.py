# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture
def sample_prompts():
    return [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
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
