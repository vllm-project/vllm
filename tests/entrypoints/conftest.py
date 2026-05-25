# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="charent/self_cognition_Alice")


@pytest.fixture(scope="session")
def qwen3_meowing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="session")
def qwen3_woofing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


@pytest.fixture(scope="session")
def opt125_lora_files() -> str:
    """Download opt-125m LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="peft-internal-testing/opt-125m-dummy-lora")


# ---------------------------------------------------------------------------
# Shared test factories for entrypoint serving unit tests
# ---------------------------------------------------------------------------

_MODEL_NAME = "test-model"


@dataclass
class _MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = _MODEL_NAME
    tokenizer = _MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    encoder_config = None
    generation_config = "auto"
    skip_tokenizer_init = False
    is_encoder_decoder = False
    is_multimodal_model = False
    hf_config = SimpleNamespace(model_type="any")
    hf_text_config = SimpleNamespace(model_type="any")

    def get_diff_sampling_param(self):
        return {}


def _make_engine():
    engine = MagicMock()
    engine.model_config = _MockModelConfig()
    engine.renderer = None
    engine.input_processor = None

    async def _echo(*args, **kwargs):
        return []

    engine.generate = _echo
    return engine


def _make_models(engine):
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels

    return OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[
            BaseModelPath(name=_MODEL_NAME, model_path=_MODEL_NAME),
        ],
    )


def _make_render_mock(models):
    mr = MagicMock(spec_set=["model_registry"])
    mr.model_registry = models.registry
    return mr


def make_base(usage_policy=None, enable_force_include_usage=False):
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    engine = _make_engine()
    models = _make_models(engine)
    return OpenAIServing(
        engine_client=engine,
        models=models,
        request_logger=None,
        usage_policy=usage_policy,
        enable_force_include_usage=enable_force_include_usage,
    )


def make_chat(usage_policy=None, enable_force_include_usage=False):
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    engine = _make_engine()
    models = _make_models(engine)
    return OpenAIServingChat(
        engine_client=engine,
        models=models,
        response_role="assistant",
        openai_serving_render=_make_render_mock(models),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        usage_policy=usage_policy,
        enable_force_include_usage=enable_force_include_usage,
    )


def make_completion(usage_policy=None, enable_force_include_usage=False):
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    engine = _make_engine()
    models = _make_models(engine)
    return OpenAIServingCompletion(
        engine_client=engine,
        models=models,
        openai_serving_render=_make_render_mock(models),
        request_logger=None,
        usage_policy=usage_policy,
        enable_force_include_usage=enable_force_include_usage,
    )


def make_disagg(usage_policy=None, enable_force_include_usage=False):
    from vllm.entrypoints.serve.disagg.serving import ServingTokens

    engine = _make_engine()
    models = _make_models(engine)
    return ServingTokens(
        engine_client=engine,
        models=models,
        openai_serving_render=_make_render_mock(models),
        request_logger=None,
        usage_policy=usage_policy,
        enable_force_include_usage=enable_force_include_usage,
    )


def make_anthropic(usage_policy=None, enable_force_include_usage=False):
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    engine = _make_engine()
    models = _make_models(engine)
    return AnthropicServingMessages(
        engine_client=engine,
        models=models,
        response_role="assistant",
        openai_serving_render=_make_render_mock(models),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        usage_policy=usage_policy,
        enable_force_include_usage=enable_force_include_usage,
    )
