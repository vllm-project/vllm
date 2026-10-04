# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import pytest
from pydantic import BaseModel

from vllm.platforms import current_platform

SAMPLE_REGEX = (
    r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
    r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"
)

# Note: Ensure this only uses attributes compatible with xgrammar
SAMPLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "grade": {
            "type": "string",
            "pattern": "^[A-D]$",  # Regex pattern
        },
        "email": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
        },
        "work_history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "duration": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 100.0,  # Numeric range
                    },
                    "position": {"type": "string"},
                },
                "required": ["company", "duration", "position"],
                "additionalProperties": False,
            },
            "minItems": 0,
            "maxItems": 3,
        },
    },
    "required": ["name", "age", "skills", "grade", "email", "work_history"],
    "additionalProperties": False,
    "minProperties": 1,
    "maxProperties": 10,
}

# A schema unsupported by xgrammar
UNSUPPORTED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "multipleOf": 5,  # Numeric multiple
        },
        "tags": {
            "type": "array",
            "items": {"type": "string", "minLength": 10, "maxLength": 20},
        },
    },
    "required": ["score", "tags"],
    "additionalProperties": False,
    "patternProperties": {
        "^score$": {"type": "integer"},
    },
}

SAMPLE_STRUCTURED_OUTPUTS_CHOICES = [
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

SAMPLE_SQL_EBNF = """
root ::= select_statement
select_statement ::= "SELECT" column "from" table "where" condition
column ::= "col_1" | "col_2"
table ::= "table_1" | "table_2"
condition ::= column "=" number
number ::= "1" | "2"
"""

SAMPLE_SQL_LARK = """
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
"""

NGRAM_SPEC_CONFIG = {
    "model": "[ngram]",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 1,
}

EAGLE_SPEC_CONFIG = {
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,
}

PARAMS_MODELS_BACKENDS_TOKENIZER_MODE = [
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar", "auto", None),
    # FIXME: Since "auto" will use Mistral tokenizer and these backends do not support
    # it, we skip these tests for now.
    # ("mistralai/Ministral-8B-Instruct-2410", "guidance", "auto", None),
    # ("mistralai/Ministral-8B-Instruct-2410", "lm-format-enforcer", "auto", None),
    ("mistralai/Ministral-8B-Instruct-2410", "guidance", "hf", None),
    pytest.param(
        "mistralai/Ministral-8B-Instruct-2410",
        "lm-format-enforcer",
        "hf",
        None,
        marks=pytest.mark.skip(
            reason=(
                "Flaky: lm-format-enforcer intermittently returns"
                "incomplete JSON."
                "See https://github.com/noamgat/lm-format-enforcer/issues/169"
            )
        ),
    ),
    ("mistralai/Ministral-8B-Instruct-2410", "xgrammar", "mistral", None),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar", "auto", None),
    pytest.param(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "lm-format-enforcer",
        "auto",
        None,
        marks=pytest.mark.skip(
            reason=(
                "Flaky: lm-format-enforcer intermittently returns"
                "incomplete JSON."
                "See https://github.com/noamgat/lm-format-enforcer/issues/169"
            )
        ),
    ),
    # FIXME: This tests are flaky on CI thus disabled. Tracking in Issue #24402
    # ("mistralai/Ministral-8B-Instruct-2410", "outlines", "auto", None),
    # ("mistralai/Ministral-8B-Instruct-2410", "outlines", "mistral", None),
    # ("Qwen/Qwen2.5-1.5B-Instruct", "guidance", "auto"),
    ("mistralai/Ministral-8B-Instruct-2410", "outlines", "auto", NGRAM_SPEC_CONFIG),
    ("mistralai/Ministral-8B-Instruct-2410", "guidance", "hf", NGRAM_SPEC_CONFIG),
    ("Qwen/Qwen2.5-1.5B-Instruct", "xgrammar", "auto", NGRAM_SPEC_CONFIG),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "xgrammar", "auto", EAGLE_SPEC_CONFIG),
]

PARAMS_MODELS_TOKENIZER_MODE = [
    ("mistralai/Ministral-8B-Instruct-2410", "auto"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "auto"),
]

platform_args = {}
if current_platform.is_rocm():
    platform_args["async_scheduling"] = False


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType
