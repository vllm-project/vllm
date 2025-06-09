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
def sample_prompts_for_structured_output():
    return [
        # Test 1: Generate JSON output based on a provided schema
        ("Give an example JSON for an employee profile that fits this "
         "schema. Make the response as short as possible. Schema:"),
        # Test 2: Generate JSON object without a schema
        ("Generate a JSON object with curly braces for a person with "
         "name and age fields for John Smith who is 31 years old. "
         "Make the response as short as possible."),
        # Test 3: test a jsonschema incompatible with xgrammar
        "Give an example JSON object for a grade that fits this schema:",
        # Test 4: Generate SQL statement using EBNF grammar
        # Test 5: Generate SQL statement using Lark grammar
        # Test 6: Test invalid grammar input
        ("Generate a sql statement that selects col_1 from "
         "table_1 where it is equal to 1. Make the response as short as "
         "possible."),
        # Test 7: Generate text based on a regex pattern
        "Give an example IPv4 address with this regex:",
        # Test 8: Generate text based on a choices
        ("The best language for type-safe systems programming is "
         "(Make the response as short as possible)."),
        # Test 9: Generate structured output using a Pydantic model
        ("Generate a JSON with the brand, model and car_type of the most "
         "iconic car from the 90's. Make the response as short as "
         "possible."),
        # Test 10: Generate structured with minLength and maxLength
        ("Generate a description of a frog using 50 characters. "
         "Make the response as short as possible."),
        # Test 11: Generate structured output using structural_tag format
        """
You have access to the following function to retrieve the weather in a city:

    {
        "name": "get_weather",
        "parameters": {
            "city": {
                "param_type": "string",
                "description": "The city to get the weather for",
                "required": True
            }
        }
    }

If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name
              as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query

You are a helpful assistant.

Given the previous instructions, what is the weather in New York City? \
Make the response as short as possible.
""",
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
    return (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
            r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")


# Note: Ensure this only uses attributes compatible with xgrammar
@pytest.fixture
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string",
                }
            },
            "grade": {
                "type": "string",
                "pattern": "^[A-D]$"  # Regex pattern
            },
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string"
                        },
                        "duration": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 100.0,  # Numeric range
                        },
                        "position": {
                            "type": "string"
                        }
                    },
                    "required": ["company", "duration", "position"],
                    "additionalProperties": False
                },
                "minItems": 0,
                "maxItems": 3
            }
        },
        "required":
        ["name", "age", "skills", "grade", "email", "work_history"],
        "additionalProperties": False
    }


# A schema unsupported by xgrammar
@pytest.fixture
def unsupported_json_schema():
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "multipleOf": 5  # Numeric multiple
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": 20
                }
            }
        },
        "required": ["score", "tags"],
        "additionalProperties": False
    }


@pytest.fixture
def sample_definition_json_schema():
    return {
        '$defs': {
            'Step': {
                'properties': {
                    'explanation': {
                        'title': 'Explanation',
                        'type': 'string'
                    },
                    'output': {
                        'title': 'Output',
                        'type': 'string'
                    }
                },
                'required': ['explanation', 'output'],
                'title': 'Step',
                'type': 'object'
            }
        },
        'properties': {
            'steps': {
                'items': {
                    '$ref': '#/$defs/Step'
                },
                'title': 'Steps',
                'type': 'array'
            },
            'final_answer': {
                'title': 'Final Answer',
                'type': 'string'
            }
        },
        'required': ['steps', 'final_answer'],
        'title': 'MathReasoning',
        'type': 'object',
        "additionalProperties": False
    }


@pytest.fixture
def sample_guided_choice():
    return [
        "Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript",
        "Ruby", "Swift", "Kotlin"
    ]


@pytest.fixture
def sample_sql_ebnf():
    return """
root ::= select_statement
select_statement ::= "SELECT" column "from" table "where" condition
column ::= "col_1" | "col_2"
table ::= "table_1" | "table_2"
condition ::= column "=" number
number ::= "1" | "2"
"""


@pytest.fixture
def sample_sql_lark():
    return ("""
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
""")
