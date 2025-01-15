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
    return (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
            r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")


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
                    "maxLength": 10
                },
                "minItems": 3
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
                            "type": "number"
                        },
                        "position": {
                            "type": "string"
                        }
                    },
                    "required": ["company", "position"]
                }
            }
        },
        "required": ["name", "age", "skills", "work_history"]
    }


@pytest.fixture
def sample_complex_json_schema():
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100  # Numeric range
            },
            "grade": {
                "type": "string",
                "pattern": "^[A-D]$"  # Regex pattern
            },
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern":
                    "^[a-z]{1,10}$"  # Combining length and pattern restrictions
                }
            }
        },
        "required": ["score", "grade", "email", "tags"]
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
        'type': 'object'
    }


@pytest.fixture
def sample_guided_choice():
    return [
        "Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript",
        "Ruby", "Swift", "Kotlin"
    ]


@pytest.fixture
def sample_sql_statements():
    return ("""
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
""")
