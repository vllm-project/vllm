# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility functions for structured output evaluation.

This module provides reusable utilities for validating and generating
structured outputs (JSON, grammar, regex, etc.) that can be used across
benchmarking and testing scripts.
"""

import json
import os
from typing import Any

import regex as re


def load_json_schema(schema_path: str | None = None) -> dict[str, Any]:
    """
    Load JSON schema from file or default location.

    Args:
        schema_path: Path to JSON schema file. If None, uses default schema.

    Returns:
        Dictionary containing the JSON schema.

    Raises:
        FileNotFoundError: If schema file doesn't exist.
        json.JSONDecodeError: If schema file contains invalid JSON.
    """
    if schema_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        schema_path = os.path.join(
            dir_path, "structured_schemas", "structured_schema_1.json"
        )

    with open(schema_path) as f:
        return json.load(f)


def generate_json_prompt(schema: dict[str, Any]) -> str:
    """
    Generate prompt for JSON structured output.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Prompt string requesting generation conforming to the schema.
    """
    schema_str = json.dumps(schema)
    return (
        f"Generate an example of a brief user profile "
        f"given the following schema: {schema_str}"
    )


def eval_json_response(response_text: str) -> bool:
    """
    Evaluate if a response is valid JSON.

    Extracts and validates JSON from response text. Handles cases where
    the JSON is embedded in other text by searching for JSON-like patterns.

    Args:
        response_text: The text response to validate.

    Returns:
        True if valid JSON is found, False otherwise.
    """
    try:
        # Use re.DOTALL to match across newlines without corrupting
        # internal whitespace in JSON string values
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match is None:
            return False
        json.loads(json_match.group())
        return True
    except (json.JSONDecodeError, AttributeError):
        return False


def validate_json_with_schema(
    text: str, schema: dict[str, Any] | None = None
) -> dict[str, bool]:
    """
    Validate JSON output and check required fields.

    Parses JSON once and derives all validation results from that
    single operation.

    Args:
        text: The text response to validate.
        schema: Optional JSON schema with 'required' field list.

    Returns:
        Dictionary with validation results:
            - valid_json: Whether output is valid JSON
            - has_required_fields: Whether all required fields are present
    """
    try:
        # Use re.DOTALL to match across newlines without corrupting
        # internal whitespace in JSON string values
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match is None:
            return {"valid_json": False, "has_required_fields": False}

        parsed = json.loads(json_match.group())

        # Check required fields if schema is provided
        has_required = True
        if schema and "required" in schema:
            has_required = all(field in parsed for field in schema["required"])

        return {"valid_json": True, "has_required_fields": has_required}
    except (json.JSONDecodeError, AttributeError):
        return {"valid_json": False, "has_required_fields": False}
