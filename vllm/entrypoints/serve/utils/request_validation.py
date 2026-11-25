# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection, Mapping
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import PydanticCustomError, core_schema

_COLLECTION_SCHEMAS = frozenset({"dict", "frozenset", "list", "set", "tuple"})
_FAIL_FAST_REF_SUFFIX = "_fail_fast"
_SDK_MODEL_PACKAGES = frozenset({"cohere", "openai", "openai_harmony"})
_MAX_SDK_VALIDATION_ENTRIES = 4096


def check_sdk_validation_budget(value: Any) -> Any:
    """Bound collection entries before a third-party validator walks them."""
    remaining = _MAX_SDK_VALIDATION_ENTRIES
    pending = [value]
    children: Collection[Any]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        else:
            continue

        remaining -= len(children)
        if remaining < 0:
            raise PydanticCustomError(
                "sdk_validation_budget",
                "SDK request objects may contain at most {limit} collection entries",
                {"limit": _MAX_SDK_VALIDATION_ENTRIES},
            )
        pending.extend(children)
    return value


def _is_sdk_model(schema: Mapping[str, Any]) -> bool:
    while schema["type"] in {"function-after", "function-before", "function-wrap"}:
        schema = schema["schema"]
    return (
        schema["type"] == "model"
        and schema["cls"].__module__.split(".", 1)[0] in _SDK_MODEL_PACKAGES
    )


class FailFastRequestMixin:
    """Stop validating each request collection after its first invalid item."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        definitions: dict[str, core_schema.CoreSchema] = {}

        def visit(value: Any) -> Any:
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, tuple):
                return tuple(visit(item) for item in value)
            if not isinstance(value, dict):
                return value

            schema_type = value.get("type")
            if schema_type == "definition-ref":
                ref = value["schema_ref"]
                if ref.endswith(_FAIL_FAST_REF_SUFFIX):
                    return value.copy()
                guarded_ref = ref + _FAIL_FAST_REF_SUFFIX
                if guarded_ref not in definitions:
                    try:
                        definition = handler.resolve_ref_schema(value)
                    except LookupError:
                        # A reference back to a model still being built.
                        return value.copy()
                    if _is_sdk_model(definition):
                        # Pydantic can reuse the model's compiled validator.
                        # Check the input before it enters that validator.
                        return core_schema.no_info_before_validator_function(
                            check_sdk_validation_budget, value.copy()
                        )
                    definitions[guarded_ref] = definition
                    guarded = visit(definition)
                    guarded["ref"] = guarded_ref
                    definitions[guarded_ref] = guarded
                return {**value, "schema_ref": guarded_ref}

            # A field mapping can itself contain fields named "type" or
            # "metadata". Only special-case keys on actual schema nodes.
            if not isinstance(schema_type, str):
                return {key: visit(item) for key, item in value.items()}
            schema = {
                key: item
                if key in {"metadata", "serialization", "config", "default", "expected"}
                else visit(item)
                for key, item in value.items()
            }
            if "metadata" in schema:
                schema["metadata"] = schema["metadata"].copy()
            if schema_type in _COLLECTION_SCHEMAS:
                schema["fail_fast"] = True
            return schema

        # Use distinct references so a guarded schema and its original can
        # coexist in a union or another enclosing request model.
        schema = visit(handler(source_type))
        return core_schema.definitions_schema(schema, list(definitions.values()))
