# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_COLLECTION_SCHEMAS = frozenset({"dict", "frozenset", "list", "set", "tuple"})
_FAIL_FAST_REF_SUFFIX = "_fail_fast"


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
