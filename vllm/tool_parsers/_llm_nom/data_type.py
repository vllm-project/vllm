# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa

import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator, List, Optional, Set

import jsonschema
import jsonschema.exceptions
from typing_extensions import Self

# fmt: off
NULL_STRINGS = (
    "null", "Null", "NULL",
    # It's bad, especially when we need enum string `none` etc.
    # "none", "null", "nil",
    # "None", "Null", "Nil",
    # "NONE", "NULL", "NIL",
)
# fmt: on


@dataclass(slots=True)
class ConversionResult:
    value: Any
    confidence: int


class AtomDataType(int, Enum):
    none = auto()
    string = auto()
    integer = auto()
    float = auto()
    boolean = auto()
    # complex types:
    array = auto()
    object = auto()
    other = auto()

    @classmethod
    def from_string(cls, string: Optional[str]) -> Self:
        if string is None:
            return cls.string
        string = string.lower()
        if string == "none" or string == "null" or string == "nil":
            return cls.none
        elif string == "str" or string == "string" or string == "text":
            return cls.string
        elif string == "int" or string == "integer":
            return cls.integer
        elif string == "float" or string == "number":
            return cls.float
        elif string == "bool" or string == "boolean":
            return cls.boolean
        elif string == "array" or string == "list":
            return cls.array
        elif (
            string == "object"
            or string == "dict"
            or string == "dictionary"
            or string == "map"
        ):
            return cls.object
        else:
            return cls.other

    def is_complex_type(self) -> bool:
        return self == self.array or self == self.object or self == self.other

    @classmethod
    def from_example(cls, example: Any) -> Self:
        if example is None:
            return cls.none
        elif isinstance(example, str):
            return cls.string
        # NOTE: Order matters. True is instance of int.
        elif isinstance(example, bool):
            return cls.boolean
        elif isinstance(example, int):
            return cls.integer
        elif isinstance(example, float):
            return cls.float
        elif isinstance(example, list):
            return cls.array
        elif isinstance(example, dict):
            return cls.object
        else:
            return cls.other

    @classmethod
    def iter_candidates_from_schema(cls, input: Any) -> Iterator[Self]:
        if isinstance(input, dict):
            if "type" in input:
                type_value = input["type"]
                if isinstance(type_value, str):
                    yield cls.from_string(type_value)
                elif isinstance(type_value, list):
                    for each in type_value:
                        yield cls.from_string(each)
                else:
                    yield cls.other
            elif ("properties" in input and isinstance(input["properties"], dict)) or (
                "additionalProperties" in input
                and input["additionalProperties"] is not False
            ):
                yield cls.object
            elif "items" in input or "prefixItems" in input:
                yield cls.array
            else:
                nothing_found = True
                if "const" in input:
                    nothing_found = False
                    yield cls.from_example(input["const"])
                elif (
                    "enum" in input
                    and isinstance(input["enum"], list)
                    and input["enum"]
                ):
                    for each in input["enum"]:
                        nothing_found = False
                        yield cls.from_example(each)
                else:
                    for choice_field in ("anyOf", "oneOf", "allOf"):
                        if choice_field in input:
                            choices = input[choice_field]
                            if isinstance(choices, list):
                                for choice in choices:
                                    for each in cls.iter_candidates_from_schema(choice):
                                        nothing_found = False
                                        yield each
                if nothing_found:
                    yield cls.string
        else:
            yield cls.string

    def convert_with_confidence(self, string: str) -> Optional[ConversionResult]:
        if self == AtomDataType.none:
            stripped = string.strip()
            if stripped:
                if len(stripped) == 4 and stripped.lower() == "null":
                    return ConversionResult(value=None, confidence=10)
                return ConversionResult(value=None, confidence=1)
            else:
                return ConversionResult(value=None, confidence=3)
        elif self == AtomDataType.string:
            return ConversionResult(value=string, confidence=2)
        elif self == AtomDataType.integer:
            try:
                return ConversionResult(value=int(string), confidence=10)
            except ValueError:
                return None
        elif self == AtomDataType.float:
            try:
                value = float(string)
            except ValueError:
                return None
            # NOTE: Python is evil.
            # In Python, we have
            #     json.dumps(math.nan) -> "NaN"
            #     json.dumps(math.inf) -> "Infinity"
            #     json.dumps(-math.inf) -> "-Infinity"
            # However, in other languages, you cannot deserialize them back.
            # No perfect solution here.
            if math.isnan(value):
                value = "NaN"
            elif math.isinf(value):
                value = "Infinity" if value > 0 else "-Infinity"
            elif value.is_integer():
                value = int(value)
            return ConversionResult(value=value, confidence=10)
        elif self == AtomDataType.boolean:
            stripped = string.strip()
            if len(stripped) > 5:
                return ConversionResult(value=False, confidence=1)
            lower = stripped.lower()
            if lower == "true":
                return ConversionResult(value=True, confidence=10)
            elif lower == "false":
                return ConversionResult(value=False, confidence=10)
            elif lower == "yes" or lower == "on" or lower == "1":
                return ConversionResult(value=True, confidence=9)
            elif lower == "no" or lower == "off" or lower == "0":
                return ConversionResult(value=False, confidence=9)
            else:
                return ConversionResult(value=False, confidence=1)
        else:
            try:
                return ConversionResult(value=json.loads(string), confidence=10)
            except json.JSONDecodeError:
                return None


@dataclass(slots=True)
class FunctionCallParameterDataType:
    candidates: List[AtomDataType]
    schema: Any
    streaming: bool

    @classmethod
    def get_schema_of_parameter(cls, schema: Any, parameter_name: str) -> Self:
        """
        $ref etc. not supported.
        """
        if not isinstance(schema, dict):
            return cls(candidates=[AtomDataType.string], schema=None, streaming=True)
        property = None
        parameters = schema.get("parameters")
        if isinstance(parameters, dict):
            properties = parameters.get("properties")
            if isinstance(properties, dict):
                property = properties.get(parameter_name)
        return cls.from_property(property)

    @classmethod
    def from_property(cls, property: Any) -> Self:
        candidate_set: Set[AtomDataType] = set()
        candidates: List[AtomDataType] = []
        # Guaranteed to be non-empty.
        for each in AtomDataType.iter_candidates_from_schema(property):
            if each not in candidate_set:
                candidate_set.add(each)
                candidates.append(each)
        # The parameter is streaming if and only if there is only one candidate, string.
        # No matter how complicated the schema is, if it can only be a string,
        streaming = len(candidates) == 1 and candidates[0] == AtomDataType.string
        return cls(candidates=candidates, schema=property, streaming=streaming)

    def get_data_type_of_property(self, key: str) -> Optional[Self]:
        if isinstance(self.schema, dict):
            property = self.get_property(self.schema, key)
            if property:
                return self.from_property(property)
            property_choices = []
            for choice_field in ("anyOf", "oneOf", "allOf"):
                if choice_field in self.schema:
                    choices = self.schema[choice_field]
                    if isinstance(choices, list):
                        for choice in choices:
                            property = self.get_property(choice, key)
                            if property:
                                property_choices.append(property)
            if property_choices:
                return self.from_property(
                    {
                        "anyOf": property_choices,
                    }
                )

    @staticmethod
    def get_property(schema: Any, key: str) -> Any:
        if "properties" in schema:
            properties = schema["properties"]
            if isinstance(properties, dict):
                property = properties.get(key)
                if property:
                    return property
        if "additionalProperties" in schema:
            additional_properties = schema["additionalProperties"]
            if isinstance(additional_properties, dict):
                return additional_properties

    def get_data_type_of_item(self, *, index: int = 0) -> Optional[Self]:
        if isinstance(self.schema, dict):
            item = self.get_item(self.schema, index=index)
            if item:
                return self.from_property(item)
            item_choices = []
            for choice_field in ("anyOf", "oneOf", "allOf"):
                if choice_field in self.schema:
                    choices = self.schema[choice_field]
                    if isinstance(choices, list):
                        for choice in choices:
                            item = self.get_item(choice, index=index)
                            if item:
                                item_choices.append(item)
            if item_choices:
                return self.from_property(
                    {
                        "anyOf": item_choices,
                    }
                )

    @staticmethod
    def get_item(schema: Any, *, index: int = 0) -> Any:
        items = schema.get("items")
        if items:
            return items
        if "prefixItems" in schema:
            prefix_items = schema["prefixItems"]
            additional_items = schema.get("additionalItems")
            if isinstance(prefix_items, list):
                if index < len(prefix_items):
                    return prefix_items[index]
                elif additional_items:
                    return additional_items
                elif len(prefix_items):
                    return prefix_items[-1]

    def convert(self, string: str, *, always_nullable: bool = False) -> Any:
        if always_nullable:
            stripped = string.strip()
            if len(stripped) == 4 and stripped.lower() == "null":
                return None
        if not string:
            if AtomDataType.object in self.candidates:
                return {}
            elif AtomDataType.array in self.candidates:
                return []
            elif AtomDataType.none in self.candidates:
                return None
            else:
                return ""
        converted_list: List[ConversionResult] = []
        has_schema = bool(self.schema)
        for candidate in self.candidates:
            converted = candidate.convert_with_confidence(string)
            if converted is None:
                continue
            if has_schema:
                try:
                    jsonschema.validate(schema=self.schema, instance=converted.value)
                    # Ensure it beats the invalid ones.
                    converted.confidence += 10
                except jsonschema.exceptions.SchemaError:
                    # The schema is invalid, ignore it.
                    has_schema = False
                except Exception:
                    # Validation failed.
                    pass
            converted_list.append(converted)
        if len(converted_list) == 1:
            return converted_list[0].value
        elif len(converted_list):
            return max(converted_list, key=lambda x: x.confidence).value
        else:
            return string
