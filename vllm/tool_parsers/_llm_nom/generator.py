# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa

import functools
import json
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from typing_extensions import Self

from .data_type import NULL_STRINGS, FunctionCallParameterDataType
from .base import Parser, ParserOutput, PatternMismatched
from .decoder import CountConsumedTokensFn

OutputKey = Union[str, Callable[[Any], Dict]]


class GeneratorParser(Parser[str, dict], ABC):
    """
    TODO:
    1. 改成接 StrOrSpecialToken.
    2. 报错时支持 row:col.
    3. 需要 self.generator.close() 吗?
    """

    @abstractmethod
    def _process(self) -> Generator[ParserOutput, None, None]:
        pass

    def __init__(self):
        self._count_consumed_tokens_fn: Optional[CountConsumedTokensFn] = None
        self._full_text: List[str] = []
        self._input_length = 0
        self._input_buffer: Deque[str] = deque()
        self._delta: Optional[Dict[str, Any]] = None
        self._final: Optional[Dict[str, Any]] = None
        self._generator = self._process()
        next(self._generator)

    def __del__(self):
        # Necessary when GC is disabled (gc.disable()),
        # otherwise the cyclic reference will never be collected.
        if hasattr(self, "_generator"):
            self._generator.close()
            del self._generator

    def reset(self):
        self._full_text.clear()
        self._input_length = 0
        self._input_buffer.clear()
        self._delta = None
        self._final = None
        # don't worry, a closed generator can be closed again.
        self._generator.close()
        self._generator = self._process()
        next(self._generator)

    def update(self, input: str):
        self._full_text.append(input)
        total = len(input)
        self._input_length += total
        self._input_buffer.extend(input)
        # at most #input times,
        # but early stop if the input buffer is already empty.
        for _ in range(total):
            if not self._input_buffer:
                break
            try:
                next(self._generator)
            except StopIteration:
                if self._input_buffer:
                    raise self._error(
                        expected=None,
                        actual=self._input_buffer[0],
                        reason="pattern exhausted",
                    )

    def _count_consumed_tokens(self) -> Optional[int]:
        if self._count_consumed_tokens_fn is None:
            return
        total_tokens_count = self._count_consumed_tokens_fn(None)
        if self._input_buffer:
            pending_text_count = len(self._input_buffer)
            consumed_text_count = self._input_length - pending_text_count
            # If total is 0, we cannot use `consumed = total - pending`.
            # Have to tokenize the consumed text, maybe again.
            is_pending = (total_tokens_count > 0) and (
                pending_text_count <= consumed_text_count
            )
            if is_pending:
                text = "".join(self._input_buffer)
            else:
                text = "".join(self._full_text)[:consumed_text_count]
            return self._count_consumed_tokens_fn((is_pending, text))
        elif total_tokens_count:
            return total_tokens_count
        else:
            # Only for non-streaming mode, should be called only once.
            # In streaming mode, we should feed in tokens instead of text,
            # thus `total_tokens_count` must be positive.
            text = "".join(self._full_text)
            return self._count_consumed_tokens_fn((False, text))

    def get_delta(self) -> Optional[Dict[str, Any]]:
        if self._delta:
            delta = _get_dict_with_joint_strings(self._delta)
            self._delta = None
            return delta

    def get_final(self) -> Optional[Dict[str, Any]]:
        return _get_dict_with_joint_strings(self._final)

    def _peek(self, index: int) -> Generator[None, None, str]:
        while index >= len(self._input_buffer):
            yield
        return self._input_buffer[index]

    def _read_one(self) -> str:
        # intentionally unchecked, will raise IndexError if the input buffer is empty.
        return self._input_buffer.popleft()

    def _consume(self, length: int):
        for _ in range(length):
            self._input_buffer.popleft()

    def _try(self, fn, *args, **kwargs):
        try:
            result = yield from fn(*args, **kwargs)
            return Tried(successful=True, result=result)
        except PatternMismatched as e:
            # Clear traceback to break traceback → frame → f_locals → self
            # circular reference chain. The traceback is not needed for
            # backtracking logic, only offset/expected/actual matter.
            e.__traceback__ = None
            return Tried(successful=False, error=e)

    def _append(self, key: OutputKey, value: Any):
        if isinstance(key, str):
            delta = {key: value}
        else:
            delta = key(value)
        self._append_delta(delta)

    def _append_delta(self, delta: Dict[str, Any]):
        self._delta = _merge_dicts(delta, self._delta)
        self._final = _merge_dicts(delta, self._final)

    def _get_context(self, offset: int) -> str:
        # NOTE: `self._error` is frequently triggered with backtracking,
        # thus we cannot afford to join the full text every time.
        # A partial function should work, which is only called when the error is stringified.
        full_text = "".join(self._full_text)
        return f"{full_text[:offset]}💥{full_text[offset:]}"

    def _error(self, *, expected: Any, actual: Any, reason: str) -> PatternMismatched:
        offset = self._input_length - len(self._input_buffer)
        # Use weakref to avoid context_fn → partial → self circular reference.
        # _full_text is a plain list, capturing it directly avoids holding self.
        full_text_ref = self._full_text
        return PatternMismatched(
            offset=offset,
            expected=expected,
            actual=actual,
            reason=reason,
            context_fn=functools.partial(
                _get_context_standalone, full_text_ref, offset
            ),
        )

    def _take_any(
        self,
        *,
        until: Optional[Union[str, Tuple[str, ...]]] = None,
        key: Optional[OutputKey] = None,
        should_consume_suffix: bool = True,
    ) -> Generator[str, None, None]:
        if until is None:
            # Never ends without `until`, thus no need to collect the values.
            while True:
                yield from self._peek(0)
                self._append(key, self._read_one())
        else:
            values = []
            while True:
                tried = yield from self._literal(
                    until,
                    should_consume=should_consume_suffix,
                    should_raise=False,
                )
                if tried is not None:
                    break
                value = self._read_one()
                values.append(value)
                if key is not None:
                    self._append(key, value)
            return "".join(values)

    def _take_data_type_as_json(
        self,
        *,
        until: Union[str, Tuple[str, ...]],
        key: OutputKey,
        data_type: FunctionCallParameterDataType,
        always_nullable: bool = False,
        should_consume_suffix: bool = True,
    ):
        """
        Unlike `.take_any`, `until` and `key` is required here, because
        - it is unnecessary and a little bit annoying to maintain the streaming state
          for most data types other than `str`.
        - if you don't need the output, just use `.take_any`.

        NOTE: if there are various acceptable data types, we choose the FIRST convertible
        one, indicating, if the first choice is `str`, the following choices will be
        IGNORED, as everything could be a string.

        TODO: if there are acceptable data types before `str`, we should peek until they
        are excluded and then choose `str`, instead of choosing at the end.
        """
        if not data_type.streaming:
            value = yield from self._take_any(
                until=until, should_consume_suffix=should_consume_suffix
            )
            value = data_type.convert(value, always_nullable=always_nullable)
            self._append(key, json_dumps(value))
            return value
        else:
            if always_nullable:
                # very tricky.
                # even if the data type is string, we still need to check if it's null.
                tried = yield from self._literal(
                    _string_cartesian_product(NULL_STRINGS, until),
                    should_consume=should_consume_suffix,
                    should_raise=False,
                )
                if tried is not None:
                    if not should_consume_suffix:
                        # NOTE: length of "null" (case-insensitive) is 4.
                        self._consume(4)
                    self._append(key, "null")
                    return None
            values = []
            self._append(key, '"')
            while True:
                tried = yield from self._literal(
                    until, should_consume=should_consume_suffix, should_raise=False
                )
                if tried is not None:
                    break
                value = self._read_one()
                values.append(value)
                self._append(key, json_dumps(value)[1:-1])
            self._append(key, '"')
            return "".join(values)

    def _literal(
        self,
        target: Union[str, Tuple[str, ...]],
        *,
        should_raise: bool = True,
        should_consume: bool = True,
    ) -> Generator[Optional[str], None, None]:
        """
        NOTE: stops at the first matched target, even if it is a substring of another later target.
        """
        if isinstance(target, str):
            target = (target,)
        char_index = 0
        target_index = 0
        target_index_max = len(target) - 1
        while True:
            ith = yield from self._peek(char_index)
            if ith != target[target_index][char_index]:
                already_matched_and_ith = target[target_index][:char_index] + ith
                prefix_matching_result = PrefixMatchingResult.mismatch
                target_index += 1
                while target_index <= target_index_max:
                    prefix_matching_result = PrefixMatchingResult.check(
                        target[target_index], already_matched_and_ith
                    )
                    if prefix_matching_result != PrefixMatchingResult.mismatch:
                        break
                    target_index += 1
                if prefix_matching_result == PrefixMatchingResult.substring_of_prefix:
                    break
                elif prefix_matching_result == PrefixMatchingResult.mismatch:
                    if should_raise:
                        if len(target) == 1:
                            raise self._error(
                                expected=target[0][char_index:],
                                actual=ith,
                                reason=f"matching {target[0]!r}",
                            )
                        else:
                            raise self._error(
                                expected=target,
                                actual=ith,
                                reason="matching any of the list",
                            )
                    else:
                        return
            char_index += 1
            if char_index == len(target[target_index]):
                break
        if should_consume:
            self._consume(len(target[target_index]))
        return target[target_index]

    def _whitespace(self) -> Iterator[None]:
        total = 0
        while True:
            peek = yield from self._peek(total)
            if peek.isspace():
                total += 1
            else:
                self._consume(total)
                break


@dataclass(slots=True)
class Tried:
    successful: bool
    result: Any = None
    error: Optional[PatternMismatched] = None


def default_tool_call_output_key(tool_call_index: int, function: Dict) -> Dict:
    return {
        "tool_calls": [
            {
                "index": tool_call_index,
                "function": function,
            }
        ]
    }


def _get_context_standalone(full_text_parts: List[str], offset: int) -> str:
    """Standalone version of _get_context that doesn't capture self."""
    full_text = "".join(full_text_parts)
    return f"{full_text[:offset]}💥{full_text[offset:]}"


def _merge_dicts(
    src: Dict[str, Any],
    dst: Optional[Dict[str, Any]],
    whitelist: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if dst is None:
        return deepcopy(src)
    for key, val_src in src.items():
        if whitelist is not None and key in whitelist:
            continue
        val_dst = dst.get(key)
        if val_dst is None:
            dst[key] = deepcopy(val_src)
        elif isinstance(val_src, dict):
            dst[key] = _merge_dicts(val_src, val_dst)
        elif isinstance(val_src, list):
            # NOTE: don't just `val_dst.extend(val_src)` here, as we need a carefully constructed delta object.
            # Assume that all elements have `index: int`, indicating its position in the final merged list.
            if len(val_src) == len(val_dst) and all(
                val_src[i]["index"] == val_dst[i]["index"] for i in range(len(val_src))
            ):
                for i in range(len(val_src)):
                    _merge_dicts(val_src[i], val_dst[i], whitelist={"index"})
            elif len(val_src):
                index_to_item = {item["index"]: item for item in val_dst}
                for item_src in val_src:
                    idx = item_src["index"]
                    item_dst = index_to_item.get(idx)
                    if item_dst is None:
                        index_to_item[idx] = deepcopy(item_src)
                    else:
                        _merge_dicts(item_src, item_dst, whitelist={"index"})
                dst[key] = sorted(
                    index_to_item.values(), key=lambda item: item["index"]
                )
        elif isinstance(val_src, str):
            if isinstance(val_dst, list):
                dst[key].append(val_src)
            else:
                # NOTE: to avoid O(n²) string concatenation, we store one string as a list of substrings.
                # Later, join the substrings back by `_get_dict_with_joint_strings`.
                dst[key] = [val_dst, val_src]
        else:
            raise TypeError(
                f"key {key} has unsupported type {type(val_src)} of {val_src!r} against {val_dst!r}"
            )
    return dst


def _get_dict_with_joint_strings(
    data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if data is None:
        return
    updated = {}
    for key, val in data.items():
        if isinstance(val, list):
            if val:
                if isinstance(val[0], str):
                    updated[key] = "".join(val)
                elif isinstance(val[0], dict):
                    updated[key] = [_get_dict_with_joint_strings(_) for _ in val]
                else:
                    updated[key] = val
            else:
                updated[key] = val
        elif isinstance(val, dict):
            updated[key] = _get_dict_with_joint_strings(val)
        else:
            updated[key] = val
    return updated


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


class PrefixMatchingResult(int, Enum):
    startswith = auto()
    substring_of_prefix = auto()
    mismatch = auto()

    @classmethod
    def check(cls, string: str, prefix: str) -> Self:
        if not prefix:
            return cls.startswith
        for i in range(len(prefix)):
            if i >= len(string):
                return cls.substring_of_prefix
            if string[i] != prefix[i]:
                return cls.mismatch
        return cls.startswith


@functools.cache
def _string_cartesian_product(
    prefix: Tuple[str, ...], suffix: Union[str, Tuple[str, ...]]
) -> Tuple[str, ...]:
    if isinstance(suffix, str):
        return tuple(_ + suffix for _ in prefix)
    else:
        return tuple(_prefix + _suffix for _prefix in prefix for _suffix in suffix)
