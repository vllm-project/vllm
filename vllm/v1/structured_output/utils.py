# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import TYPE_CHECKING, TypeVar

import regex as re
import torch
from cachetools import LRUCache

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.import_utils import LazyLoader
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

if TYPE_CHECKING:
    import outlines_core as oc
    import transformers.convert_slow_tokenizer as convert_slow_tokenizer
    import transformers.file_utils as file_utils
    import xgrammar as xgr

    from vllm.tokenizers import TokenizerLike
    from vllm.v1.worker.gpu_input_batch import InputBatch
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    oc = LazyLoader("oc", globals(), "outlines_core")
    file_utils = LazyLoader("file_utils", globals(), "transformers.file_utils")
    convert_slow_tokenizer = LazyLoader(
        "convert_slow_tokenizer", globals(), "transformers.convert_slow_tokenizer"
    )


logger = init_logger(__name__)

_T = TypeVar("_T")

CACHE = None
_BITMASK_BIT_OFFSETS: dict[tuple[str, int | None], torch.Tensor] = {}


def _get_bitmask_bit_offsets(device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    key = (device.type, device.index)
    bit_offsets = _BITMASK_BIT_OFFSETS.get(key)
    if bit_offsets is None:
        bit_offsets = torch.arange(32, dtype=torch.int32, device=device)
        _BITMASK_BIT_OFFSETS[key] = bit_offsets
    return bit_offsets


def _apply_grammar_bitmask_torch(
    logits: torch.Tensor,
    grammar_bitmask: torch.Tensor,
    out_indices: list[int],
    skip_out_indices: bool,
) -> None:
    """Apply packed xgrammar bitmasks with PyTorch tensor ops."""
    if skip_out_indices:
        selected_logits = logits
        selected_bitmask = grammar_bitmask
        index_tensor = None
    else:
        if not out_indices:
            return
        index_tensor = torch.tensor(out_indices, dtype=torch.long, device=logits.device)
        selected_logits = logits.index_select(0, index_tensor)
        selected_bitmask = grammar_bitmask.index_select(0, index_tensor)

    bit_offsets = _get_bitmask_bit_offsets(logits.device)
    disallowed = (((selected_bitmask[..., None] >> bit_offsets) & 1) == 0).reshape(
        selected_bitmask.shape[0], -1
    )
    selected_logits.masked_fill_(disallowed[:, : logits.shape[-1]], -float("inf"))

    if index_tensor is not None:
        logits.index_copy_(0, index_tensor, selected_logits)


@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    bitmask_ptr,
    indices_ptr,
    vocab_size,
    logits_stride,
    bitmask_stride,
    HAS_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    block_id = tl.program_id(0)
    row_id = tl.program_id(1)
    logits_row = row_id
    if HAS_INDICES:
        logits_row = tl.load(indices_ptr + row_id)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bitmask_offsets = block_id * (BLOCK_SIZE // 32) + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + row_id * bitmask_stride + bitmask_offsets,
        mask=bitmask_offsets < bitmask_stride,
        other=0,
    )
    disallowed = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    disallowed = disallowed.reshape(BLOCK_SIZE)

    tl.store(
        logits_ptr + logits_row * logits_stride + offsets,
        -float("inf"),
        mask=(offsets < vocab_size) & disallowed,
    )


def _apply_grammar_bitmask_triton(
    logits: torch.Tensor,
    grammar_bitmask: torch.Tensor,
    logits_indices: torch.Tensor | None = None,
) -> None:
    vocab_size = min(logits.shape[-1], grammar_bitmask.shape[-1] * 32)
    num_rows = grammar_bitmask.shape[0]
    if num_rows == 0:
        return

    block_size = 4096
    props = torch.cuda.get_device_properties(logits.device)
    arch = getattr(props, "gcnArchName", "")
    # AMD CDNA wavefronts use 64 lanes. Keep this in sync with xgrammar's
    # Triton bitmask kernel configuration.
    warp_size = 64 if torch.version.hip is not None and "gfx1" not in arch else 32

    grid = (triton.cdiv(vocab_size, block_size), num_rows)
    _apply_grammar_bitmask_kernel[grid](
        logits,
        grammar_bitmask,
        logits_indices,
        vocab_size,
        logits.stride(0),
        grammar_bitmask.stride(0),
        logits_indices is not None,
        block_size,
        num_warps=block_size // warp_size // (16 // logits.element_size()),
        num_stages=3,
    )


def compile_regex_with_timeout(fn: Callable[[str], _T], pattern: str) -> _T:
    """Run a regex compilation callable with a timeout.

    Prevents ReDoS attacks where adversarial regex patterns (e.g. nested
    quantifiers like ``(a+)+b``) cause exponential DFA state-space explosion,
    hanging the inference worker indefinitely.

    Args:
        fn: Single-argument callable that takes the pattern and performs
            the regex compilation.
        pattern: The regex pattern string, passed to *fn* and included in
            timeout error messages.

    Raises:
        ValueError: If compilation exceeds the configured timeout.
    """
    timeout = envs.VLLM_REGEX_COMPILATION_TIMEOUT_S
    if timeout <= 0:
        return fn(pattern)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, pattern)
    try:
        result = future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise ValueError(
            f"Regex compilation timed out after {timeout}s. "
            "The pattern may be too complex or contain constructs that "
            "cause exponential state-space explosion (e.g. nested "
            f"quantifiers). Pattern: {pattern[:200]}"
        ) from None
    else:
        executor.shutdown(wait=False)
        return result


def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices: list[int] = []
    bitmask_indices: list[int] = []

    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                out_indices.append(bitmask_index)
                bitmask_indices.append(cumulative_index + i)
        cumulative_index += 1 + num_spec_tokens

    if not out_indices:
        return

    if not logits.is_cpu and current_platform.is_rocm() and HAS_TRITON:
        # Keep the ROCm path compact: one bitmask row per row that will be
        # masked, plus an optional logits-row mapping for sparse batches.
        paired_indices = sorted(zip(out_indices, bitmask_indices))
        out_indices = [logit_idx for logit_idx, _ in paired_indices]
        bitmask_indices = [bitmask_idx for _, bitmask_idx in paired_indices]

        compact_bitmask_tensor = torch.empty(
            (len(bitmask_indices), grammar_bitmask.shape[1]),
            dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
            pin_memory=PIN_MEMORY,
        )
        compact_bitmask = compact_bitmask_tensor.numpy()
        for row, bitmask_idx in enumerate(bitmask_indices):
            compact_bitmask[row] = grammar_bitmask[bitmask_idx]

        grammar_bitmask_tensor = compact_bitmask_tensor.to(
            logits.device, non_blocking=True
        )
        logits_indices = None
        if out_indices != list(range(logits.shape[0])):
            logits_indices = async_tensor_h2d(
                out_indices, dtype=torch.int32, device=logits.device
            )
        _apply_grammar_bitmask_triton(
            logits,
            grammar_bitmask_tensor,
            logits_indices,
        )
        return

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask_tensor = torch.full(
        (logits.shape[0], grammar_bitmask.shape[1]),
        -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
        pin_memory=PIN_MEMORY,
    )
    sorted_bitmask = sorted_bitmask_tensor.numpy()
    for logit_idx, bitmask_idx in zip(out_indices, bitmask_indices):
        sorted_bitmask[logit_idx] = grammar_bitmask[bitmask_idx]

    # Copy async to device.
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
        if current_platform.is_rocm():
            # xgrammar auto-dispatches HIP tensors to its Triton bitmask
            # kernel because PyTorch reports them as cuda tensors. Recent
            # ROCm CI failures in structured-output generation point at that
            # kernel being JITed during inference, so keep xgrammar for
            # grammar compilation/matching. If Triton is unavailable, fall
            # back to PyTorch with the full reordered bitmask.
            _apply_grammar_bitmask_torch(
                logits, grammar_bitmask, out_indices, skip_out_indices
            )
            return

        index_tensor = None
        if not skip_out_indices:
            # xgrammar expects a python list of indices but it will actually work with
            # a tensor. If we copy the tensor ourselves here we can do it in a
            # non_blocking manner and there should be no cpu sync within xgrammar.
            index_tensor = async_tensor_h2d(
                out_indices, dtype=torch.int32, device=logits.device
            )

        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
        return

    # CPU case, use list for indices.
    indices = None if skip_out_indices else out_indices
    # Handle dtype conversion for CPU (older xgrammar CPU kernels require float32)
    # See: https://github.com/vllm-project/vllm/issues/31901
    if logits.dtype != torch.float32:
        # Convert to float32, apply bitmask, then convert back
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, grammar_bitmask, indices=indices)
        # Copy the modified values back to the original tensor
        logits.copy_(logits_fp32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=indices)


class OutlinesVocabulary:
    """
    Wrapper class for `outlines_core.Vocabulary`,
    which allows us to store a hash with the vocabulary
    """

    def __init__(self, vocabulary: oc.Vocabulary) -> None:
        # Actual vocabulary object
        self.inner = vocabulary
        # Have to do abs(hash()) because python hashes can
        # be negative, and we are using hash as a cache key.
        hex_str = hashlib.sha256(vocabulary.__repr__().encode("utf-8")).hexdigest()
        hash_int = int(hex_str, 16)
        self._hash = hash_int


def get_outlines_cache_path() -> str:
    """Get the context object that contains previously-computed return values"""
    outlines_cache_dir = os.getenv("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    home_dir = os.path.expanduser("~")

    if outlines_cache_dir:
        # OUTLINES_CACHE_DIR takes precedence
        return outlines_cache_dir
    if xdg_cache_home:
        return os.path.join(xdg_cache_home, ".cache", "outlines")
    # If homedir is "/", we may be inside a container, and thus writing to
    # root would be problematic, so we fall back to using a tempfile.
    # Also validate the path exists, since os.path.expanduser does
    # not guarantee existence.
    if os.path.isdir(home_dir) and home_dir != "/":
        # Default Unix fallback: ~/.cache/outlines
        return os.path.join(home_dir, ".cache", "outlines")

    # home_dir may be / inside a docker container without existing user
    tempdir = tempfile.gettempdir()
    return os.path.join(tempdir, ".cache", "outlines")


def get_outlines_cache():
    """Get the Cache instance to be used for index caching"""

    cache_dir = get_outlines_cache_path()
    if envs.VLLM_V1_USE_OUTLINES_CACHE:
        from diskcache import Cache

        logger.warning(
            "Enabling outlines cache. This is an unbounded on-disk "
            "cache. It may consume a lot of disk space and should "
            "not be used with untrusted clients."
        )
        cache = Cache(cache_dir, eviction_policy="none", cull_limit=0)
        outlines_version = importlib.metadata.version("outlines_core")

        cached_version = cache.get("__version__", None)
        if cached_version != outlines_version:
            cache.clear()
        cache.set("__version__", outlines_version)
        return cache

    return LRUCache(maxsize=128)


re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")
re_replacement_seq = re.compile(r"^.{0,6}�+.{0,6}$")


def _reduced_vocabulary(tokenizer: TokenizerLike) -> dict[bytes, list[int]]:
    """Create a map from vocabulary tokens to lists of equivalent token ids.

    Returns:
        A Dict of token string -> equivalent token ids
    """
    eos_token_id = tokenizer.eos_token_id

    unicode_to_bytes = {
        v: k for k, v in convert_slow_tokenizer.bytes_to_unicode().items()
    }

    def convert_token_to_string(token: str) -> str:
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (
            type(token) is str
            and token.startswith(file_utils.SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    vocabulary: dict[bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        if token in tokenizer.all_special_tokens:
            continue

        token_str = convert_token_to_string(token)
        if token_str:
            if isinstance(token, (bytes, bytearray)):
                # For BPE tokenizers where tokens are stored as bytes.

                # safe to ignore since token_str is of type (bytearray, bytes)
                # by this point.
                token_bytes = bytes(token_str)  # type: ignore[arg-type]

            elif (token_str == "\ufffd" and token != "\ufffd") or (
                "\ufffd" in token_str and not re_replacement_seq.match(token_str)
            ):
                # Handle tokens with invalid UTF-8 sequences.
                if re_llama_byte_token.match(token):
                    # Llama-like tokenizers use <0xXX> for incomplete sequences.
                    token_bytes = bytes([int(token[3:5], 16)])
                else:
                    # GPT2 tokenizers: map each byte back using unicode_to_bytes
                    byte_vals = [unicode_to_bytes.get(c) for c in token]
                    if None in byte_vals:
                        raise RuntimeError(
                            f"Cannot convert token `{token}`"
                            f" ({token_idx}) to bytes: {token_str}"
                        )
                    # safe to ignore, since if None in byte_vals,
                    # an error is thrown.
                    token_bytes = bytes(byte_vals)  # type: ignore[arg-type]
            else:
                token_bytes = token_str.encode("utf-8")

            if token_idx != eos_token_id:
                vocabulary.setdefault(token_bytes, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)

    return vocabulary


def get_outlines_vocabulary(tokenizer: TokenizerLike) -> oc.Vocabulary:
    """Get the `Vocabulary` object for a given tokenizer."""
    if hasattr(tokenizer, "_outlines_vocabulary"):
        return tokenizer._outlines_vocabulary  # type: ignore

    reduced_vocab = _reduced_vocabulary(tokenizer)
    vocabulary = OutlinesVocabulary(
        oc.Vocabulary(tokenizer.eos_token_id, reduced_vocab)
    )
    tokenizer._outlines_vocabulary = vocabulary  # type: ignore

    return vocabulary


def grammar_is_likely_lark(grammar_str: str) -> bool:
    """
    Check if grammar appears to use Lark syntax.

    Args:
        grammar_str: Input grammar string

    Returns:
        bool: True if grammar appears to be in Lark format, False otherwise

    Examples:
        >>> grammar_is_likely_lark("rule: 'abc'")
        True
        >>> grammar_is_likely_lark("rule ::= 'abc'")
        False
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split("\n"):
        # Remove both comment styles
        line = re.sub(r"(#|//).*$", "", line).strip()
        if not line:
            continue

        # Look for EBNF rule definition
        if "::=" in line:
            return False

    return True


def convert_lark_to_ebnf(grammar_str: str) -> str:
    """
    Convert a Lark grammar string to EBNF format.

    EBNF reference:
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    Lark grammar reference:
    https://lark-parser.readthedocs.io/en/latest/grammar.html

    Args:
        grammar_str: Input grammar in Lark format

    Returns:
        str: Converted grammar in EBNF format

    Examples:
        >>> print(convert_lark_to_ebnf("rule: 'hello'"))
        root ::= rule
        rule ::= "hello"
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"Grammar must be a string, got {type(grammar_str)}")
    if not grammar_str.strip():
        raise ValueError("Grammar string cannot be empty")

    defined_rules = set()
    referenced_rules = set()
    output_lines = []

    def clean_line(line: str) -> str:
        """Remove comments and whitespace from line."""
        return re.sub(r"(#|//).*$", "", line).strip()

    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        """Validate quote matching in text."""
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(f"Mismatched quotes in {rule_name} on line {line_num}")

    def extract_references(text: str) -> set[str]:
        """Extract rule references from text."""
        # Remove quoted strings and special characters
        text = re.sub(r'"[^"]*"', "", text)
        text = re.sub(r"[+*?()|\[\]{}]", " ", text)
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text))

    # First pass: Find root rule and validate rule definitions
    lines = [clean_line(line) for line in grammar_str.split("\n")]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith("|"):
            continue

        if ":" in line:
            try:
                name = line.split(":", 1)[0].strip().strip("?")
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                if name == "start":
                    first_rule = "start"
            except IndexError as e:
                raise ValueError(
                    f"Invalid rule format on line {line_num}. "
                    "Expected 'rule_name: definition'"
                ) from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    # Add root rule
    output_lines.append(f"root ::= {first_rule}")

    # Second pass: Process rule definitions and alternatives
    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue

        try:
            if ":" in line and not line.startswith("|"):
                # Save previous rule if exists
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}"
                    )

                # Process new rule
                name, definition = line.split(":", 1)
                current_rule = name.strip().strip("?")

                check_quotes(definition, f"rule '{current_rule}'", line_num)
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]

            elif line.startswith("|"):
                if not current_rule:
                    raise ValueError(
                        f"Alternative '|' on line {line_num} "
                        "without a preceding rule definition"
                    )

                alt_def = line[1:].strip()
                check_quotes(
                    alt_def, f"alternative for rule '{current_rule}'", line_num
                )
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)

        except ValueError as e:
            raise ValueError(f"Error on line {line_num}: {str(e)}") from e

    # Add final rule if exists
    if current_rule:
        output_lines.append(f"{current_rule} ::= {' | '.join(current_definition)}")

    # Validate all rules are defined
    undefined_rules = referenced_rules - defined_rules - {"root"}
    if undefined_rules:
        raise ValueError(
            f"Referenced rules are not defined: {', '.join(sorted(undefined_rules))}"
        )

    return "\n".join(output_lines)


def choice_as_grammar(choice: list[str]) -> str:
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r"\\\1", s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
