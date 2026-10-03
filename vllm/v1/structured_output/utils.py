# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import os
import queue
import signal
import sqlite3
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import regex as re
import torch
from cachetools import LRUCache

import vllm.envs as envs
from vllm.logger import init_logger
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

# Separate from the compile deadline so interpreter startup is not counted
# against a legitimate pattern. The worker is reused after the first start.
_WORKER_STARTUP_TIMEOUT_S = 120
_xgr_tokenizer_cache: LRUCache = LRUCache(maxsize=2)
_compile_pool: _RegexCompilePool | None = None
_compile_pool_lock = threading.Lock()


def strip_speculative_padding(token_ids: list[int]) -> list[int]:
    """Drop speculative-decoding padding from a token block.

    ngram and other speculative backends pad rejected draft positions with a
    -1 sentinel. Structured-output grammars treat every entry as a real token
    id, so the sentinels (and everything after the first one) are removed here,
    before tokens reach any backend, rather than inside a single backend.
    """
    for i, token_id in enumerate(token_ids):
        if token_id < 0:
            return token_ids[:i]
    return token_ids


def _pattern_excerpt(pattern: str) -> str:
    return pattern[:200]


def _timeout_error(timeout: float, pattern: str) -> str:
    excerpt = _pattern_excerpt(pattern)
    return (
        f"Regex compilation timed out after {timeout}s. "
        "The pattern may be too complex or contain constructs that "
        "cause exponential state-space explosion (e.g. nested "
        f"quantifiers). Pattern: {excerpt}"
    )


def _process_exit_error(exitcode: int | None, pattern: str) -> str:
    excerpt = _pattern_excerpt(pattern)
    return f"Regex compilation process exited with code {exitcode}. Pattern: {excerpt}"


def _regex_compile_worker_main(job_queue: Any, result_queue: Any) -> None:
    """Run compilation jobs until the parent kills this process."""
    result_queue.put(("ready", None))
    while True:
        item = job_queue.get()
        if item is None:
            return
        fn, args = item
        try:
            payload: tuple[str, Any] = ("ok", fn(*args))
        except Exception as exc:
            payload = ("err", exc)
        try:
            result_queue.put(payload)
        except Exception as exc:
            result_queue.put(("err", RuntimeError(f"{type(exc).__name__}: {exc}")))


class _CompileWorker:
    """One long-lived compiler process and the queues that talk to it."""

    def __init__(self) -> None:
        self.process: Any = None
        self.job_queue: Any = None
        self.result_queue: Any = None
        self.last_pid: int | None = None
        self.start_method: str | None = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    @property
    def exitcode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.exitcode

    def ensure_started(self) -> None:
        if self.is_alive():
            return
        self.kill()
        self._start()

    def _start(self) -> None:
        # Resolve the start method at process creation. Caching an earlier
        # fork context would ignore a CUDA init that happened since then.
        from vllm.utils.system_utils import get_mp_context

        ctx = get_mp_context()
        self.start_method = ctx.get_start_method()
        self.job_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(
            target=_regex_compile_worker_main,
            args=(self.job_queue, self.result_queue),
            daemon=True,
            name="RegexCompileWorker",
        )
        self.process.start()
        self.last_pid = self.process.pid
        try:
            status, _payload = self.result_queue.get(timeout=_WORKER_STARTUP_TIMEOUT_S)
        except queue.Empty:
            self.kill()
            raise ValueError(
                "Regex compilation worker failed to start within "
                f"{_WORKER_STARTUP_TIMEOUT_S}s."
            ) from None
        if status != "ready":
            self.kill()
            raise ValueError(
                "Regex compilation worker sent an unexpected startup message."
            )

    def kill(self) -> None:
        """SIGKILL the child and drop its queues so they cannot be reused."""
        proc = self.process
        job_queue = self.job_queue
        result_queue = self.result_queue
        self.process = None
        self.job_queue = None
        self.result_queue = None
        if proc is not None and proc.pid is not None and proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(timeout=5)
        elif proc is not None:
            proc.join(timeout=1)
        for worker_queue in (job_queue, result_queue):
            if worker_queue is None:
                continue
            worker_queue.cancel_join_thread()
            with contextlib.suppress(ValueError, OSError):
                worker_queue.close()


def _kill_compile_workers(workers: list[_CompileWorker]) -> None:
    for worker in workers:
        worker.kill()


class _RegexCompilePool:
    """Bounded pool of reusable regex-compilation workers."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(
                f"VLLM_REGEX_COMPILATION_MAX_CONCURRENT must be at least 1, got {size}."
            )
        self._size = size
        self._idle: queue.Queue[_CompileWorker] = queue.Queue()
        self._workers: list[_CompileWorker] = []
        for _ in range(size):
            worker = _CompileWorker()
            self._workers.append(worker)
            self._idle.put(worker)
        self._finalizer = weakref.finalize(self, _kill_compile_workers, self._workers)

    def submit(
        self,
        fn: Callable[..., _T],
        args: tuple[Any, ...],
        timeout: float,
        pattern: str,
    ) -> _T:
        try:
            worker = self._idle.get(timeout=timeout)
        except queue.Empty:
            excerpt = _pattern_excerpt(pattern)
            raise ValueError(
                "Regex compilation could not acquire a compile slot within "
                f"{timeout}s (max concurrent: {self._size}). Pattern: {excerpt}"
            ) from None
        try:
            return self._run(worker, fn, args, timeout, pattern)
        finally:
            self._idle.put(worker)

    def _run(
        self,
        worker: _CompileWorker,
        fn: Callable[..., _T],
        args: tuple[Any, ...],
        timeout: float,
        pattern: str,
    ) -> _T:
        worker.ensure_started()
        assert worker.job_queue is not None
        assert worker.result_queue is not None
        worker.job_queue.put((fn, args))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                worker.kill()
                raise ValueError(_timeout_error(timeout, pattern)) from None
            try:
                status, payload = worker.result_queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                if not worker.is_alive():
                    exitcode = worker.exitcode
                    worker.kill()
                    raise ValueError(_process_exit_error(exitcode, pattern)) from None
                continue
            except (EOFError, OSError, BrokenPipeError):
                exitcode = worker.exitcode
                worker.kill()
                raise ValueError(_process_exit_error(exitcode, pattern)) from None
            if status == "ok":
                return payload
            if status == "err":
                raise payload
            worker.kill()
            raise ValueError(
                "Regex compilation process produced no result. "
                f"Pattern: {_pattern_excerpt(pattern)}"
            ) from None

    def shutdown(self) -> None:
        if self._finalizer.alive:
            self._finalizer()


def _get_compile_pool() -> _RegexCompilePool:
    global _compile_pool
    if _compile_pool is None:
        with _compile_pool_lock:
            if _compile_pool is None:
                _compile_pool = _RegexCompilePool(
                    envs.VLLM_REGEX_COMPILATION_MAX_CONCURRENT
                )
    return _compile_pool


def shutdown_regex_compile_pool() -> None:
    """Stop compiler workers and drop the pool so the next compile starts fresh."""
    global _compile_pool
    with _compile_pool_lock:
        pool = _compile_pool
        _compile_pool = None
    if pool is not None:
        pool.shutdown()


def compile_regex_with_timeout(fn: Callable[..., _T], *args: Any, pattern: str) -> _T:
    """Run a regex compilation callable with a timeout in a killable process.

    The compile runs in a worker started via ``get_mp_context()``, so the
    start method follows vLLM's multiprocessing policy (spawn once CUDA is
    initialized). Workers are reused across compiles. On timeout the worker
    is SIGKILL'd and a replacement is started on the next job, so the work
    cannot keep running after the error is returned.

    A timeout of 0 or less runs ``fn`` in-process and does not start a worker.

    Args:
        fn: Picklable callable that performs the compilation.
        *args: Picklable arguments passed to ``fn``.
        pattern: Regex text included in timeout error messages. Not passed
            to ``fn`` unless it is also one of ``args``.

    Raises:
        ValueError: If compilation exceeds the configured timeout or a
            worker cannot be acquired in time.

    """
    timeout = envs.VLLM_REGEX_COMPILATION_TIMEOUT_S
    if timeout <= 0:
        return fn(*args)
    return _get_compile_pool().submit(fn, args, timeout, pattern)


def _xgr_grammar_from_regex(pattern: str) -> str:
    """Picklable worker: compile regex via xgrammar, return serialized JSON."""
    import xgrammar as xgr

    return xgr.Grammar.from_regex(pattern).serialize_json()


def _xgr_compile_regex(tokenizer_info_json: str, pattern: str) -> str:
    """Picklable worker: rebuild compiler in subprocess and compile regex."""
    import xgrammar as xgr

    info = _xgr_tokenizer_cache.get(tokenizer_info_json)
    if info is None:
        info = xgr.TokenizerInfo.deserialize_json(tokenizer_info_json)
        _xgr_tokenizer_cache[tokenizer_info_json] = info
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    return compiler.compile_regex(pattern).serialize_json()


def _outlines_compile_index(pattern: str, vocabulary: Any) -> Any:
    """Picklable worker: build outlines Index from pattern + vocabulary."""
    import outlines_core as oc

    return oc.Index(pattern, vocabulary)


def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        grammar_output (GrammarOutput): The grammar bitmask to apply.
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

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask_tensor = torch.full(
        (logits.shape[0], grammar_bitmask.shape[1]),
        -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
        pin_memory=PIN_MEMORY,
    )
    sorted_bitmask = sorted_bitmask_tensor.numpy()
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # Copy async to device.
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
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
    """Wrapper class for `outlines_core.Vocabulary`,
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
    """Get the context object that contains previously-computed return values."""
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


class OutlinesDiskCache:
    """SQLite-backed cache for outlines_core.Index objects.

    Uses outlines_core's native binary serialization (via Rust serde)
    instead of pickle, eliminating arbitrary code execution risk on
    deserialization.
    """

    _TYPE_INDEX = "I"
    _TYPE_STRING = "S"

    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        db_path = os.path.join(path, "outlines_cache.db")
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, type_tag TEXT NOT NULL, value BLOB NOT NULL)"
        )
        self._db.commit()

    def __contains__(self, key: str) -> bool:
        row = self._db.execute("SELECT 1 FROM cache WHERE key=?", (key,)).fetchone()
        return row is not None

    def __getitem__(self, key: str):
        row = self._db.execute(
            "SELECT type_tag, value FROM cache WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        type_tag, data = row
        if type_tag == self._TYPE_STRING:
            return data.decode("utf-8")
        return oc.Index.from_binary(data)

    def __setitem__(self, key: str, value):
        if isinstance(value, str):
            type_tag = self._TYPE_STRING
            data = value.encode("utf-8")
        else:
            type_tag = self._TYPE_INDEX
            data = value.__reduce__()[1][0]
        self._db.execute(
            "INSERT OR REPLACE INTO cache (key, type_tag, value) VALUES (?, ?, ?)",
            (key, type_tag, data),
        )
        self._db.commit()

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value):
        self[key] = value

    def clear(self):
        self._db.execute("DELETE FROM cache")
        self._db.commit()


def get_outlines_cache():
    """Get the Cache instance to be used for index caching."""
    cache_dir = get_outlines_cache_path()
    if envs.VLLM_V1_USE_OUTLINES_CACHE:
        logger.warning(
            "Enabling outlines cache. This is an unbounded on-disk "
            "cache. It may consume a lot of disk space and should "
            "not be used with untrusted clients."
        )
        cache = OutlinesDiskCache(cache_dir)
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
    """Check if grammar appears to use Lark syntax.

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


def choice_as_grammar(choice: list[str]) -> str:
    def escape_ebnf_string(s: str) -> str:
        """Escape EBNF literals, including raw LF, CR, and NUL terminators."""
        escapes = {"\\": r"\\", '"': r"\"", "\n": r"\n", "\r": r"\r", "\t": r"\t"}

        def escape_char(ch: str) -> str:
            if ch in escapes:
                return escapes[ch]
            # Escape remaining C0 controls (U+0000-U+001F) and DEL (U+007F).
            if ord(ch) < 0x20 or ord(ch) == 0x7F:
                return f"\\u{ord(ch):04x}"
            return ch

        return "".join(escape_char(ch) for ch in s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
