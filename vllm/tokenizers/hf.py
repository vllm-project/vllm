# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import queue
from pathlib import Path
from typing import TypeAlias, TypeVar

from tokenizers import Tokenizer, processors
from transformers import AutoTokenizer, PythonBackend, TokenizersBackend
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import cached_file

from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PythonBackend | TokenizersBackend
_T = TypeVar("_T", bound=TokenizerLike)


class ThreadSafeHFTokenizerMixin:
    """Mixin class for thread-safe HF fast tokenizers."""

    pass


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1):
    """If `tokenizer` is a `TokenizersBackend`, modify the tokenizer
    in-place to make the public interface thread-safe by routing calls
    through a deep-copied tokenizer pool.

    Note that:
    - Only ``TokenizerLike``'s public interface is thread-safe.
      This doesn't include ``_tokenizer`` property nor any mutation
      methods like ``add_special_tokens`` or ``add_tokens``.
    - Adjacent method calls could happen on different deep copies.
    """
    if not isinstance(tokenizer, TokenizersBackend) or isinstance(
        tokenizer, ThreadSafeHFTokenizerMixin
    ):
        return tokenizer

    og_tokenizer = copy.copy(tokenizer)

    tokenizer_pool: queue.Queue[TokenizersBackend] = queue.Queue()
    for _ in range(copies):
        tokenizer_pool.put(copy.deepcopy(og_tokenizer))

    @contextlib.contextmanager
    def _borrow_from_pool():
        try:
            tok = tokenizer_pool.get_nowait()
            yield tok
        except queue.Empty:
            tok = copy.deepcopy(og_tokenizer)
            yield tok
        finally:
            tokenizer_pool.put(tok)

    class TokenizerPool(tokenizer.__class__, ThreadSafeHFTokenizerMixin):  # type: ignore
        def apply_chat_template(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.apply_chat_template(*args, **kwargs)

        def batch_decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_decode(*args, **kwargs)

        def batch_encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_encode(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_ids(*args, **kwargs)

        def convert_ids_to_tokens(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_ids_to_tokens(*args, **kwargs)

        def convert_tokens_to_string(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_string(*args, **kwargs)

        def decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.decode(*args, **kwargs)

        def encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.encode(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok(*args, **kwargs)

        def __reduce__(self):
            return maybe_make_thread_pool, (og_tokenizer, copies)

    TokenizerPool.__name__ = f"TokenizerPool{og_tokenizer.__class__.__name__}"

    tokenizer.__class__ = TokenizerPool
    # Return the tokenizer: TokenizerPool.__reduce__ reconstructs through this
    # function, so falling off the end would unpickle to None (issue #45433).
    return tokenizer


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)
    # The underlying tokenizer class could be a specific backend,
    # which does not always implement is_fast in Transformers
    tokenizer_is_fast = getattr(tokenizer, "is_fast", True)

    max_token_id = max(tokenizer_vocab.values())
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)

    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        @property
        def max_chars_per_token(self) -> int:
            return max_chars_per_token

        @property
        def is_fast(self) -> bool:
            return tokenizer_is_fast

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def save_pretrained(self, *args, **kwargs):
            # Serialize the original class, not this process-local cache wrapper.
            uncached_tokenizer = copy.copy(self)
            uncached_tokenizer.__class__ = tokenizer.__class__
            return uncached_tokenizer.save_pretrained(*args, **kwargs)

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


def _maybe_fix_gte_tokenizer(
    tokenizer: HfTokenizer,
    path_or_repo_id: str | Path,
    *,
    revision: str | None,
    download_dir: str | None,
    **kwargs,
) -> None:
    config = kwargs.get("config")
    if (
        getattr(config, "model_type", None) != "qwen2"
        or getattr(config, "is_causal", True) is not False
        or not isinstance(tokenizer, TokenizersBackend)
        or type(tokenizer).__name__ != "Qwen2TokenizerFast"
        or "add_eos_token" in kwargs
    ):
        return

    # GTE's old initializer replaces the serialized rules with this empty template.
    empty_template = processors.TemplateProcessing(single="$A:0", pair="$A:0 $B:1")
    post_processor = tokenizer.backend_tokenizer.post_processor
    if (
        not isinstance(post_processor, processors.TemplateProcessing)
        or post_processor.__getstate__() != empty_template.__getstate__()
    ):
        return

    asset_kwargs = dict(
        revision=revision,
        cache_dir=download_dir,
        token=kwargs.get("token"),
        local_files_only=kwargs.get("local_files_only", False),
        subfolder=kwargs.get("subfolder", ""),
    )
    tokenizer_config = get_tokenizer_config(path_or_repo_id, **asset_kwargs)
    auto_map = tokenizer_config.get("auto_map", {})
    if isinstance(auto_map, dict):
        auto_map = auto_map.get("AutoTokenizer")
    if auto_map != [
        "tokenization_qwen.Qwen2Tokenizer",
        "tokenization_qwen.Qwen2TokenizerFast",
    ]:
        return

    tokenizer_file = kwargs.get("tokenizer_file") or cached_file(
        path_or_repo_id,
        "tokenizer.json",
        _raise_exceptions_for_missing_entries=False,
        **asset_kwargs,
    )
    if tokenizer_file is None:
        return
    saved_processor = Tokenizer.from_file(str(tokenizer_file)).post_processor
    if isinstance(saved_processor, processors.ByteLevel):
        if tokenizer_config.get("add_eos_token") is not True:
            return
        eos, eos_id = tokenizer.eos_token, tokenizer.eos_token_id
        if eos is None or eos_id is None:
            return
        saved_processor = processors.TemplateProcessing(
            single=["$A:0", f"{eos}:0"],
            pair=["$A:0", f"{eos}:0", "$B:1", f"{eos}:1"],
            special_tokens=[(eos, eos_id)],
        )
    if isinstance(saved_processor, processors.TemplateProcessing):
        tokenizer.backend_tokenizer.post_processor = saved_processor
        # Keep the custom class's EOS property consistent with its restored rules.
        tokenizer._add_eos_token = tokenizer.encode("")[-1:] == [tokenizer.eos_token_id]


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                cache_dir=download_dir,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI. If the "
                    "model was created with a newer version of "
                    "transformers, consider upgrading: "
                    "`uv pip install --upgrade transformers`"
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        _maybe_fix_gte_tokenizer(
            tokenizer,
            path_or_repo_id,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        return get_cached_tokenizer(tokenizer)
