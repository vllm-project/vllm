# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class _XgrammarDraftTree:
    tree_choices: tuple[tuple[int, ...], ...]
    _topology_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _branch_indices_cache: dict[int, list[tuple[int, ...] | None]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def num_draft_tokens(self) -> int:
        return len(self.tree_choices)

    def topology(self, prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return xgrammar child/sibling topology for the requested prefix."""
        if prefix_len not in self._topology_cache:
            num_nodes = prefix_len + 1
            retrieve_next_token = [-1] * num_nodes
            retrieve_next_sibling = [-1] * num_nodes
            path_to_node = self._path_to_node(prefix_len)
            children_by_parent: dict[int, list[int]] = {}

            for node_idx, path in enumerate(self.tree_choices[:prefix_len], start=1):
                parent_path = path[:-1]
                if parent_path:
                    parent_idx = path_to_node.get(parent_path)
                    if parent_idx is None:
                        continue
                else:
                    parent_idx = 0
                children_by_parent.setdefault(parent_idx, []).append(node_idx)

            for parent_idx, children in children_by_parent.items():
                retrieve_next_token[parent_idx] = children[0]
                for left, right in zip(children, children[1:]):
                    retrieve_next_sibling[left] = right
            self._topology_cache[prefix_len] = (
                torch.tensor(retrieve_next_token, dtype=torch.int64),
                torch.tensor(retrieve_next_sibling, dtype=torch.int64),
            )
        return self._topology_cache[prefix_len]

    def branch_token_indices(self, prefix_len: int) -> list[tuple[int, ...] | None]:
        """Return draft-token indices needed to reach each node."""
        if prefix_len not in self._branch_indices_cache:
            branch_indices: list[tuple[int, ...] | None] = [()] + [None] * prefix_len
            path_to_node = self._path_to_node(prefix_len)
            for node_idx, path in enumerate(self.tree_choices[:prefix_len], start=1):
                indices: list[int] = []
                for depth in range(1, len(path) + 1):
                    ancestor_idx = path_to_node.get(path[:depth])
                    if ancestor_idx is None:
                        indices = []
                        break
                    indices.append(ancestor_idx - 1)
                else:
                    branch_indices[node_idx] = tuple(indices)
            self._branch_indices_cache[prefix_len] = branch_indices
        return self._branch_indices_cache[prefix_len]

    def _path_to_node(self, prefix_len: int) -> dict[tuple[int, ...], int]:
        return {
            path: idx
            for idx, path in enumerate(self.tree_choices[:prefix_len], start=1)
        }


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )

        if is_mistral_tokenizer(self.tokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            stop_token_ids = [self.tokenizer.eos_token_id]

            # not self.tokenizer.vocab_size as self.tokenizer.vocab
            # collapses all decoded errors into a single token.
            self.vocab_size = len(self.tokenizer.vocab)
            tokenizer_info = xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=self.tokenizer.vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                vocab_type=xgr.VocabType.RAW
                if self.tokenizer.is_tekken
                else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens or 0
            )
        self.draft_tree = _XgrammarDraftTree(
            tree_choices=self.vllm_config.speculative_config.speculative_token_tree
        )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            if "structures" in s_tag:
                # Falling back to deprecated method of compiling structural tag
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])
            else:
                ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
            draft_tree=self.draft_tree,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    draft_tree: _XgrammarDraftTree | None = field(default=None, repr=False, hash=False)
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, batch_index)

    def fill_speculative_bitmask(
        self,
        bitmask: torch.Tensor,
        batch_index: int,
        tokens: list[int],
        apply_bitmask: bool,
    ) -> bool:
        if self.draft_tree is None:
            return False
        if len(tokens) > self.draft_tree.num_draft_tokens:
            return False

        known_len = _get_speculative_prefix_len(tokens)
        num_rows = len(tokens) + 1
        rows = bitmask[batch_index : batch_index + num_rows]
        rows.fill_(-1)
        if not apply_bitmask or self.is_terminated():
            return True

        if known_len == 0:
            self.matcher.fill_next_token_bitmask(bitmask, batch_index)
            return True

        token_bitmask = bitmask[batch_index : batch_index + known_len + 1]
        draft_tokens = torch.empty(known_len + 1, dtype=torch.int64)
        draft_tokens[0] = 0
        draft_tokens[1:] = torch.tensor(tokens[:known_len], dtype=torch.int64)

        retrieve_next_token, retrieve_next_sibling = self.draft_tree.topology(known_len)
        try:
            traverse_completed = self.matcher.fork().traverse_draft_tree(
                retrieve_next_token,
                retrieve_next_sibling,
                draft_tokens,
                token_bitmask,
                -1.0,
            )
        except RuntimeError:
            traverse_completed = False
        if traverse_completed:
            return True

        self._fill_speculative_bitmask_with_forks(
            token_bitmask,
            tokens[:known_len],
            known_len,
        )
        return True

    def _fill_speculative_bitmask_with_forks(
        self,
        bitmask: torch.Tensor,
        tokens: list[int],
        prefix_len: int,
    ) -> None:
        assert self.draft_tree is not None
        self.matcher.fill_next_token_bitmask(bitmask, 0)
        branch_token_indices = self.draft_tree.branch_token_indices(prefix_len)
        for node_idx in range(1, prefix_len + 1):
            token_indices = branch_token_indices[node_idx]
            if token_indices is None:
                bitmask[node_idx].fill_(-1)
                continue

            matcher = self.matcher.fork()
            accepted = True
            for token_idx in token_indices:
                if not matcher.accept_token(tokens[token_idx]):
                    accepted = False
                    break

            if accepted:
                matcher.fill_next_token_bitmask(bitmask, node_idx)
            else:
                bitmask[node_idx].fill_(-1)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()


def _get_speculative_prefix_len(tokens: list[int]) -> int:
    for idx, token in enumerate(tokens):
        if token == -1:
            return idx
    return len(tokens)


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # Unsupported keywords for strings
        if (
            obj.get("type") == "string"
            and "format" in obj
            and obj["format"] not in STRING_SUPPORTED_FORMATS
        ):
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(
            key in obj for key in ("patternProperties", "propertyNames")
        ):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        try:
            xgr.Grammar.from_regex(so_params.regex)
        except Exception as err:
            raise ValueError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(
                f"Failed to transform choices into a grammar: {err}"
            ) from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar supports EBNF grammars only
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. "
                ) from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    if so_params.structural_tag:
        try:
            s_tag = json.loads(so_params.structural_tag)

            # Using the deprecated method of compiling structural tag
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])
            else:
                xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError("Invalid structural tag specification.") from e
