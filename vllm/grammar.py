import collections
from copy import deepcopy, copy
from dataclasses import dataclass, fields
import functools
import regex
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, List, Set, Union
import weakref

import ray

from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState
from lark.lexer import Token, Pattern, PatternStr, PatternRE


#########################################################################
# Fix Lark Interactive LALR Parser Speed Issue
# https://github.com/lark-parser/lark/issues/1142#issuecomment-1863209804
#########################################################################
class FastParserState(ParserState):
    copy_memo = {}

    def __copy__(self):
        new_value_stack = []
        for value in self.value_stack:
            key = f"{id(self)}_{id(value)}"
            if key not in self.copy_memo:
                self.copy_memo[key] = deepcopy(value, self.copy_memo)
            new_value_stack.append(self.copy_memo[key])

        new_instance = type(self)(
            self.parse_conf,
            self.lexer,
            copy(self.state_stack),
            new_value_stack,
        )

        self.copy_memo[id(self)] = new_instance
        return new_instance


class FastInteractiveParser(InteractiveParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser_state = FastParserState(
            self.parser_state.parse_conf,
            self.parser_state.lexer,
            self.parser_state.state_stack,
            self.parser_state.value_stack,
        )
        self.hash_val = None

    def __hash__(self):
        if self.hash_val is None:
            self.hash_val = hash(tuple(self.parser_state.state_stack))
        return self.hash_val

    def __copy__(self):
        return type(self)(
            self.parser,
            copy(self.parser_state),
            copy(self.lexer_thread),
        )


#########################################################################
#########################################################################


def get_pattern_validator(pattern: Pattern):
    """
    Accepts a pattern object, either lark.lexer.PatternStr or lark.lexer.PatternRE
    Returns a function which validates a complete or partial string

    Returns Tuple with 2 values
    - 0) The processed portion of the sequence (None if no match at all)
    - 1) None if doesn't complete terminal, "" if completes terminal with no remainder, or "remainder"
    """
    if isinstance(pattern, PatternRE):
        compiled_pattern = regex.compile(pattern.value)

        @functools.lru_cache(int(1e6))
        def get_re_matched_parts(seq):
            # match complete terminal, potentially with leftover seq
            complete_terminal_match = compiled_pattern.match(seq)
            if complete_terminal_match:
                spans = complete_terminal_match.spans()
                if spans:
                    span = complete_terminal_match.spans()[0]
                    if span[0] == 0:
                        processed_seq = seq[:span[1]]
                        remainder_seq = seq[span[1]:]
                        return processed_seq, remainder_seq

            # match doesn't complete terminal, but the sequence is fully allowed
            partial_terminal_match = compiled_pattern.fullmatch(seq,
                                                                partial=True)
            if partial_terminal_match:
                return seq, None

            return None, None

        return get_re_matched_parts

    elif isinstance(pattern, PatternStr):
        base_str = pattern.value

        @functools.lru_cache(int(1e6))
        def get_str_matched_parts(seq):
            if seq.startswith(base_str):
                processed_seq = seq[:len(base_str)]
                remainder_seq = seq[len(base_str):]
                return processed_seq, remainder_seq
            elif base_str.startswith(seq):
                return seq, None
            else:
                return None, None

        return get_str_matched_parts

    else:
        raise TypeError(f"Invalid pattern type: {type(pattern)}")


def method_lru_cache(*lru_args, **lru_kwargs):
    # https://stackoverflow.com/a/44078118
    def decorator(func):

        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


memoize_by_instance = method_lru_cache(int(1e7))


class TrieNode:

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, key, value):
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.value = value

    def get_best(self, word):
        node = self.root
        prefix = ""
        best_prefix = ""
        best_value = node.value
        for char in word:
            if char in node.children:
                node = node.children[char]
                prefix += char
                if node.is_end_of_word:
                    best_prefix = prefix
                    best_value = node.value
            else:
                break  # break if char not in trie

        remainder = word[len(best_prefix):]
        return best_prefix, best_value, remainder


@dataclass
class IncrementalParserState:
    """
    Parsing utility which enforces uniqueness of
    - interactive parser state stack
    - incomplete `partial_token` string
    the state of the parser and the incomplete token comprise a unique parser state

    Core function exposed is `self.step(new_seq)`
    - Returns a new IncrementalParserState based with new_seq applied

    Memoization strategy is
    - 1) Ensure uniqueness of (interactive_parser, partial_token)
    - 2) Cache class methods via `memoize_by_instance` which considers id(self) and fn arguments
    """

    # unique state key
    interactive_parser: FastInteractiveParser

    # function of key
    terminal_candidates: list

    # shared across instances
    _ignored_terms: set
    _seq_validator: dict
    _memo: dict
    _full_seq_trie: Trie

    def __repr__(self):
        return f"{self.__class__.__name__}({self.interactive_parser.parser_state.state_stack})"

    @classmethod
    @functools.lru_cache(1000)
    def from_grammar(cls, grammar: str, start: str):
        lark_parser = Lark(
            grammar,
            regex=True,  # use `regex` not `re`
            start=start,
            parser='lalr',
            cache=True,  # results in 2-3x faster loading
        )
        base_interactive_parser = lark_parser.parse_interactive()
        interactive_parser = FastInteractiveParser(
            base_interactive_parser.parser,
            base_interactive_parser.parser_state,
            base_interactive_parser.lexer_thread)
        interactive_parser.lexer_thread.state.text = ""

        _seq_validator = {(term.name): get_pattern_validator(term.pattern)
                          for term in lark_parser.terminals}
        _seq_validator["$END"] = lambda seq: tuple(
            ["" if seq is None else None] * 2)

        parser = cls(interactive_parser=interactive_parser,
                     terminal_candidates=None,
                     _ignored_terms=set(lark_parser.lexer_conf.ignore),
                     _seq_validator=_seq_validator,
                     _memo={},
                     _full_seq_trie=Trie())
        parser._full_seq_trie.insert("", parser)
        return parser

    def new(self, **kwargs):
        """Cached create now state"""
        parser_state_key = hash(kwargs["interactive_parser"])
        if parser_state_key in self._memo:
            return self._memo[parser_state_key]

        instance_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        instance_dict.update(kwargs)
        inst = self.__class__(**instance_dict)

        self._memo[parser_state_key] = inst

        return inst

    def __getitem__(self, full_seq):
        """Get the parser state, given a full sequence"""
        match_seq, parser, remainder_seq = self._full_seq_trie.get_best(
            full_seq)
        if parser is None:
            return
        if remainder_seq:
            result = parser.step(remainder_seq)
            if result is None:
                return None
            remainder_seq, parser = result
            processed_seq = full_seq
            if remainder_seq:
                processed_seq = processed_seq[:-len(remainder_seq)]
            self._full_seq_trie.insert(processed_seq, parser)
        return remainder_seq, parser

    @memoize_by_instance
    def step(self, new_seq: str):
        """
        - Construct extended (maybe-partial) token candidate
        - If complete match, create new-terminal incremented parser state
          - there is leftover from new_seq, recurse on the new parser
        - If partial matches,
              return new parser with updated partial token str and updated terminal candidates
        - If no partial matches, return None
        """
        if new_seq == "":
            return "", self

        best_terminal, processed_seq, remainder_seq = self.get_best_matched_terminal(
            new_seq)

        # invalid
        if best_terminal is None:
            return None

        # candidate doesn't complete terminal
        elif remainder_seq is None:
            return processed_seq, self

        # candidate completes terminal
        else:
            new_parser = self._next_with_new_terminal(best_terminal)
            if remainder_seq == "":
                return "", new_parser
            else:
                return new_parser.step(remainder_seq)

    @memoize_by_instance
    def _next_with_new_terminal(self, terminal):
        if terminal in self._ignored_terms:
            new_interactive_parser = self.interactive_parser
        else:
            new_interactive_parser = self.get_stepped_parser_state(terminal)

        return self.new(
            interactive_parser=new_interactive_parser,
            terminal_candidates=None,
        )

    def get_best_matched_terminal(self, seq):
        for terminal in self.accepts():
            processed_seq, remainder_seq = self._seq_validator[terminal](seq)
            if processed_seq:
                return terminal, processed_seq, remainder_seq

        return None, None, None

    @memoize_by_instance
    def get_stepped_parser_state(self, new_token_str):
        ip = copy(self.interactive_parser)
        ip.feed_token(Token(new_token_str, ''))
        return ip

    @memoize_by_instance
    def accepts(self):
        return set(self.interactive_parser.accepts()) | self._ignored_terms

    @memoize_by_instance
    def allowed_terminals(self):
        if self.terminal_candidates is not None:
            return tuple(sorted(self.terminal_candidates))
        return tuple(sorted(self.accepts()))

    @memoize_by_instance
    def is_valid_next_seq(self, new_seq: Optional[str]):
        if new_seq is None:
            return "$END" in self.allowed_terminals()
        return self.step(new_seq) is not None


class TokenVocab:
    """
    Normalized token vocabulary accounting for whitespace and multiple IDs per token
    - iter: iterate over normalized token strings
    - vocab[token_str]: return token id set
    """

    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer,
                                  PreTrainedTokenizerFast],
                 legal_chars: Optional[Set[str]] = None):

        self.norm_vocab = collections.defaultdict(set)
        for token_id in tokenizer.vocab.values():
            if token_id == tokenizer.eos_token_id:
                self.norm_vocab[None].add(token_id)
                continue
            bos_len = len(tokenizer.bos_token)
            norm_token = tokenizer.decode([tokenizer.bos_token_id,
                                           token_id])[bos_len:]
            if legal_chars is None or all(
                [char in legal_chars for char in norm_token]):
                self.norm_vocab[norm_token].add(token_id)

    def __iter__(self):
        return iter(self.norm_vocab)

    def __getitem__(self, tok_str):
        return self.norm_vocab[tok_str]


class NextTokenValidator:

    def __init__(
        self,
        tokenizer,
        grammar: str,
        grammar_start: str = "start",
        legal_chars: Optional[set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.vocab = TokenVocab(tokenizer, legal_chars=legal_chars)
        self.root_parser = IncrementalParserState.from_grammar(
            grammar, grammar_start)

    def get_valid_next_token_strs(self, full_seq):
        """
        Generate valid token strings given the full sequence
        """

        result = self.root_parser[full_seq]
        if result is None:
            return []
        partial_term, parser = result
        for token in self.vocab:
            if token is None:
                if partial_term == "" and parser.is_valid_next_seq(token):
                    yield None
            else:
                if parser.is_valid_next_seq(partial_term + token):
                    yield token

    def get_valid_next_token_ids(self, full_seq):
        """
        Generate valid token ids given the full sequence
        """
        for tok_str in self.get_valid_next_token_strs(full_seq):
            yield from self.vocab[tok_str]


class GrammarLogitsProcessor(NextTokenValidator):
    """
    Apply NextTokenValidator in __call__ and set excluded tokens logits to -inf
    """

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        # get valid token IDs given prior tokens
        sequence = self.tokenizer.decode(token_ids)
        valid_token_ids = self.get_valid_next_token_ids(sequence)
        valid = torch.tensor(list(valid_token_ids), dtype=torch.long)

        # modify logits given valid token IDs
        N = len(logits)
        mask = torch.zeros(N, dtype=torch.bool)
        mask[valid] = True
        logits[~mask] = float('-inf')
        return logits


@ray.remote
class GrammarLogitsProcessorActor:

    def __init__(self, *args, **kwargs):
        self.processor = GrammarLogitsProcessor(*args, **kwargs)

    def process_logits(self, token_ids: List[int],
                       logits: torch.Tensor) -> torch.Tensor:
        return self.processor(token_ids, logits)


class RayRemoteGrammarLogitsProcessor:

    def __init__(self, *args, **kwargs):
        self.actor = GrammarLogitsProcessorActor.remote(*args, **kwargs)

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        logits_cpu = logits.cpu()
        result_id = self.actor.process_logits.remote(token_ids, logits_cpu)
        return ray.get(result_id)
