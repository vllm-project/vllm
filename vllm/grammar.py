import collections
from copy import deepcopy, copy
from dataclasses import dataclass, fields
from functools import wraps, lru_cache
import regex
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, List, Set, Union

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

        @lru_cache(int(1e6))
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

        @lru_cache(int(1e6))
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


def memoize_by_instance(method):
    """
    Memoize by id(self) and fn args
    """
    mname = method.__name__

    @wraps(method)
    def wrapper(self, *args):
        key = (mname, id(self), args)
        if key in self._memo:
            return self._memo[key]
        result = method(self, *args)
        self._memo[key] = result
        return result

    return wrapper


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
    partial_token: str

    # orthogonal unique state key
    full_seq: str

    # function of key
    prior_terminal_ids: tuple[str]
    terminal_candidates: list

    # shared across instances
    _ignored_terms: set
    _seq_validator: dict
    _memo: dict
    _full_seq_trie: Trie

    def __repr__(self):
        shown = [
            "partial_token", "full_seq", "terminal_candidates",
            "prior_terminal_ids"
        ]
        attrs_str = ", ".join(f"{s}={repr(getattr(self, s))}" for s in shown)
        return f"{self.__class__.__name__}({attrs_str})"

    @classmethod
    @lru_cache(1000)
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
                     prior_terminal_ids=tuple(),
                     full_seq="",
                     partial_token="",
                     terminal_candidates=None,
                     _ignored_terms=set(lark_parser.lexer_conf.ignore),
                     _seq_validator=_seq_validator,
                     _memo={},
                     _full_seq_trie=Trie())
        parser._full_seq_trie.insert("", parser)
        return parser

    def new(self, **kwargs):
        """Cached create now state"""
        parser_state_key = (hash(kwargs["interactive_parser"]),
                            kwargs["partial_token"])
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
            parser = parser.step(remainder_seq)
            self._full_seq_trie.insert(full_seq, parser)
        return parser

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
            return self

        new_maybe_partial_token = self.partial_token + new_seq

        best_terminal, processed_seq, remainder_seq = self.get_best_matched_terminal(
            self.allowed_terminals, new_maybe_partial_token)
        if best_terminal is None:
            return None

        # candidate doesn't complete terminal
        if remainder_seq is None:
            partial_terminal_ids = self.get_partial_terminal_ids(
                self.allowed_terminals,
                new_maybe_partial_token,
            )
            return self.new(
                interactive_parser=self.interactive_parser,
                full_seq=self.full_seq + new_seq,
                prior_terminal_ids=self.prior_terminal_ids,
                partial_token=new_maybe_partial_token,
                terminal_candidates=partial_terminal_ids,
            )

        # terminal completes rule
        else:
            if best_terminal in self._ignored_terms:
                new_interactive_parser = self.interactive_parser
            else:
                new_interactive_parser = self.get_stepped_parser_state(
                    best_terminal)

            if self.partial_token:
                base_seq = self.full_seq[:-len(self.partial_token)]
            else:
                base_seq = self.full_seq

            new_parser = self.new(
                full_seq=base_seq + processed_seq,
                interactive_parser=new_interactive_parser,
                prior_terminal_ids=hash(
                    (self.prior_terminal_ids, best_terminal)),
                partial_token="",
                terminal_candidates=None,
            )

            # no leftover to process
            if remainder_seq == "":
                return new_parser

            # process remainder
            else:
                return new_parser.step(remainder_seq)

    def get_best_matched_terminal(self, checked_terminals, seq):
        for terminal in checked_terminals:
            processed_seq, remainder_seq = self._seq_validator[terminal](seq)
            if processed_seq:
                return terminal, processed_seq, remainder_seq

        return None, None, None

    def get_partial_terminal_ids(self, checked_terminals, seq):
        return set([
            term for term in checked_terminals
            if self._seq_validator[term](seq)[0] is not None
        ])

    @memoize_by_instance
    def get_stepped_parser_state(self, new_token_str):
        ip = copy(self.interactive_parser)
        ip.feed_token(Token(new_token_str, ''))
        return ip

    @memoize_by_instance
    def accepts(self):
        return set(self.interactive_parser.accepts()) | self._ignored_terms

    @property
    @memoize_by_instance
    def allowed_terminals(self):
        if self.terminal_candidates is not None:
            return tuple(sorted(self.terminal_candidates))
        return tuple(sorted(self.accepts()))

    @memoize_by_instance
    def is_valid_next_seq(self, new_seq: Optional[str]):
        if new_seq is None:
            return "$END" in self.allowed_terminals
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
        parser = self.root_parser[full_seq]
        if parser is None:
            return
        for tok_str in self.vocab:
            if parser.is_valid_next_seq(tok_str):
                yield tok_str

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

    def process_logits(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        return self.processor(token_ids, logits)


class RayRemoteGrammarLogitsProcessor:
    def __init__(self, *args, **kwargs):
        self.actor = GrammarLogitsProcessorActor.remote(*args, **kwargs)

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        logits_cpu = logits.cpu()
        result_id = self.actor.process_logits.remote(token_ids, logits_cpu)
        return ray.get(result_id)
