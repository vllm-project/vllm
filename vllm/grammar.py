import collections
from copy import deepcopy, copy
from dataclasses import dataclass, fields
from functools import wraps
import regex
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, List, Set, Union

from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState
from lark.lexer import Token, Pattern, PatternStr, PatternRE
from lark.exceptions import UnexpectedCharacters, UnexpectedToken


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

    def __copy__(self):
        return type(self)(
            self.parser,
            copy(self.parser_state),
            copy(self.lexer_thread),
        )


#########################################################################
#########################################################################


def get_pattern_validator(pattern: Pattern, is_complete: bool):
    """
    Accepts a pattern object, either lark.lexer.PatternStr or lark.lexer.PatternRE
    Returns a function which validates a complete or partial string

    e.g. for PatternRE "abc*", is_complete=False returns true for "a", "ab", "abc", "abcccc"
    """
    if isinstance(pattern, PatternRE):
        compiled_pattern = regex.compile(pattern.value)
        if is_complete:
            return (lambda seq: compiled_pattern.fullmatch(seq)
                is not None)
        else:
            return (lambda seq: compiled_pattern.fullmatch(seq, partial=True)
                is not None)
    elif isinstance(pattern, PatternStr):
        base_str = pattern.value
        if is_complete:
            return (lambda seq: seq == base_str)
        else:
            return (lambda seq: base_str.startswith(seq))
    else:
        raise TypeError(f"Invalid pattern type: {type(pattern)}")


def memoize_with_key(*key_attrs):
    def decorator(method):
        mname = method.__name__
        @wraps(method)
        def wrapper(self, *args):
            # Construct a simple key from key attributes and method arguments
            key_elements = tuple(getattr(self, attr, None) for attr in key_attrs)
            key = (mname, key_elements, args)

            # Check cache for existing result
            if key in self._memo:
                return self._memo[key]

            # Call the method and store the result
            result = method(self, *args)
            self._memo[key] = result
            return result

        return wrapper
    return decorator


@dataclass
class IncrementalParser:
    interactive_parser: FastInteractiveParser
    tokens_key: str  # "\n" separated, for caching purposes
    partial_token: str
    terminal_candidates: list
    _ignored_terms: set
    _seq_validator: dict
    _memo: dict

    @classmethod
    def from_lark_parser(cls, lark_parser):
        print(lark_parser.terminals)
        base_interactive_parser = lark_parser.parse_interactive()
        interactive_parser = FastInteractiveParser(
                base_interactive_parser.parser,
                base_interactive_parser.parser_state,
                base_interactive_parser.lexer_thread)
        interactive_parser.lexer_thread.state.text = ""

        _seq_validator = {
            (term.name, "partial"): get_pattern_validator(term.pattern, is_complete=False)
            for term in lark_parser.terminals
        }
        _seq_validator.update({
            (term.name, "complete"): get_pattern_validator(term.pattern, is_complete=True)
            for term in lark_parser.terminals
        })

        _seq_validator[("$END", "partial")] = lambda seq: seq is None
        _seq_validator[("$END", "complete")] = lambda seq: seq is None


        return cls(
            interactive_parser=interactive_parser,
            tokens_key="",
            partial_token="",
            terminal_candidates=None,
            _ignored_terms=set(lark_parser.lexer_conf.ignore),
            _seq_validator=_seq_validator,
            _memo={}
        )

    def new(self, **kwargs):
        instance_dict = {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }
        instance_dict.update(kwargs)
        return self.__class__(**instance_dict)

    @memoize_with_key('tokens_key', 'partial_token')
    def new_parser_for_appended_char(self, char: str):
        """
        - Construct extended (maybe-partial) token candidate
        - If no partial matches, None
        - If partial matches, but not complete,
              return new parser with updated partial token str and updated terminal candidates
        - If complete match, reset partial token, return parser with token-updated parser state
        """
        assert len(char) == 1

        new_maybe_partial_token = self.partial_token + char
        new_allowed_terminals = self.filter_terminals(
            self.allowed_terminals,
            new_maybe_partial_token,
            require_complete=False
        )
        if not new_allowed_terminals:
            return None

        complete_terminals = self.filter_terminals(
            self.allowed_terminals,
            new_maybe_partial_token,
            require_complete=True
        )
        if complete_terminals:
            assert len(complete_terminals) == 1
            new_token_str = next(iter(complete_terminals))
            return self.new(
                interactive_parser=self.get_stepped_parser_state(new_token_str),
                tokens_key=self.tokens_key + "\n" + new_token_str,
                partial_token="",
                terminal_candidates=None,
            )
        else:
            return self.new(
                partial_token=new_maybe_partial_token,
                terminal_candidates=new_allowed_terminals,
            )

    def filter_terminals(self, checked_terminals, seq, require_complete):
        validator_type = "complete" if require_complete else "partial"
        return set([
            term for term in checked_terminals
            if self._seq_validator[(term, validator_type)](seq)
        ])

    @memoize_with_key('tokens_key')
    def get_stepped_parser_state(self, new_token_str):
        ip = copy(self.interactive_parser)
        ip.feed_token(
            Token(new_token_str, '')
        )
        return ip

    @memoize_with_key('tokens_key')
    def accepts(self):
        return set(self.interactive_parser.accepts()) | self._ignored_terms

    @property
    def allowed_terminals(self):
        if self.terminal_candidates is not None:
            return self.terminal_candidates
        return self.accepts()


class SpeculativeParser:
    def __init__(self, grammar: str, start: str):
        self.parser = Lark(
            grammar,
            regex=True,  # use `regex` not `re`
            start=start,
            parser='lalr',
            cache=True,  # results in 2-3x faster loading
        )
        self.incr_parser = IncrementalParser.from_lark_parser(self.parser)
        self.fallback_incr_parser = copy(self.incr_parser)

    def step_seq(self, new_seq: str):
        """
        Append sequence to parser and apply state updates
        - Append the sequence to the canonical self.sequence_history
        - Parse the changes
        - Update the character position of the last complete terminal
        - Update the set of candidate terminals
        """
        for char in new_seq:
            new_incr_parser = self.incr_parser.new_parser_for_appended_char(char)
            if new_incr_parser is None:
                self.incr_parser = self.fallback_incr_parser
            else:
                self.incr_parser = new_incr_parser

    def is_valid_next_seq(self, new_seq: Optional[str]):
        if new_seq is None:
            return "$END" in self.incr_parser.allowed_terminals
        new_incr_parser = self.incr_parser
        for i, char in enumerate(new_seq):
            new_incr_parser = new_incr_parser.new_parser_for_appended_char(char)
            if new_incr_parser is None:
                return False
        return True


class TokenTrie:
    """
    Trie structure for efficiently finding tokens which are suffixes of other sequences
    """

    IS_TOKEN = (None, "is complete token")

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

        # faster lookups, reduce time by 10%
        self.norm_vocab_set = set(self.norm_vocab)

        self.trie = {}
        for word in self.norm_vocab:
            current_dict = self.trie
            if word is None:
                continue
            for char in word:
                if char not in current_dict:
                    current_dict[char] = {}
                current_dict = current_dict[char]
            current_dict[self.IS_TOKEN] = True

        self._next_level_token_prefixes_cache = {}

    def get_next_level_token_prefixes(self, subprefix: str) -> Set[str]:
        if subprefix not in self._next_level_token_prefixes_cache:
            self._next_level_token_prefixes_cache[subprefix] = (
                self.get_next_level_token_prefixes_uncached(subprefix))
        return self._next_level_token_prefixes_cache[subprefix]

    def get_next_level_token_prefixes_uncached(self,
                                               subprefix: str,
                                               _node: dict = None) -> Set[str]:
        """
        Traverse the trie starting from a specified subprefix to identify all child nodes that represent
        the longest possible strings without omitting any nodes that contain complete tokens.
        """
        # cache
        if _node is None and subprefix in self._next_level_token_prefixes_cache:
            return self._next_level_token_prefixes_cache[subprefix]

        # if not first level of recursion, and at a branching point or is a token, or return self
        if _node is not None and (len(_node) > 1 or self.IS_TOKEN in _node):
            return {subprefix}

        # get the current node if at the first level of recursion
        if _node is None:
            _node = self.trie
            for char in subprefix:
                if char not in _node:
                    return set()
                _node = _node[char]

        # Single child, need to go deeper
        results = set()
        for char, next_node in _node.items():
            if char != self.IS_TOKEN:
                results |= self.get_next_level_token_prefixes_uncached(
                    subprefix + char, _node=next_node)

        return results

    def is_token(self, seq: Optional[str]) -> bool:
        return seq in self.norm_vocab_set


class NextTokenValidator:
    """
    Given a grammar and a tokenset, construct a parser and token trie.

    Interface:
    - step_seq(new_seq): Append a sequence, update internal states
    - property valid_token_str_set: The valid set of vocabulary tokens strings which can occur next
    """

    def __init__(
        self,
        tokenizer,
        grammar: str,
        grammar_start: str = "start",
        legal_chars: Optional[set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.token_trie = TokenTrie(tokenizer, legal_chars=legal_chars)

        self.parser = SpeculativeParser(grammar=grammar,
                                        start=grammar_start)

    def step_seq(self, new_seq: str):
        self.parser.step_seq(new_seq)

    @property
    def valid_token_str_set(self):
        """
        Generate the set of valid tokens given the current sequence

        1) Push all first level token prefixes to the stack
        2) for each token in the stack, validate against the parser
          - if valid, add all children to the stack for later processing
          - if valid AND a token, add to valid_token_set

        TODO: this can be improved with multi-threading
        """
        valid_token_str_set = set()
        if self.parser.is_valid_next_seq(None):
            valid_token_str_set.add(None)
        token_prefix_stack = collections.deque([""])
        while token_prefix_stack:
            token_prefix = token_prefix_stack.pop()
            for child_token_prefix in self.token_trie.get_next_level_token_prefixes(
                    token_prefix):
                if self.parser.is_valid_next_seq(child_token_prefix):
                    token_prefix_stack.append(child_token_prefix)
                    if self.token_trie.is_token(child_token_prefix):
                        valid_token_str_set.add(child_token_prefix)

        return valid_token_str_set

    @property
    def valid_token_id_set(self):
        """
        get valid token id based on self.valid_token_str_set
        note that some token strings correspond to multiple token IDs
        """
        return set.union(*[
            self.token_trie.norm_vocab[tok_str]
            for tok_str in self.valid_token_str_set
        ])


# TODO: replace with subclass called NextTokenIDValidator to make things cleaner
@dataclass
class BatchDataItemParser:
    text: str
    token_ids: List[str]
    parser: NextTokenValidator


class GrammarLogitsProcessor:
    """
    Apply NextTokenValidator in __call__ and set excluded tokens logits to -inf
    """

    def __init__(
        self,
        tokenizer,
        grammar: str,
        grammar_start: str = "start",
        legal_chars: Optional[set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.grammar = grammar
        self.grammar_start = grammar_start
        self.legal_chars = legal_chars

        # track multiple parsers for batch requests
        self.batch_data_item_parsers: List[BatchDataItemParser] = []

    def _new_batch_data_item_parser(self):
        return BatchDataItemParser(
            "", [],
            NextTokenValidator(tokenizer=self.tokenizer,
                               grammar=self.grammar,
                               grammar_start=self.grammar_start,
                               legal_chars=self.legal_chars))

    def _get_batch_data_item_parser(self, token_ids: List[int]):
        """
        Get longest batch data item parser which matches the seen tokens.
        This is generally the corresponding parser, but if there's a collision
        their parsers are interchangable
        """
        for bdip in sorted(self.batch_data_item_parsers,
                           key=lambda bdip: -len(bdip.token_ids)):
            if token_ids[:len(bdip.token_ids)] == bdip.token_ids:
                return bdip

        # no match, make new
        return self._new_batch_data_item_parser()

    def _update_seen_token_ids(self, bdip: BatchDataItemParser,
                               token_ids: List[int]):

        # update batch item token tracker
        bdip.token_ids = token_ids

        # step forward
        all_text = self.tokenizer.decode(token_ids)
        new_text = all_text[len(bdip.text):]
        bdip.text = all_text
        bdip.parser.step_seq(new_text)

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        # get the batch item data and parser for batch item, given provided token sequence
        bdip = self._get_batch_data_item_parser(token_ids)

        self._update_seen_token_ids(bdip, token_ids)

        # modify logits given valid token IDs
        N = len(logits)
        mask = torch.zeros(N, dtype=torch.bool)
        valid = torch.tensor(list(bdip.parser.valid_token_id_set),
                             dtype=torch.long)
        mask[valid] = True
        logits[~mask] = float('-inf')
        return logits
