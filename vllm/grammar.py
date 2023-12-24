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
            # False: No match
            # int: match length
            def get_fullmatch_length(seq):
                r = compiled_pattern.match(seq)
                if r is None or not r.spans():
                    return None
                spans = r.spans()[0]
                return spans[1] - spans[0]
            return get_fullmatch_length
        else:
            return (lambda seq: compiled_pattern.fullmatch(seq, partial=True)
                is not None)
    elif isinstance(pattern, PatternStr):
        base_str = pattern.value
        if is_complete:
            def get_strmatch_length(seq):
                if not seq.startswith(base_str):
                    return None
                return len(base_str)
            return get_strmatch_length
        else:
            return (lambda seq: base_str.startswith(seq))
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


@dataclass
class IncrementalParserState:
    """
    Parsing utility which tracks state provided
    - sequence of prior terminal ids
    - incomplete `partial_token` string
    the set of prior terminal_ids and a partial token comprise a unique parser state

    Core function exposed is `self.new_parser_for_appended(new_seq)`
    - Returns a new IncrementalParserState based with new_seq applied

    Memoization strategy is
    - 1) Ensure uniqueness of (prior_terminal_ids, partial_token)
    - 2) Cache class methods via `memoize_by_instance` which considers id(self) and fn arguments
    """

    # unique state key
    prior_terminal_ids: tuple[str]
    partial_token: str

    # function of key
    interactive_parser: FastInteractiveParser
    terminal_candidates: list

    # shared across instances
    _ignored_terms: set
    _seq_validator: dict
    _memo: dict

    @classmethod
    def from_lark_parser(cls, lark_parser):
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
            prior_terminal_ids=tuple(),
            partial_token="",
            terminal_candidates=None,
            _ignored_terms=set(lark_parser.lexer_conf.ignore),
            _seq_validator=_seq_validator,
            _memo={}
        )

    def new(self, prior_terminal_ids, partial_token, **kwargs):
        # cache
        key = (prior_terminal_ids, partial_token)
        if key in self._memo:
            return self._memo[key]
        instance_dict = {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }
        instance_dict.update(kwargs)
        inst = self.__class__(**instance_dict)
        self._memo[key] = inst
        return inst

    @memoize_by_instance
    def new_parser_for_appended(self, new_seq: str):
        """
        - Construct extended (maybe-partial) token candidate
        - If complete match, create new-terminal incremented parser state
          - there is leftover from new_seq, recurse on the new parser
        - If partial matches,
              return new parser with updated partial token str and updated terminal candidates
        - If no partial matches, return None
        """
        new_maybe_partial_token = self.partial_token + new_seq

        complete_terminal = self.get_complete_terminal(
            tuple(sorted(self.allowed_terminals)),
            new_maybe_partial_token,
        )
        if complete_terminal is not None:
            terminal_name = complete_terminal["terminal_id"]
            if terminal_name in self._ignored_terms:
                new_interactive_parser = self.interactive_parser
            else:
                new_interactive_parser = self.get_stepped_parser_state(terminal_name)
            new_parser = self.new(
                interactive_parser=new_interactive_parser,
                prior_terminal_ids=tuple(list(self.prior_terminal_ids) + [terminal_name]),
                partial_token="",
                terminal_candidates=None,
            )

            leftover = len(new_seq) - complete_terminal["match_length"]
            if leftover:
                return new_parser.new_parser_for_appended(
                    new_maybe_partial_token[-leftover:]
                )
            else:
                return new_parser

        partial_terminal_ids = self.get_partial_terminal_ids(
            tuple(sorted(self.allowed_terminals)),
            new_maybe_partial_token,
        )
        if partial_terminal_ids:
            return self.new(
                prior_terminal_ids=self.prior_terminal_ids,
                partial_token=new_maybe_partial_token,
                terminal_candidates=partial_terminal_ids,
            )
        else:
            return None


    def get_complete_terminal(self, checked_terminals, seq):
        terminal_matchlens = {
            term: self._seq_validator[(term, "complete")](seq)
            for term in checked_terminals
        }
        terminal_matchlens = {term: ml for term, ml in terminal_matchlens.items() if ml}
        if not terminal_matchlens:
            return None
        if len(terminal_matchlens) > 1:
            terminal_matchlens = {
                t: ml for t, ml in terminal_matchlens.items()
                if t not in self._ignored_terms
            }
        assert len(terminal_matchlens) == 1
        result = next(iter(terminal_matchlens.items()))
        return {"terminal_id": result[0], "match_length": result[1]}

    def get_partial_terminal_ids(self, checked_terminals, seq):
        return set([
            term for term in checked_terminals
            if self._seq_validator[(term, "partial")](seq)
        ])

    @memoize_by_instance
    def get_stepped_parser_state(self, new_token_str):
        ip = copy(self.interactive_parser)
        ip.feed_token(
            Token(new_token_str, '')
        )
        return ip

    @memoize_by_instance
    def accepts(self):
        return set(self.interactive_parser.accepts()) | self._ignored_terms

    @property
    @memoize_by_instance
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
        self.incr_parser = IncrementalParserState.from_lark_parser(self.parser)
        self.fallback_incr_parser = self.incr_parser

    def step_seq(self, new_seq: str):
        """
        Append sequence to parser and apply state updates
        - Append the sequence to the canonical self.sequence_history
        - Parse the changes
        - Update the character position of the last complete terminal
        - Update the set of candidate terminals
        """
        new_incr_parser = self.incr_parser.new_parser_for_appended(new_seq)
        if new_incr_parser is None:
            self.incr_parser = self.fallback_incr_parser
        else:
            self.incr_parser = new_incr_parser

    def is_valid_next_seq(self, new_seq: Optional[str]):
        if new_seq is None:
            return "$END" in self.incr_parser.allowed_terminals
        return self.incr_parser.new_parser_for_appended(new_seq) is not None
        if new_incr_parser is None:
            return False
        return True


class TokenVocab:
    """
    Normalized token vocabulary accounting for whitespace and multiple IDs per token
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

    def __get__(self, tok_str):
        return self.norm_vocab[tok_str]


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
        self.vocab = TokenVocab(tokenizer, legal_chars=legal_chars)

        self.parser = SpeculativeParser(grammar=grammar,
                                        start=grammar_start)

    def step_seq(self, new_seq: str):
        self.parser.step_seq(new_seq)

    @property
    def valid_token_str_set(self):
        """
        Generate the set of valid tokens given the current sequence
        """
        for tok in self.vocab:
            if self.parser.is_valid_next_seq(tok):
                yield tok

    @property
    def valid_token_id_set(self):
        """
        get valid token id based on self.valid_token_str_set
        note that some token strings correspond to multiple token IDs
        """
        return set.union(*[
            self.vocab[tok_str]
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
