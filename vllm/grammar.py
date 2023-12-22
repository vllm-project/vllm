import collections
from copy import deepcopy, copy
from dataclasses import dataclass
import regex
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, List, Set, Union

from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState
from lark.lexer import Pattern, PatternStr, PatternRE
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


class InteractivePredictiveLALRParser:
    """
    Parser which consumes an EBNF grammar and provides helpers to determine allowable language model tokens

    Interfaces:
    - step_seq(new_seq): Update the parser with a new sequence to append
    - is_valid_next_seq(new_seq): Determine whether a candidate sequence is valid

    Core components for terminal level, and sub-terminal level processing:
    - 1) Lark LALR parser: Applies state transitions, determining set of valid next-terminals
    - 2) Incremental terminal filter: Eliminates next-terminal candidates if terminal pattern doesn't match
    """

    def __init__(self, grammar: str, start: str):
        self.parser = Lark(
            grammar,
            regex=True,  # use `regex` not `re`
            start=start,
            parser='lalr',
            cache=True,  # results in 2-3x faster loading
        )
        base_interactive_parser = self.parser.parse_interactive()
        self.interactive_parser = FastInteractiveParser(
            base_interactive_parser.parser,
            base_interactive_parser.parser_state,
            base_interactive_parser.lexer_thread)

        # fallback parser from start of terminal in case of ambiguous (LR(1))
        self._terminal_start_parser = self.interactive_parser.copy()

        self.partial_seq_validator = {
            term.name: self._get_partial_pattern_validator(term.pattern)
            for term in self.parser.terminals
        }

        self._ignored_terms = set(self.parser.lexer_conf.ignore)

        # for calculating `accepts()` efficiently
        self._accepts_cache = {}

        self.sequence_history = ""

        # for processing terminals interactively
        self.valid_next_terminals = {"": self._accepts() | self._ignored_terms}

    @staticmethod
    def _get_partial_pattern_validator(pattern: Pattern):
        """
        Accepts a pattern object, either lark.lexer.PatternStr or lark.lexer.PatternRE
        Returns a function which validates a partial string

        e.g. for PatternRE "abc*", returns true for "a", "ab", "abc", "abcccc"
        """
        if isinstance(pattern, PatternRE):
            compiled_pattern = regex.compile(pattern.value)
            return (lambda seq: compiled_pattern.fullmatch(seq, partial=True)
                    is not None)
        elif isinstance(pattern, PatternStr):
            base_str = pattern.value
            return (lambda seq: base_str.startswith(seq))
        else:
            raise TypeError(f"Invalid pattern type: {type(pattern)}")

    def _accepts(self):
        if self.sequence_history not in self._accepts_cache:
            accepted_terminals = self.interactive_parser.accepts()
            self._accepts_cache[self.sequence_history] = accepted_terminals
        return self._accepts_cache[self.sequence_history]

    def step_seq(self, new_seq: str):
        """
        Append sequence to parser and apply state updates
        - Append the sequence to the canonical self.sequence_history
        - Parse the changes
        - Update the character position of the last complete terminal
        - Update the set of candidate terminals
        """
        for char in new_seq:
            # update canonical sequence and lexer sequence
            self.sequence_history += char
            self.interactive_parser.lexer_thread.state.text = self.sequence_history

            success = False
            try:
                self.interactive_parser.exhaust_lexer()
            except UnexpectedCharacters:
                pass
            except UnexpectedToken:
                # fall back so full token can be reprocessed
                self.interactive_parser = self._terminal_start_parser.copy()
            else:
                success = True

            self.valid_next_terminals = {
                (incomplete_seq + char): term
                for incomplete_seq, term in self.valid_next_terminals.items()
            }

            # if successfully parsed new token, add blank state and set fallback checkpoint
            if success:
                self.valid_next_terminals[""] = self._accepts(
                ) | self._ignored_terms
                self._terminal_start_parser = self.interactive_parser.copy()

            self._filter_candidate_terminals()

            if not self.valid_next_terminals:
                raise ValueError(
                    f"Invalid continuation for `{self.sequence_history}` `{new_seq}`"
                )

    def _filter_candidate_terminals(self):
        """
        Filter the set of candidate terminals
        - If a new terminal is reached, get the accepted set of terminals from the parser
        - If the new sequence doesn't comprise a full terminal, filter based on partial pattern match

        Handles ambiguity by allowing terminals which are potentially complete
        """
        to_prune_sequences = set()
        for incomplete_seq, terminals in self.valid_next_terminals.items():
            if incomplete_seq != "":
                self.valid_next_terminals[incomplete_seq] = set([
                    term for term in terminals if term != "$END"
                    and self.partial_seq_validator[term](incomplete_seq)
                ])
            if not self.valid_next_terminals[incomplete_seq]:
                to_prune_sequences.add(incomplete_seq)

        for to_prune_seq in to_prune_sequences:
            del self.valid_next_terminals[to_prune_seq]

    def is_valid_next_seq(self, new_seq: Optional[str]):
        """
        Check if current un-terminalized sequence + new_seq is valid for any terminal

        new_seq can be a string or None representing EOS
        """
        if new_seq is None:
            return "$END" in [
                term for terminals in self.valid_next_terminals.values()
                for term in terminals
            ]
        for incomplete_seq, terminals in self.valid_next_terminals.items():
            candidate = incomplete_seq + new_seq
            for term in terminals:
                if term != "$END" and self.partial_seq_validator[term](
                        candidate):
                    return True
        return False


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
        if _node is None:
            if subprefix in self._next_level_token_prefixes_cache:
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

        self.parser = InteractivePredictiveLALRParser(grammar=grammar,
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
