import collections
from copy import deepcopy, copy
import functools
import os
import regex
from typing import Optional

from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState
from lark.lexer import Token, LexerState, PatternStr, PatternRE
from lark.exceptions import UnexpectedCharacters, UnexpectedToken



class FastParserState(ParserState):
    """
    https://github.com/lark-parser/lark/issues/1142#issuecomment-1863209804
    """
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
            self.lexer, # XXX copy
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


def get_partial_pattern_validator(pattern):
    """
    Accepts a pattern object, either lark.lexer.PatternStr or lark.lexer.PatternRE
    Returns a function which validates a partial string

    e.g. for PatternRE "abc*", returns true for "a", "ab", "abc", "abcccc"
    """
    if isinstance(pattern, PatternRE):
        compiled_pattern = regex.compile(pattern.value)
        return (
            lambda seq: compiled_pattern.fullmatch(seq, partial=True) is not None
        )
    elif isinstance(pattern, PatternStr):
        base_str = pattern.value
        return (
            lambda seq: base_str.startswith(seq)
        )
    else:
        raise TypeError(f"Invalid pattern type: {type(pattern)}")


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
        )
        base_interactive_parser = self.parser.parse_interactive()
        self.interactive_parser = FastInteractiveParser(
            base_interactive_parser.parser,
            base_interactive_parser.parser_state,
            base_interactive_parser.lexer_thread
        )

        self.partial_seq_validator = {
            term.name: get_partial_pattern_validator(term.pattern)
            for term in self.parser.terminals
        }

        self._ignored_terms = set(self.parser.lexer_conf.ignore)

        # for processing terminals interactively
        self.last_terminal_pos = 0
        self.valid_next_terminals = None

        # for calculating `accepts()` efficiently
        self._accepts_cache = {}

        self.sequence_history = ""

        # initiate
        self.step_seq("")

    def _accepts(self):
        if self.sequence_history not in self._accepts_cache:
            accepted_terminals = self.interactive_parser.accepts()
            self._accepts_cache[self.sequence_history] = accepted_terminals
        return self._accepts_cache[self.sequence_history]

    @property
    def terminal_partial_seq(self):
        """
        Return the incomplete subsequence which will eventually comprise a terminal
        """
        return self.sequence_history[self.last_terminal_pos:]

    def step_seq(self, new_seq: str):
        """
        Append sequence to parser and apply state updates
        - Append the sequence to the canonical self.sequence_history
        - Parse the changes
        - Update the character position of the last complete terminal
        - Update the set of candidate terminals
        """
        self._append_to_sequence(new_seq)

        try:
            self.interactive_parser.exhaust_lexer()
        except UnexpectedCharacters as e:
            self.last_terminal_pos = e.pos_in_stream
        else:
            self.last_terminal_pos = len(self.sequence_history)

        self._update_candidate_terminals()

        if not self.valid_next_terminals:
            raise ValueError(f"Invalid continuation for `{self.sequence_history}` `{sequence}`")

    def _append_to_sequence(self, new_seq: str):
        """Set the complete sequences value in the lexer and base"""
        self.sequence_history += new_seq
        self.interactive_parser.lexer_thread.state.text = self.sequence_history

    def _update_candidate_terminals(self):
        """
        Update the set of candidate terminals
        - If a new terminal is reached, get the accepted set of terminals from the parser
        - If the new sequence doesn't comprise a full terminal, filter based on partial pattern match
        """
        if not self.terminal_partial_seq:
            self.valid_next_terminals = self._accepts() | self._ignored_terms
        else:
            self.valid_next_terminals = set([
                term for term in self.valid_next_terminals
                if self.partial_seq_validator[term](self.terminal_partial_seq)
            ])

    def is_valid_next_seq(self, new_seq: Optional[str]):
        """
        Check if current un-terminalized sequence + new_seq is valid for any terminal

        new_seq can be a string or None representing EOS
        """
        if new_seq is None:
            return "$END" in self.valid_next_terminals
        for term in self.valid_next_terminals:
            if term != "$END":
                if self.partial_seq_validator[term](self.terminal_partial_seq + new_seq):
                    return True
        return False


class TokenTrie:
    IS_TOKEN = (None, "is complete token")

    def __init__(self, tokenizer, legal_chars: Optional[set[str]] = None):
        """
        Trie structure for efficiently finding tokens which are suffixes of other sequences
        """
        self.norm_vocab = {}
        for token_id in tokenizer.vocab.values():
            norm_token = tokenizer.decode([tokenizer.bos_token_id, token_id])[len(tokenizer.bos_token):]
            if legal_chars is None or all([char in legal_chars for char in norm_token]):
                self.norm_vocab[norm_token] = token_id

        self.trie = {}
        for word in self.norm_vocab:
            current_dict = self.trie
            for char in word:
                if char not in current_dict:
                    current_dict[char] = {}
                current_dict = current_dict[char]
            current_dict[self.IS_TOKEN] = True

    def get_next_level_token_prefixes(self, subprefix: str, _node=None):
        """
        Traverse the trie starting from a specified subprefix to identify all child nodes that represent
        the longest possible strings without omitting any nodes that contain complete tokens.
        """
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
                results |= self.get_next_level_token_prefixes(subprefix + char, _node=next_node)
        return results

    def is_token(self, seq):
        return seq in self.norm_vocab


class NextTokenValidator:
    """
    Given a grammar and a tokenset, construct a parser and token trie.

    Interface:
    - step_seq(new_seq): Append a sequence, update internal states
    - property valid_token_set: The valid set of tokens within the vocabulary that can occure next
    """
    def __init__(
            self,
            tokenizer,
            grammar: str,
            grammar_start: str = "start",
            num_threads: Optional[int] = None
    ):
        self.parser = InteractivePredictiveLALRParser(
            grammar=grammar,
            start=grammar_start
        )
        self.tokenizer = tokenizer
        self.token_trie = TokenTrie(tokenizer)

        if num_threads is None:
            self.num_threads = os.cpu_count() // 2

    def step_seq(self, new_seq):
        self.parser.step_seq(new_seq)

    @property
    def valid_token_set(self):
        """
        Generate the set of valid tokens given the current sequence

        1) Push all first level token prefixes to the stack
        2) for each token in the stack, validate against the parser
          - if valid, add all children to the stack
          - if valid AND a token, add to valid_token_set
        """
        valid_token_set = set()
        token_prefix_stack = collections.deque([""])
        while token_prefix_stack:
            print(len(token_prefix_stack))
            token_prefix = token_prefix_stack.pop()
            for child_token_prefix in self.token_trie.get_next_level_token_prefixes(token_prefix):
                # TODO: Handle EOS token by passing None
                if self.parser.is_valid_next_seq(child_token_prefix):
                    token_prefix_stack.append(child_token_prefix)
                    if self.token_trie.is_token(child_token_prefix):
                        valid_token_set.add(child_token_prefix)

        return valid_token_set



def test_next_token_validator_simple():
    grammar = """
    ?value: "hello" | "world"
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    ntv = NextTokenValidator(tokenizer, json_grammar, "value")

    valid_toks = ntv.valid_token_set
    assert valid_tokns == {'wo', 'hell', 'h', 'he', 'hel', 'world', 'wor', 'w', 'hello'}


def test_token_trie_sanity_hf_tokenizer():
    """Ensure token trie produces the same number of N 3 letter tokens"""
    tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    toktrie = TokenTrie(tokenizer)

    all_prefixes = toktrie.get_next_level_token_prefixes("")

    # every token should be composable from a single unique char, so they will all be len of 1
    assert all([len(p) == 1 for p in all_prefixes])

    # every token should have one of these prefixes as a start character
    assert all([
        t[0] in all_prefixes
        for t in toktrie.norm_vocab
    ])

    # construct the set of next level prefixes
    all_subprefixes = set()
    for pfx in all_prefixes:
        all_subprefixes |= toktrie.get_next_level_token_prefixes(pfx)

    import pdb;pdb.set_trace()

    # these should have varying length because some tokens don't have level-2 prefixes
    assert len(set([len(spfx) for spfx in all_subprefixes])) > 1


def test_simple_sequence(parser):
    for token in ['{', '"', 'k', 'ey', '":', '"val', 'ue', '"']:
        print("full:", parser.sequence_history)
        print("adding", token)
        parser.step_seq(token)
        print("partial:", parser.terminal_partial_seq)
        print("valid terms:", parser.valid_next_terminals)


def test_valid_next_tokens(parser):
    # random complicated json file courtesy of https://github.com/simdjson/simdjson/issues/1316#issue-748663718
    complex_json_file = '{"$schema": "http://json-schema.org/draft-04/schema#", "additionalProperties": false, "properties": {"nc:Vehicle": {"description": "A conveyance designed to carry an operator, passengers and/or cargo, over land.", "oneOf": [{"$ref": "#/definitions/nc:VehicleType"}, {"type": "array", "items": {"$ref": "#/definitions/nc:VehicleType"}}]}, "nc:VehicleAxleQuantity": {"description": "A count of common axles of rotation of one or more wheels of a vehicle, whether power driven or freely rotating.", "oneOf": [{"$ref": "#/definitions/niem-xs:nonNegativeInteger"}, {"type": "array", "items": {"$ref": "#/definitions/niem-xs:nonNegativeInteger"}}]}, "nc:VehicleMSRPAmount": {"description": "A manufacturer\'s suggested retail price of a vehicle; a price at which a manufacturer recommends a vehicle be sold.", "oneOf": [{"$ref": "#/definitions/nc:AmountType"}, {"type": "array", "items": {"$ref": "#/definitions/nc:AmountType"}}]}, "nc:Amount": {"description": "An amount of money.", "oneOf": [{"$ref": "#/definitions/niem-xs:decimal"}, {"type": "array", "items": {"$ref": "#/definitions/niem-xs:decimal"}}]}, "nc:Currency": {"description": "A data concept for a unit of money or exchange.", "oneOf": [{"anyOf": [{"$ref": "#/properties/nc:CurrencyCode"}]}, {"type": "array", "items": {"anyOf": [{"$ref": "#/properties/nc:CurrencyCode"}]}}]}, "nc:CurrencyCode": {"description": "A unit of money or exchange.", "oneOf": [{"$ref": "#/definitions/iso_4217:CurrencyCodeType"}, {"type": "array", "items": {"$ref": "#/definitions/iso_4217:CurrencyCodeType"}}]}, "nc:VehicleIdentification": {"description": "A unique identification for a specific vehicle.", "oneOf": [{"$ref": "#/definitions/nc:IdentificationType"}, {"type": "array", "items": {"$ref": "#/definitions/nc:IdentificationType"}}]}, "nc:IdentificationID": {"description": "An identifier.", "oneOf": [{"$ref": "#/definitions/niem-xs:string"}, {"type": "array", "items": {"$ref": "#/definitions/niem-xs:string"}}]}}, "definitions": {"nc:VehicleType": {"description": "A data type for a conveyance designed to carry an operator, passengers and/or cargo, over land.", "allOf": [{"$ref": "#/definitions/nc:ConveyanceType"}, {"type": "object", "properties": {"nc:VehicleAxleQuantity": {"$ref": "#/properties/nc:VehicleAxleQuantity"}, "nc:VehicleIdentification": {"$ref": "#/properties/nc:VehicleIdentification"}, "nc:VehicleMSRPAmount": {"$ref": "#/properties/nc:VehicleMSRPAmount"}}}]}, "nc:ConveyanceType": {"description": "A data type for a means of transport from place to place.", "allOf": [{"$ref": "#/definitions/_base"}, {"$ref": "#/definitions/nc:ItemType"}, {"type": "object", "properties": {}}]}, "nc:ItemType": {"description": "A data type for an article or thing.", "allOf": [{"$ref": "#/definitions/_base"}, {"type": "object", "properties": {}}]}, "nc:AmountType": {"description": "A data type for an amount of money.", "type": "object", "properties": {"nc:Amount": {"$ref": "#/properties/nc:Amount"}, "nc:Currency": {"$ref": "#/properties/nc:Currency"}}}, "iso_4217:CurrencyCodeType": {"description": "A data type for a currency that qualifies a monetary amount.", "oneOf": [{"$ref": "#/definitions/iso_4217:CurrencyCodeSimpleType"}, {"type": "object", "properties": {"rdf:value": {"$ref": "#/definitions/iso_4217:CurrencyCodeSimpleType"}}}]}, "iso_4217:CurrencyCodeSimpleType": {"type": "string", "description": "A data type for a currency that qualifies a monetary amount.", "oneOf": [{"enum": ["EUR"], "description": "Euro"}, {"enum": ["GBP"], "description": "Pound Sterling"}, {"enum": ["USD"], "description": "US Dollar"}]}, "nc:IdentificationType": {"description": "A data type for a representation of an identity.", "type": "object", "properties": {"nc:IdentificationID": {"$ref": "#/properties/nc:IdentificationID"}}}, "niem-xs:decimal": {"description": "A data type for arbitrary precision decimal numbers.", "type": "number"}, "niem-xs:nonNegativeInteger": {"description": "A data type for an integer with a minimum value of 0.", "type": "number"}, "niem-xs:string": {"description": "A data type for character strings in XML.", "type": "string"}, "_base": {"type": "object", "patternProperties": {"^ism:.*": {"type": "string"}, "^ntk:.*": {"type": "string"}}, "properties": {"@id": {"format": "uriref"}, "@base": {"format": "uriref"}}}}}'

    test_chars_per_iter = 1000
    unicode_chars = [chr(i) for i in range(test_chars_per_iter)]

    import time
    start = time.time()
    for char in complex_json_file:
        parser.step_seq(char)
        for ch in unicode_chars:
            parser.is_valid_next_seq(ch)

    print("took",
          (time.time() - start) /  (len(complex_json_file)),
          "seconds per step with",
          test_chars_per_iter, "characters in vocabulary")


def main():
    # Usage
    json_grammar = """
    ?value: dict
          | list
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : string ":" value

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
    """

    parser = InteractivePredictiveLALRParser(json_grammar, 'value')
    test_valid_next_tokens(parser)


if __name__ == "__main__":
    import transformers
    test_next_token_validator()
    import sys
    sys.exit()

    profile = True
    if profile:
        import cProfile
        import pstats
        from io import StringIO
        profile = cProfile.Profile()
        profile.enable()
        main()
        profile.disable()

        # Sorting the statistics by cumulative time
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    else:
        main()
