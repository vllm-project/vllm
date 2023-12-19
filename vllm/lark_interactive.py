from typing import Optional

from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState
from lark.lexer import Token, LexerState, PatternStr, PatternRE
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

import regex

from copy import deepcopy, copy
import functools


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


@functools.lru_cache(10000)
def check_pattern_partial_match(compiled_pattern, seq):
    return


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
        return (
            lambda seq: pattern.value.startswith(seq)
        )
    else:
        raise TypeError(f"Invalid pattern type: {type(pattern)}")


class InteractivePredictiveLALRParser:
    """
    Parser which consumes an EBNF grammar and provides helpers to determine allowable language model tokens

    Interfaces:
    - step_seq(sequence): Update the parser with a new sequence to append
    - is_valid_next_seq(sequence): Determine whether a candidate sequence is valid

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

    def step_seq(self, sequence: str):
        """
        Append sequence to parser and apply state updates
        - Append the sequence to the canonical self.sequence_history
        - Parse the changes
        - Update the character position of the last complete terminal
        - Update the set of candidate terminals
        """
        new_seq = self.sequence_history + sequence
        self.interactive_parser.lexer_thread.state.text = new_seq
        try:
            self.interactive_parser.exhaust_lexer()
        except UnexpectedCharacters as e:
            self.last_terminal_pos = e.pos_in_stream
        else:
            self.last_terminal_pos = len(new_seq)

        self._update_candidate_terminals()

        if not self.valid_next_terminals:
            raise ValueError(f"Invalid continuation for `{self.sequence_history}` `{sequence}`")

    def _update_sequence(self, full_sequence: str):
        """Set the complete sequences value in the lexer and base"""
        assert self.full_sequence.startswith(self.sequence_history)
        self.interactive_parser.lexer_thread.state.text = full_sequence
        self.sequence_history = full_sequence

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
            if term == "$END":
                continue
            if self.partial_seq_validator[term](self.terminal_partial_seq + new_seq):
                return True
        return False


    def get_valid_next_tokens(self, token_trie):
        valid_node_stack = []
        for term in self.valid_next_terminals:
            import pdb;pdb.set_trace()



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

    unicode_chars = [chr(i) for i in range(256)]

    import time
    start = time.time()
    for char in complex_json_file:
        parser.step_seq(char)
        for ch in unicode_chars:
            parser.is_valid_next_seq(ch)

    print("took", time.time() - start, "seconds to process", len(complex_json_file), "characters")


def main():
    # Usage
    ebnf_grammar = """
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

    parser = InteractivePredictiveLALRParser(ebnf_grammar, 'value')
    test_valid_next_tokens(parser)


if __name__ == "__main__":
    main()

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
