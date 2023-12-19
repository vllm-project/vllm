import collections
from parsimonious.grammar import Grammar

import parsimonious

from lark import Lark, Transformer, v_args
from parsley import makeGrammar


class TokenIndex:
    def __init__(self, tokenizer):
        # map id -> token str including whitespace
        self.norm_vocab = {}
        for token_id in tokenizer.vocab.values():
            # TODO: look into difference between tokens, e.g. 28705, 35 are both " 9"
            # assert norm_token not in self.norm_vocab,
            norm_token = tokenizer.decode([tokenizer.bos_token_id, token_id])[len(tokenizer.bos_token):]
            self.norm_vocab[norm_token] = token_id

        # get index allowing efficient retrieval of valid tokens given a sequence
        # given tokens ["art", "artist", "argument", "alice"]
        # map "a" -> ["ar", "al"]
        # map "ar" -> ["art", "artist"]
        # map "art" -> [None, "artist"]  (None indicates match)
        self.char_map = collections.defaultdict(set)
        for word in self.norm_vocab:
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                if i < len(word):
                    self.char_map[prefix].add(word[i])
                else:
                    # Add None for complete matches
                    self.char_map[prefix].add(None)


    def get_valid_next_charset(self, seq, legal_chars):
        results = set(self.char_map[seq]) & legal_chars
        return results

    def is_token(self, tok):
        return tok in self.norm_vocab


class EpsilonNFA:
    """
    Traverses a Character-Based Epsilon-NFA.

    Used by find valid next character sequences.

    self.nfa (dict): A dictionary representing the NFA. It includes:
        - 'states' (list): A list of states (UUIDsn) inn the NFA.
        - 'initial_state' (UUID or any hashable ID): The initial state of the NFA.
        - 'final_states' (list): A list of final or accepting states (UUIDs).
        - 'alphabets' (list): The set of input symbols (characters).
        - 'transition_function' (dict): A dictionary representing the state
          transitions. Each key is a state (UUID), and its value is another
          dictionary mapping input symbols to lists of next states (UUIDs).

    self.nfa should never be mutated.
    """
    def __init__(self, nfa):
        self.nfa = nfa

        # Set of states you may be in
        self.current_states = set([self.nfa["initial_state"]])
        self.current_str = ""

        self.legal_chars = set([char for char in self.nfa["alphabets"] if char != "$"])

        self._resolved_epsilon_cache = {}

    def step_seq(self, seq):
        for char in seq:
            self.step(char)

    def step(self, char):
        """
        Updates the canonical state
        """
        next_states = self.next_char_state_map()[char]
        if not next_states:
            raise ValueError(f"Illegal transition from '{self.current_str}', no next state for '{char}'")
        self.current_states = next_states
        self.current_str += char

    def simulate_step(self, chars, state_set=None):
        """
        Return map of chars and their resulting next state-set given a current state-set and new chars
        """
        if state_set is None:
            state_set = self.current_states
        state_map = self.next_char_state_map(state_set)
        return {tok: state_map[tok] for tok in chars if tok in state_map}

    def next_char_state_map(self, current_states=None):
        """
        Creates a mapping of possible next chars to a set of valid states for each char
        """
        if current_states is None:
            current_states = self.current_states

        char_to_states = collections.defaultdict(set)

        if bool(current_states & set(self.nfa["final_states"])):
            char_to_states[None] = None

        for state in self._resolve_epsilon_closure(current_states):
            for char, next_states in self.nfa["transition_function"][state].items():
                if next_states and char != "$":
                    char_to_states[char].update(next_states)

        return char_to_states

    def _resolve_epsilon_closure(self, states):
        closure = set()
        for state in states:
            if state in self._resolved_epsilon_cache:
                new_closures = self._resolved_epsilon_cache[state]
            else:
                new_closures = self._get_epsilon_closure(state)
                self._resolved_epsilon_cache[state] = new_closures
            closure.update(self._get_epsilon_closure(state))
        return closure

    def _get_epsilon_closure(self, state, visited=None):
        if visited is None:
            visited = set()

        stack = [state]
        while stack:
            current_state = stack.pop()
            if current_state not in visited:
                visited.add(current_state)
                stack.extend(self.nfa["transition_function"][current_state].get('$', []))

        return visited


class TokenConstraintLogitProcessor:
    def __init__(self, tokenizer, nfa):
        self.tokenizer = tokenizer
        self.token_index = TokenIndex(tokenizer)
        self.nfa = nfa
        self.prev_token_ids = []
        self.prev_text = ""

    def __call__(self, token_ids, logits):

        # ensure integrity
        assert token_ids[:len(self.prev_token_ids)] == self.prev_token_ids
        self.prev_token_ids = token_ids

        # get new text and step NFA forward
        text = tokenizer.decode(token_ids)
        new_text = text[len(self.prev_text):]
        self.prev_text = text
        self.nfa.step_seq(new_text)

        # get valid new token ids
        valid_tokens = set(self.get_allowable_next_token_set())
        valid_token_ids = [
            self.tokenizer.eos_token_id if t is None else self.token_index.norm_vocab[t]
            for t in valid_tokens
        ]

        if not valid_token_ids:
            raise ValueError("Found no valid tokens, this should never occur.")

        logits = [
            logit_val if tok_id in valid_token_ids else -float("inf")
            for tok_id, logit_val in zip(sorted(self.tokenizer.vocab.values()), logits)
        ]
        return logits


    def get_allowable_next_token_set(self, current_text="", nfa_next_tok_states=False):
        """
        Get set of valid tokens.
        1) Ask NFA for legal first char
        3) While legal TokenIndex hasn't been exhausted
          A) Ask TokenIndex for legal Nth char set
          B) Ask NFA for
        """
        if nfa_next_tok_states is None:
            return [None]
        if nfa_next_tok_states == False:
            nfa_next_tok_states = self.nfa.next_char_state_map()

        legal_tokens = []

        if None in nfa_next_tok_states:
            legal_tokens.append(None)
            del nfa_next_tok_states[None]

        for char, next_states in nfa_next_tok_states.items():
            # all current sequences are legal per nfa, find legal next token with token index
            new_seq = current_text + char
            tokidx_next_chars = self.token_index.get_valid_next_charset(
                new_seq,
                self.nfa.legal_chars
            )

            if self.token_index.is_token(new_seq):
                legal_tokens.append(new_seq)

            # given legal next chars in token index, get the subset allowed by NFA and recurse
            legal_tokens += self.get_allowable_next_token_set(
                new_seq,
                self.nfa.simulate_step(tokidx_next_chars, next_states)
            )

        return legal_tokens


class LarkPDA:
    """
    Traverses a Lark Parser's PushDown Automata character by character
    """
    def __init__(self, lark_grammar: str, start: str = "value"):
        self.parser = Lark(
            lark_grammar,
            start=start,
            lexer="basic"
        )

    def step_seq(self, seq):
        for char in seq:
            self.step(char)

    def step(self, char):
        """
        Updates the canonical state
        """
        next_states = self.next_char_state_map()[char]
        if not next_states:
            raise ValueError(f"Illegal transition from '{self.current_str}', no next state for '{char}'")
        self.current_states = next_states
        self.current_str += char

    def simulate_step(self, chars, state_set=None):
        """
        Return map of chars and their resulting next state-set given a current state-set and new chars
        """
        if state_set is None:
            state_set = self.current_states
        state_map = self.next_char_state_map(state_set)
        return {tok: state_map[tok] for tok in chars if tok in state_map}

    def next_char_state_map(self, current_states=None):
        """
        Creates a mapping of possible next chars to a set of valid states for each char
        """
        if current_states is None:
            current_states = self.current_states

        char_to_states = collections.defaultdict(set)

        if bool(current_states & set(self.nfa["final_states"])):
            char_to_states[None] = None

        for state in self._resolve_epsilon_closure(current_states):
            for char, next_states in self.nfa["transition_function"][state].items():
                if next_states and char != "$":
                    char_to_states[char].update(next_states)

        return char_to_states


def lark_to_pushdown_automata_spec(lark_grammar: str, start: str = "value") -> dict:
    parser = Lark(lark_grammar, start=start, lexer="basic")
    import pdb;pdb.set_trace()


def handle_regex(pattern):
    """
    Handle a regex pattern and convert it into a state representation for PDA.

    Args:
    pattern (str): The pattern of the regular expression.

    Returns:
    str: A unique representation of the regex for the PDA.
    """
    # Convert the regex pattern into a suitable representation for the PDA.
    # This is a placeholder for the actual conversion logic.
    return f"REGEX({pattern})"


from parsimonious.nodes import NodeVisitor
from parsimonious.expressions import Literal, Sequence, Regex

def convert_grammar_to_pda(grammar):
    pda = {
        'states': set(),
        'initial_state': 'q0',
        'final_states': {'q_accept'},
        'stack_alphabet': set(),
        'transition_function': collections.defaultdict(lambda: collections.defaultdict(list))
    }

    def process_expression(state, expr, is_terminal=False):
        if isinstance(expr, Literal):
            pda['stack_alphabet'].add(expr.literal)
            next_state = 'q_accept' if is_terminal else 'q' + str(len(pda['states']) + 1)
            pda['transition_function'][state][('epsilon', expr.literal)].append((next_state, []))
            if not is_terminal:
                pda['states'].add(next_state)
        elif isinstance(expr, Sequence):
            current_state = state
            for i, member in enumerate(expr.members):
                is_last_member = i == len(expr.members) - 1
                process_expression(current_state, member, is_terminal=is_last_member)
                if not is_last_member:
                    current_state = 'q' + str(len(pda['states']))
        elif isinstance(expr, Regex):
            raise NotImplementedError("Regex handling not implemented")

    for non_terminal, production in grammar.items():
        pda['states'].add(non_terminal)
        pda['stack_alphabet'].add(non_terminal)
        process_expression(non_terminal, production)

    return dict(pda)


# Test Case
def test_convert_simple_grammar_to_pda():
    grammar = Grammar("""
    expression = term "+" term
    term       = "number"
    """)

    expected_pda = {
        'states': {'expression', 'term', 'q0', 'q_accept'},
        'initial_state': 'q0',
        'final_states': {'q_accept'},
        'stack_alphabet': {'expression', 'term', '+', 'number'},
        'transition_function': {
            'q0': {('epsilon', 'epsilon'): [('expression', ['term', '+', 'term'])]},
            'expression': {('epsilon', 'term'): [('term', [])]},
            'term': {('epsilon', 'number'): [('q_accept', [])]}
        }
    }

    actual_pda = convert_grammar_to_pda(grammar)
    import pprint
    pprint.pprint(actual_pda)
    assert actual_pda == expected_pda, "PDA configuration does not match expected output"


if __name__ == "__main__":
    import transformers
    import numpy as np

    test_convert_simple_grammar_to_pda()
    import pdb;pdb.set_trace()


    grammar = r"""
    expr        = (entry / emptyline)*
    entry       = section pair*

    section     = lpar word rpar ws
    pair        = key equal value ws?

    key         = word+
    value       = (word / quoted)+
    word        = ~r"[-\w]+"
    quoted      = ~'"[^\"]+"'
    equal       = ws? "=" ws?
    lpar        = "["
    rpar        = "]"
    ws          = ~"\s*"
    emptyline   = ws+
    """

    pda_config = convert_grammar_to_pda(Grammar(grammar))
    import pdb;pdb.set_trace()

    try_parsley()

    json_grammar = r"""
        ?start: value

        ?value: object
              | array
              | string
              | SIGNED_NUMBER      -> number
              | "true"             -> true
              | "false"            -> false
              | "null"             -> null

        array  : "[" [value ("," value)*] "]"
        object : "{" [pair ("," pair)*] "}"
        pair   : string ":" value

        string : ESCAPED_STRING

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS

        %ignore WS
    """
    #lark_to_pushdown_automata_spec(json_grammar)

    tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    sample_from_logits = lambda lgts: np.random.choice(len(lgts), p=np.exp(lgts)/np.sum(np.exp(lgts)))

    for i in range(4):

        logit_processor = TokenConstraintLogitProcessor(
            tokenizer=tokenizer,
            nfa=EpsilonNFA(nfa=regex_to_nfa.regex_to_nfa(
                r"(large )?(language )((models )+(inference engines ))(are )((useful)+((very )*complex))."
            )),
        )

        token_ids = []
        while True:
            logits = logit_processor(
                token_ids=token_ids,
                logits=np.random.uniform(-10, 10, len(tokenizer.vocab))
            )
            new_token_id = sample_from_logits(logits)
            token_ids.append(new_token_id)
            if new_token_id == tokenizer.eos_token_id:
                break
        print(f"run #{i}")
        print("\ttokenid", token_ids)
        print("\ttokens:", [tokenizer.decode(tok_id, ) for tok_id in token_ids])
        print("\tresult:", tokenizer.decode(token_ids, skip_special_tokens=False))
