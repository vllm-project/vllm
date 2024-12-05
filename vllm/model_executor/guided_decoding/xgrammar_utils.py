import re


def grammar_is_likely_lark(grammar_str: str) -> bool:
    """
    Check if grammar appears to use Lark syntax.
    
    Args:
        grammar_str: Input grammar string
        
    Returns:
        bool: True if grammar appears to be in Lark format, False otherwise
        
    Examples:
        >>> grammar_is_likely_lark("rule: 'abc'")
        True
        >>> grammar_is_likely_lark("rule ::= 'abc'")
        False
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split('\n'):
        # Remove both comment styles
        line = re.sub(r'(#|//).*$', '', line).strip()
        if not line:
            continue

        # Look for Lark-style rule definitions
        if ':' in line and '::=' not in line:
            return True

        # Look for Lark-specific features
        if any(pattern in line for pattern in ['?start:', '|', '~']):
            return True

    return False


def convert_lark_to_gbnf(grammar_str: str) -> str:
    """
    Convert a Lark grammar string to GBNF format.

    GBNF reference:
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    Lark grammar reference:
    https://lark-parser.readthedocs.io/en/latest/grammar.html
    
    Supports:
    - Lark rule definitions to GBNF productions
    - String literals with proper escaping
    - Multi-line rules with alternatives (|)
    - Basic terminal definitions
    - Comments (both # and // style)
    
    Args:
        grammar_str: Input grammar in Lark format
        
    Returns:
        str: Converted grammar in GBNF format
        
    Examples:
        >>> print(convert_lark_to_gbnf("rule: 'hello'"))
        root ::= rule
        rule ::= "hello"
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"Grammar must be a string, got {type(grammar_str)}")

    if not grammar_str.strip():
        raise ValueError("Grammar string cannot be empty")

    # Track rules for validation while being lenient about rule names
    defined_rules = set()
    referenced_rules = set()

    # First identify what rule should be used as root
    first_rule = None
    lines = grammar_str.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Remove both comment styles
        line = re.sub(r'(#|//).*$', '', line).strip()
        if not line:
            continue

        if ':' in line and not line.startswith('|'):
            try:
                name = line.split(':', 1)[0].strip().strip('?')
                defined_rules.add(name)

                if first_rule is None:
                    first_rule = name
                if name == 'start':  # If we find 'start', use it
                    first_rule = 'start'

            except IndexError as e:
                raise ValueError(f"Invalid rule format on line {line_num}. "
                                 "Expected 'rule_name: definition'") from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    root_rule = first_rule
    output_lines = [f"root ::= {root_rule}"]

    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        line = re.sub(r'(#|//).*$', '', line).strip()
        if not line:
            continue

        if ':' in line and not line.startswith('|'):
            if current_rule:
                output_lines.append(
                    f"{current_rule} ::= {' | '.join(current_definition)}")

            try:
                name, definition = line.split(':', 1)
                current_rule = name.strip().strip('?')

                # Basic quote validation to catch obvious errors
                if definition.count("'") % 2 != 0 or definition.count(
                        '"') % 2 != 0:
                    raise ValueError("Mismatched quotes in rule "
                                     f"'{current_rule}' on line {line_num}")

                # Convert string literals from single to double quotes
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)

                # Extract referenced rules (excluding quoted strings and
                # special characters)
                # Remove quoted strings
                temp = re.sub(r'"[^"]*"', '', definition)
                # Remove special chars
                temp = re.sub(r'[+*?()|\[\]{}]', ' ', temp)
                tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', temp)
                referenced_rules.update(tokens)

                current_definition = [definition.strip()]

            except ValueError as e:
                raise ValueError("Error parsing rule definition on "
                                 f"line {line_num}: {str(e)}") from e
            except Exception as e:
                raise ValueError("Unexpected error parsing rule on "
                                 f"line {line_num}: {str(e)}") from e

        elif line.startswith('|'):
            if not current_rule:
                raise ValueError(f"Alternative '|' on line {line_num} "
                                 "without a preceding rule definition")

            try:
                # Convert string literals from single to double quotes
                line = re.sub(r"'([^']*)'", r'"\1"', line[1:].strip())

                # Basic quote validation
                if line.count("'") % 2 != 0 or line.count('"') % 2 != 0:
                    raise ValueError(
                        "Mismatched quotes in alternative for "
                        f"rule '{current_rule}' on line {line_num}")

                # Extract referenced rules (same as above)
                temp = re.sub(r'"[^"]*"', '', line)
                temp = re.sub(r'[+*?()|\[\]{}]', ' ', temp)
                tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', temp)
                referenced_rules.update(tokens)

                current_definition.append(line)

            except ValueError as e:
                raise ValueError("Error parsing alternative on line "
                                 f"{line_num}: {str(e)}") from e

    if current_rule:
        output_lines.append(
            f"{current_rule} ::= {' | '.join(current_definition)}")

    # Check for undefined rules, excluding common terminals and special cases
    undefined_rules = referenced_rules - defined_rules - {'root'}
    if undefined_rules:
        raise ValueError("Referenced rules are not defined: "
                         f"{', '.join(sorted(undefined_rules))}")

    return '\n'.join(output_lines)
