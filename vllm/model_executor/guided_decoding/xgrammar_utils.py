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
    # First identify what rule should be used as root
    first_rule = None
    for line in grammar_str.split('\n'):
        # Remove both comment styles
        line = re.sub(r'(#|//).*$', '', line).strip()
        if not line:
            continue

        if ':' in line and not line.startswith('|'):
            name = line.split(':', 1)[0].strip().strip('?')
            if first_rule is None:
                first_rule = name
            if name == 'start':  # If we find 'start', use it
                first_rule = 'start'
                break

    if first_rule is None:
        raise ValueError("No rules found in grammar")

    # Use provided root_name if specified
    root_rule = first_rule
    output_lines = [f"root ::= {root_rule}"]

    current_rule = None
    current_definition = []

    for line in grammar_str.split('\n'):
        # Remove both comment styles
        line = re.sub(r'(#|//).*$', '', line).strip()
        if not line:
            continue

        # Handle rule definition
        if ':' in line and not line.startswith('|'):
            # If we were building a rule, add it
            if current_rule:
                output_lines.append(
                    f"{current_rule} ::= {' | '.join(current_definition)}")

            # Start new rule
            name, definition = line.split(':', 1)
            current_rule = name.strip().strip('?')
            # Convert string literals from single to double quotes if needed
            definition = re.sub(r"'([^']*)'", r'"\1"', definition)
            current_definition = [definition.strip()]

        # Handle continuation with |
        elif line.startswith('|'):
            if current_rule:
                # Convert string literals in alternatives too
                line = re.sub(r"'([^']*)'", r'"\1"', line[1:].strip())
                current_definition.append(line)

    # Add the last rule if exists
    if current_rule:
        output_lines.append(
            f"{current_rule} ::= {' | '.join(current_definition)}")

    return '\n'.join(output_lines)
