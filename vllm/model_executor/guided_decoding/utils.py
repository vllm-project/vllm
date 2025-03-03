# SPDX-License-Identifier: Apache-2.0

import re


def has_xgrammar_unsupported_json_features(schema: dict) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for pattern restrictions
        if "pattern" in obj:
            return True

        # Check for enum restrictions
        if "enum" in obj:
            return True

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and any(
                key in obj for key in [
                    "minimum", "maximum", "exclusiveMinimum",
                    "exclusiveMaximum", "multipleOf"
                ]):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(key in obj for key in [
                "uniqueItems", "contains", "minContains", "maxContains",
                "minItems", "maxItems"
        ]):
            return True

        # Unsupported keywords for strings
        if obj.get("type") == "string" and any(
                key in obj for key in ["minLength", "maxLength", "format"]):
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(key in obj for key in [
                "minProperties", "maxProperties", "propertyNames",
                "patternProperties"
        ]):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def has_lmf_unsupported_json_features(schema: dict) -> bool:
    """
    Check if JSON schema contains features unsupported 
    by lm_format_enforcer.

    Known issues:
    - Regex patterns:
        "grade": {
            "type": "string",
            "pattern": "^[A-D]$"  # Regex pattern
        },
    """

    def check_object(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for pattern restrictions
        if "pattern" in obj:
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


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

        # Look for GBNF rule definition
        if '::=' in line:
            return False

    return True


def convert_lark_to_gbnf(grammar_str: str) -> str:
    """
    Convert a Lark grammar string to GBNF format.

    GBNF reference:
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    Lark grammar reference:
    https://lark-parser.readthedocs.io/en/latest/grammar.html
    
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

    defined_rules = set()
    referenced_rules = set()
    output_lines = []

    def clean_line(line: str) -> str:
        """Remove comments and whitespace from line."""
        return re.sub(r'(#|//).*$', '', line).strip()

    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        """Validate quote matching in text."""
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(
                f"Mismatched quotes in {rule_name} on line {line_num}")

    def extract_references(text: str) -> set:
        """Extract rule references from text."""
        # Remove quoted strings and special characters
        text = re.sub(r'"[^"]*"', '', text)
        text = re.sub(r'[+*?()|\[\]{}]', ' ', text)
        return set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text))

    # First pass: Find root rule and validate rule definitions
    lines = [clean_line(line) for line in grammar_str.split('\n')]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith('|'):
            continue

        if ':' in line:
            try:
                name = line.split(':', 1)[0].strip().strip('?')
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                if name == 'start':
                    first_rule = 'start'
            except IndexError as e:
                raise ValueError(f"Invalid rule format on line {line_num}. "
                                 "Expected 'rule_name: definition'") from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    # Add root rule
    output_lines.append(f"root ::= {first_rule}")

    # Second pass: Process rule definitions and alternatives
    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue

        try:
            if ':' in line and not line.startswith('|'):
                # Save previous rule if exists
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}")

                # Process new rule
                name, definition = line.split(':', 1)
                current_rule = name.strip().strip('?')

                check_quotes(definition, f"rule '{current_rule}'", line_num)
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]

            elif line.startswith('|'):
                if not current_rule:
                    raise ValueError(f"Alternative '|' on line {line_num} "
                                     "without a preceding rule definition")

                alt_def = line[1:].strip()
                check_quotes(alt_def, f"alternative for rule '{current_rule}'",
                             line_num)
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)

        except ValueError as e:
            raise ValueError(f"Error on line {line_num}: {str(e)}") from e

    # Add final rule if exists
    if current_rule:
        output_lines.append(
            f"{current_rule} ::= {' | '.join(current_definition)}")

    # Validate all rules are defined
    undefined_rules = referenced_rules - defined_rules - {'root'}
    if undefined_rules:
        raise ValueError("Referenced rules are not defined: "
                         f"{', '.join(sorted(undefined_rules))}")

    return '\n'.join(output_lines)
