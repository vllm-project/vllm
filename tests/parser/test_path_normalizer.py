# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.path_normalizer import (
    normalize_pathlike_text,
    normalize_tool_arguments_json,
)


def test_normalize_full_path_spacing():
    assert (
        normalize_pathlike_text(r"scripts/monkey_character. gd")
        == r"scripts/monkey_character.gd"
    )


def test_normalize_duplicate_dot_before_extension():
    assert normalize_pathlike_text("forest_platform..py") == "forest_platform.py"


def test_normalize_shell_command_with_path():
    text = (
        'Get-Content "scripts/monkey_character. gd" | '
        'Select-String -Pattern "position. x"'
    )
    assert (
        normalize_pathlike_text(text)
        == 'Get-Content "scripts/monkey_character.gd" | '
        'Select-String -Pattern "position. x"'
    )


def test_does_not_rewrite_plain_prose():
    assert normalize_pathlike_text("note. and") == "note. and"


def test_normalize_json_arguments():
    arguments = '{"path":"scripts/monkey_character. gd","other":"note. and"}'
    assert (
        normalize_tool_arguments_json(arguments)
        == '{"path": "scripts/monkey_character.gd", "other": "note. and"}'
    )
