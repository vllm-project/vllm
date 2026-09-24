# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for k8s-friendly (non-TTY) tqdm behavior."""

import io

from vllm.utils.tqdm_utils import (
    _NON_TTY_BAR_FORMAT,
    _NON_TTY_MININTERVAL,
    vllm_tqdm,
)


def test_non_tty_emits_newline_terminated_lines():
    out = io.StringIO()
    assert not out.isatty()

    with vllm_tqdm(
        range(10), file=out, miniters=1, mininterval=0, desc="warmup"
    ) as pbar:
        for _ in pbar:
            pass

    # tqdm's status printer prepends '\r' to every refresh; strip it, then
    # require each refresh to be one newline-terminated self-contained line.
    lines = [line.lstrip("\r") for line in out.getvalue().split("\n")]
    lines = [line for line in lines if line]
    assert lines, "expected at least one refresh line on non-TTY output"
    for line in lines:
        assert "warmup" in line
        assert "%" in line


def test_non_tty_respects_mininterval_default():
    out = io.StringIO()
    with vllm_tqdm(range(10), file=out) as pbar:
        assert pbar.bar_format == _NON_TTY_BAR_FORMAT
        assert pbar.mininterval == _NON_TTY_MININTERVAL


def test_non_tty_drops_dynamic_ncols():
    out = io.StringIO()
    with vllm_tqdm(range(2), file=out, dynamic_ncols=True) as pbar:
        assert pbar.dynamic_ncols is False


def test_explicit_bar_format_is_not_overridden():
    out = io.StringIO()
    custom = "{desc}\n"
    with vllm_tqdm(range(2), file=out, bar_format=custom) as pbar:
        assert pbar.bar_format == custom
