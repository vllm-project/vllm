# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for the tiering tests."""

import threading

import pytest


@pytest.fixture
def manager_lock() -> threading.Lock:
    """Fresh lock shared by a test's TieringOffloadingManager and its tiers.

    Rule: a test that composes a TieringOffloadingManager builds its tiers
    and the manager on this lock (ParentContext.lock is the manager's lock).
    A test that exercises a tier standalone, with no manager, is free to
    give the tier its own throwaway lock (see _parent_ctx's fallback).
    """
    return threading.Lock()
