# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared hw-agnostic plumbing — candidate to lift to ``vllm/model_executor/hw_agnostic/``.

Files under this subtree contain no DSv4-specific math. The structure
mirrors ``vllm/model_executor/`` so a future move to a shared package is
mechanical (single ``git mv shared/ vllm/model_executor/hw_agnostic/``).
"""
