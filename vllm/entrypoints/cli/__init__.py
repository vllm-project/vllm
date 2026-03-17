# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CLI package for vLLM.

Keep this module intentionally light so importing `vllm.entrypoints.cli.main`
does not eagerly pull in benchmark modules before argument parsing.
"""

__all__: list[str] = []
