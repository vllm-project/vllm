# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ gathered context (see `context`). `gather` imports `cutlass`, so
the constants live here."""

PAGE_TOKENS = 64
# Workspace block tables are padded to whole multiples of this many pages, so
# their per-request stride has one Triton specialization.
TABLE_PAGES_ALIGN = 16
