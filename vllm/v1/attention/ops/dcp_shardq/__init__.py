# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ gathered context.

``gather.py`` imports ``cutlass`` at module level. ``PAGE_TOKENS`` lives here
so an importer that needs only the page size does not import ``gather``.
"""

PAGE_TOKENS = 64
