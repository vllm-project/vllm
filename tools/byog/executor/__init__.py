# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4: BYOG execution coordinator, platform adapters, and workload plugins.

IMPORTANT: Keep this package `__init__.py` free of submodule imports.
`tools.byog.catalog` imports `tools.byog.executor.interfaces`, while
`tools.byog.executor.coordinator` imports `tools.byog.catalog`. Re-exporting
`coordinator` here would trigger a circular import at package load time.
"""
