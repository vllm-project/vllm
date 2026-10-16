# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel code AMD specialized decodes share, one subpackage a kernel flavor.

Nothing here is imported by the specialized decode contract itself, which stays
free of vendor runtimes so that it loads on any platform.
"""
