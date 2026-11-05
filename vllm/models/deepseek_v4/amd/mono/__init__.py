# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Pro mono decode layer: one decoder layer of a decode step in two
persistent FlyDSL launches, its TP all-reduces in-kernel (README.md). The host
side is ``runner``; vLLM docks it in ``..mono_decode``."""
