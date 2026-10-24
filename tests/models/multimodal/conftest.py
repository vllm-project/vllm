# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM multimodal tests."""

import torch.nn.functional as F

# openbmb/MiniCPM-Llama3-V-2_5's remote ``resampler.py`` annotates with ``List``
# without importing it, relying on the alias leaking from
# ``from torch.nn.functional import *``. Recent torch uses builtin generics and no
# longer re-exports the typing aliases, so executing that module raises NameError.
# TODO: Remove once the checkpoint fixes its imports.
if not hasattr(F, "List"):
    F.List = list
