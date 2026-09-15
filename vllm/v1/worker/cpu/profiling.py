# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase labels for the V2 model runner's host work on CPU.

The op table attributes that work to bare aten calls, which does not say which
phase of the step paid for it. Off by default so a timed run is not perturbed;
set VLLM_CPU_MRV2_PROFILE_PHASES=1 for a profile run.
"""

import functools
import os

from torch.profiler import record_function

ENABLED = os.getenv("VLLM_CPU_MRV2_PROFILE_PHASES", "0") == "1"


def label(cls, method: str, name: str) -> None:
    fn = getattr(cls, method)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)

    setattr(cls, method, wrapper)


def label_model_forward(cls, name: str) -> None:
    """Label the model's forward, which splits a step into host and forward.

    Hooks ``load_model`` because the model does not exist before it runs, and
    must be given the concrete runner class: the CPU runners override it.
    """
    fn = cls.load_model

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        inner = self.model.forward

        @functools.wraps(inner)
        def forward(*args, **kwargs):
            with record_function(name):
                return inner(*args, **kwargs)

        self.model.forward = forward
        return out

    cls.load_model = wrapper
