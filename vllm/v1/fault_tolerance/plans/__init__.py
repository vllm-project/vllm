# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery plans bundled with vLLM.

Each plan is a small subclass of ``BaseRecoveryPlan`` registered via
``@register_recovery_plan(...)``. New plans are added by writing one file
in this directory; no edits to existing classes are required.

Importing this module registers the bundled plans so they can be looked up
by name from ``/fault_tolerance/apply``.
"""

# Importing the plan modules executes their module-scope ``@register_*``
# decorators, populating the registry.
from vllm.v1.fault_tolerance.plans import (  # noqa: F401
    abort_communicator,
    pause,
    retry,
    scale_down,
)
