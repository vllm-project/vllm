# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM environment variables.

This module re-exports the lazy :class:`~vllm.envs_impl.Envs` singleton so
that the canonical usage pattern continues to work unchanged::

    import vllm.envs as envs

    print(envs.VLLM_PORT)
    print(envs.VLLM_HOST_IP)

All attribute access is lazy: the corresponding ``os.environ`` lookup is
performed on each access (or once if caching is enabled via
``envs.enable_envs_cache()``).

Helper utilities are importable from :mod:`vllm.envs_impl`:

    from vllm.envs_impl import env_with_choices, environment_variables
"""

import sys

from vllm.envs_impl import (  # noqa: F401 - re-exported for backward compatibility
    Envs,
    env_list_with_choices,
    env_set_with_choices,
    env_with_choices,
    environment_variables,
    envs,
)

# Replace this module in sys.modules with the Envs singleton so that
#   import vllm.envs as envs
# binds ``envs`` directly to the Envs() instance, giving IDEs full type
# information and hover docs from the class annotations.
sys.modules[__name__] = envs  # type: ignore[assignment]
