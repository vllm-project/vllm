# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Cudagraph-aware routing ops for Helion kernels.

Helion custom ops carry a per-call Python dispatch overhead (~30us) that is a
loss in **eager** execution but is captured away under a **CUDA graph** (only the
GPU kernel launches are replayed). So we want Helion *only* when the op is being
captured into a CUDA graph, and the native ``_C`` kernel otherwise.

The decision must be made at **runtime**, inside a custom op's implementation:
``torch.cuda.is_current_stream_capturing()`` is False in eager and True during
capture. A plain ``if`` at a Python call site is evaluated at torch.compile
*trace* time (always False) and baked in, so it cannot express this -- hence a
routing custom op whose impl runs at execution time.

For each native ``_C.<name>`` op with a Helion equivalent, we register
``vllm_helion::routed_<name>`` (same schema) that dispatches:
    capturing -> torch.ops.vllm_helion.<name>   (Helion, captured into the graph)
    eager     -> torch.ops._C.<name>            (native, low dispatch overhead)
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _schema_tail(op: torch._ops.OpOverload) -> str:
    """Return the ``(...) -> ...`` part of an op's schema (drop the ns::name)."""
    s = str(op._schema)
    return s[s.index("(") :]


def build_routed_op_map(
    helion_map: dict[torch._ops.OpOverload, torch._ops.OpOverload],
) -> dict[torch._ops.OpOverload, torch._ops.OpOverload]:
    """Register ``vllm_helion::routed_<name>`` ops and return native->routed map.

    ``helion_map`` maps native ``_C`` overloads to their (schema-compatible)
    Helion overloads. Idempotent: ops are only defined once per process.
    """
    from vllm.kernels.helion.register import vllm_helion_lib

    routed_map: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}
    for native_ov, helion_ov in helion_map.items():
        name = native_ov._schema.name.split("::")[-1]
        routed_name = f"routed_{name}"

        if not hasattr(torch.ops.vllm_helion, routed_name):
            # Copy the Helion op's schema (clean ``(aN!)`` alias format) so the
            # mutated-arg annotations carry over for auto_functionalized.
            vllm_helion_lib.define(routed_name + _schema_tail(helion_ov))

            def _make_impl(
                nat: torch._ops.OpOverload,
                hel: torch._ops.OpOverload,
                schema_args: list,
            ):
                # auto_functionalized omits trailing args equal to their default,
                # and the native _C op has no defaults, so we must reconstruct the
                # full positional arg list (filling defaults) before dispatching.
                names = [a.name for a in schema_args]
                defaults = {
                    a.name: a.default_value
                    for a in schema_args
                    if a.has_default_value()
                }

                def impl(*args, **kwargs):
                    vals = list(args)
                    for i in range(len(args), len(names)):
                        n = names[i]
                        vals.append(kwargs[n] if n in kwargs else defaults[n])
                    # Helion only when captured into a CUDA graph (its per-call
                    # dispatch overhead is a loss in eager); native otherwise.
                    if torch.cuda.is_current_stream_capturing():
                        return hel(*vals)
                    return nat(*vals)

                return impl

            vllm_helion_lib.impl(
                routed_name, _make_impl(native_ov, helion_ov, list(helion_ov._schema.arguments)), "CUDA"
            )
            # void-return mutating ops: fake just returns None; the mutation
            # layout is taken from the schema's (aN!) annotations.
            vllm_helion_lib._register_fake(routed_name, lambda *a, **k: None)

        routed_ov = getattr(torch.ops.vllm_helion, routed_name).default
        routed_map[native_ov] = routed_ov

    if routed_map:
        logger.info(
            "Registered cudagraph-routed Helion ops: %s",
            sorted(str(v) for v in routed_map.values()),
        )
    return routed_map
