# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant inherits TurboQuant's forward/do_kv_cache_update orchestration.

UltraQuant only overrides the store/decode/prefill seams, so a TurboQuant-side
change to a seam signature, to an inherited helper, or to the ``layer._tq_*``
attributes ``forward`` reads will break UltraQuant at serve time without
touching any UltraQuant kernel test. These checks pin that contract.
"""

import ast
import inspect
import textwrap

from vllm.v1.attention.backends import ultraquant_attn
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
from vllm.v1.attention.backends.ultraquant_attn import UltraQuantAttentionImpl

_SEAMS = ("_ensure_on_device", "_store_kv", "_decode_attention", "_prefill_attention")


def _method_ast(cls: type, name: str) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name))))


def _tq_attrs_read(cls: type, name: str) -> set[str]:
    """``_tq_*`` attribute names read (not assigned) in ``cls.name``."""
    tree = _method_ast(cls, name)
    assigned = {
        t.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute)
    }
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_tq_")
    }
    return read - assigned


def _tq_attrs_written(cls: type, name: str) -> set[str]:
    tree = _method_ast(cls, name)
    return {
        t.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and t.attr.startswith("_tq_")
    }


def test_ultraquant_matches_turboquant_seam_contract():
    # 1. Each overridden seam accepts every parameter the base declares, so any
    #    call the inherited orchestration makes stays valid for UltraQuant.
    for name in _SEAMS:
        base = list(
            inspect.signature(getattr(TurboQuantAttentionImpl, name)).parameters
        )
        override = list(
            inspect.signature(getattr(UltraQuantAttentionImpl, name)).parameters
        )
        assert override[: len(base)] == base, (
            f"UltraQuantAttentionImpl.{name} no longer accepts the arguments "
            f"TurboQuantAttentionImpl passes: base={base} override={override}"
        )

    # 2. Keyword arguments UltraQuant passes to *inherited* TurboQuant helpers
    #    must exist on those helpers (e.g. _flash_attn_varlen).
    own = vars(UltraQuantAttentionImpl)
    module = ast.parse(inspect.getsource(ultraquant_attn))
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "self"
        ):
            continue
        target = getattr(TurboQuantAttentionImpl, fn.attr, None)
        if target is None or fn.attr in own or not callable(target):
            continue
        params = inspect.signature(target).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            assert kw.arg in params, (
                f"UltraQuant calls inherited {fn.attr}(..., {kw.arg}=...) but "
                f"TurboQuantAttentionImpl.{fn.attr} accepts {list(params)}"
            )

    # 3. Every layer._tq_* attribute the inherited forward reads is set by
    #    UltraQuant's _ensure_on_device.
    required = _tq_attrs_read(TurboQuantAttentionImpl, "forward")
    provided = _tq_attrs_written(UltraQuantAttentionImpl, "_ensure_on_device")
    assert required, "expected inherited forward to read layer._tq_* attributes"
    assert required <= provided, (
        f"UltraQuant._ensure_on_device does not set {sorted(required - provided)}, "
        f"which the inherited TurboQuant forward reads"
    )
