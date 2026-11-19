# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's extra levers (c1 / c4 latency): each an env flag,
on by default, that takes effect only once ``decode_backend=latency_specialized``
activates the gate (``SpecializedDecodeMixin.init_specialized_decode``, in every
worker before the weights load). Without it every lever is off and nothing
below is installed, so the stock decode runs exactly as before; a later stock
model in the same process turns the gate back off (installed patches then run
stock).

The levers that change vLLM's v1 engine code run as patches installed here on
activation (``install``): ``dspark`` (Markov sampling, the draft-only attention
groups), ``mla`` (the bs1 MLA planner, the sparse MLA indptr), ``runner``
(the logits rows slice, the target step's mark for the draft-only groups),
``tpgather`` (the draft's TP-split fc, the logits all-gather, the Markov
launches' TP split) and ``ctxkv`` (the draft's context K/V store). The
Kimi-K3 model state (``model_state``) and the model's own paths read
``lever_on`` directly."""

import vllm.envs as envs

# DSpark's sequential Markov draft steps as one launch each (ops/markov_sample),
# for the preview's MLA draft and the release's GQA sliding-window draft.
MARKOV = "VLLM_K31_BEAT_MARKOV"
# The target step skips draft-only attention groups; the bs1 MLA planner (AITER
# MLA groups only: the preview's MLA draft; a GQA draft plans no schedule).
DRAFTMETA = "VLLM_K31_BEAT_DRAFTMETA"
# The mono kernels' cheap spin waits and bounded router ranking.
K2SYNC = "VLLM_K31_BEAT_K2SYNC"
# One launch each for the eager per-step prep; the logits rows slice.
EAGERPREP = "VLLM_K31_BEAT_EAGERPREP"
# K1's non-temporal per-token state snapshots.
KDAK1 = "VLLM_K31_BEAT_KDAK1"
# AITER's sampled top-k for the indexer's decode rows.
SAMPLED_TOPK = "VLLM_K3_INDEXER_SAMPLED_TOPK"
# D1a also runs the MXFP4 indexer's prep.
PROLOGUE_INDEXER = "VLLM_KIMI_SD_DSA_PROLOGUE_INDEXER"
# A DFlash-family draft's replicated aux-hidden fc split over TP by output
# columns, then all-gathered (``tpgather``).
DRAFTFC = "VLLM_K31_BEAT_DRAFTFC"
# The lm_head logits all-gather over the TP group's AITER IPC all-gather
# (``tpgather``).
LOGITSAG = "VLLM_K31_BEAT_LOGITSAG"
# The fused Markov launches split the vocabulary over TP (``tpgather``).
MARKOVTP = "VLLM_K31_BEAT_MARKOVTP"
# A DFlash-family draft's context K norm, RoPE and cache writes as one launch
# (``ctxkv``).
CTXKV = "VLLM_K31_BEAT_CTXKV"

LEVERS = (
    MARKOV,
    DRAFTMETA,
    K2SYNC,
    EAGERPREP,
    KDAK1,
    SAMPLED_TOPK,
    PROLOGUE_INDEXER,
    DRAFTFC,
    LOGITSAG,
    MARKOVTP,
    CTXKV,
)

_active = False
_installed = False
# Set on the functions ``wrap`` installs.
_MARK = "__sd_lever__"


def is_active() -> bool:
    return _active


def lever_on(name: str) -> bool:
    """The lever's flag, under an active gate only."""
    return _active and bool(getattr(envs, name))


def activate(install_patches: bool = True) -> None:
    """Turns the gate on for this process (the patches first: an install that
    raises leaves the gate as it was). ``install_patches``: False for tests that
    call the levers directly."""
    global _active
    if install_patches:
        install()
    _active = True


def deactivate() -> None:
    """The gate off: a stock model's build. Installed patches stay; off, they
    run stock."""
    global _active
    _active = False


def install() -> None:
    """Installs the v1 patches, once (``wrap`` skips a method already wrapped,
    so a retry after a failed install does not wrap twice)."""
    global _installed
    if _installed:
        return
    from vllm.models.kimi_k3.amd.specialized_decode.levers import (
        ctxkv,
        dspark,
        mla,
        runner,
        tpgather,
    )

    dspark.install()
    mla.install()
    runner.install()
    tpgather.install()
    ctxkv.install()
    _installed = True


def installed() -> bool:
    return _installed


def wrap(owner, name: str, make) -> None:
    """``owner.name = make(owner.name)``, unless it is already ours."""
    stock = getattr(owner, name)
    if getattr(stock, _MARK, False):
        return
    wrapped = make(stock)
    setattr(wrapped, _MARK, True)
    setattr(owner, name, wrapped)


def is_wrapped(fn) -> bool:
    return bool(getattr(fn, _MARK, False))
