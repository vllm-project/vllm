# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 77 — adaptive ngram K controller.

Activates `vllm/_genesis/kernels/adaptive_ngram_controller.py`
(EMA + hysteresis + auto-disable).

================================================================
WHAT THIS PATCH DOES
================================================================

Wraps `NgramProposer.propose()`:

1. Before calling `batch_propose`:
   - Query `AdaptiveNgramController.decide_K()` for the current optimal K
   - If K == 0: skip ngram entirely for this batch, return empty drafts
     (delivers ~no-spec baseline TPS — fastest path on low-acceptance workloads)
   - If K > 0: temporarily override `self.k = K` for this propose() call

2. After `batch_propose`:
   - Compute approx accepted_lens by diffing num_tokens_no_spec vs previous
     batch (this is a coarse signal — exact acceptance lives in sampler;
     approx is good enough for the 0/1/3/5 step controller)
   - Call `controller.update(accepted_lens, drafted_lens)` to feed the EMA

3. Logs at debug level (or warning when K transitions occur — see controller)

================================================================
ENV
================================================================

Master switch:
  GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1

Tunables (forwarded to controller):
  GENESIS_P77_STEPS="0,1,3,5"
  GENESIS_P77_EMA_ALPHA=0.2
  GENESIS_P77_WARMUP_BATCHES=10
  GENESIS_P77_UPDATE_INTERVAL=5
  GENESIS_P77_HYSTERESIS_DOWN=0.25
  GENESIS_P77_DISABLE_THRESHOLD=0.30
  GENESIS_P77_PROBE_INTERVAL=100

================================================================
INTERACTIONS
================================================================

- P75 (suffix decoding): if active, NgramProposer is replaced with
  SuffixDecodingProposer → P77 wiring's text-patch on `ngram_proposer.py`
  is harmless (file still loaded but never instantiated). No conflict.
- P70 (auto-strict-ngram min>=8): orthogonal — P70 sets
  `prompt_lookup_min`, P77 controls `K = num_speculative_tokens`. Both
  active simultaneously is fine and complementary.
- MTP method: P77 does nothing (only NgramProposer is patched).

================================================================
LIMITATIONS
================================================================

- The accepted_lens we compute is APPROXIMATE (diff of
  `num_tokens_no_spec` between consecutive calls). Exact per-request
  acceptance lives downstream in the rejection sampler and is not
  trivially accessible from the proposer side without a callback hook
  in gpu_model_runner.py. The approximation is sufficient for K
  selection at granularity [0, 1, 3, 5].
- K=0 path: NgramProposer returns empty drafts; vLLM scheduler still
  runs the K+1 cudagraph capture path BUT receives no draft tokens, so
  acceptance is N/A and the verify forward pass is short-circuited.
  Net: ~no-spec TPS, ~3-4× faster than wasted-K=3 path on free-form.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Algorithm: SGLang Apache-2.0 + Nightjar arXiv 2512.22420.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p77_adaptive_ngram_k")

GENESIS_P77_MARKER = "Genesis P77 adaptive ngram K controller v7.43"


# ─── Sub-patch: wrap NgramProposer.propose with adaptive-K logic ────────────
# Anchor on the FULL propose() function body. Replace with adaptive-aware
# version that: gates on env, queries controller, optionally short-circuits,
# and feeds back accepted_lens approximation.

P77_OLD = (
    "    def propose(\n"
    "        self,\n"
    "        sampled_token_ids: list[list[int]],\n"
    "        num_tokens_no_spec: np.ndarray,\n"
    "        token_ids_cpu: np.ndarray,\n"
    "        slot_mappings: dict[str, torch.Tensor]\n"
    "        | list[dict[str, torch.Tensor]]\n"
    "        | None = None,  # unused\n"
    "    ) -> list[list[int]]:\n"
    "        # find which requests need ngram proposals\n"
    "        valid_ngram_requests = []\n"
    "        for i, sampled_ids in enumerate(sampled_token_ids):\n"
    "            num_sampled_ids = len(sampled_ids)\n"
    "            if not num_sampled_ids:\n"
    "                # Skip speculative decoding.\n"
    "                continue\n"
    "\n"
    "            num_tokens = num_tokens_no_spec[i]\n"
    "            if num_tokens >= self.max_model_len:\n"
    "                # Skip requests that have already reached the max model length.\n"
    "                continue\n"
    "\n"
    "            valid_ngram_requests.append(i)\n"
    "\n"
    "        draft_token_ids = self.batch_propose(\n"
    "            len(sampled_token_ids),\n"
    "            valid_ngram_requests,\n"
    "            num_tokens_no_spec,\n"
    "            token_ids_cpu,\n"
    "        )\n"
    "\n"
    "        return draft_token_ids\n"
)

P77_NEW = (
    "    def propose(\n"
    "        self,\n"
    "        sampled_token_ids: list[list[int]],\n"
    "        num_tokens_no_spec: np.ndarray,\n"
    "        token_ids_cpu: np.ndarray,\n"
    "        slot_mappings: dict[str, torch.Tensor]\n"
    "        | list[dict[str, torch.Tensor]]\n"
    "        | None = None,  # unused\n"
    "    ) -> list[list[int]]:\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P77 v7.43] Adaptive K controller. EMA + hysteresis +\n"
    "        # auto-disable on low acceptance. Fast-path short-circuit when K=0.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        _genesis_p77_active = False\n"
    "        _genesis_p77_K = self.k\n"
    "        _genesis_p77_orig_k = self.k\n"
    "        _genesis_p77_controller = None\n"
    "        try:\n"
    "            from vllm._genesis.kernels.adaptive_ngram_controller import (\n"
    "                is_active as _genesis_p77_is_active,\n"
    "                get_controller as _genesis_p77_get_controller,\n"
    "            )\n"
    "            if _genesis_p77_is_active():\n"
    "                _genesis_p77_active = True\n"
    "                _genesis_p77_controller = _genesis_p77_get_controller()\n"
    "                _genesis_p77_K = _genesis_p77_controller.decide_K()\n"
    "                # Track last num_tokens for approximate acceptance signal.\n"
    "                if not hasattr(self, '_genesis_p77_last_num_tokens'):\n"
    "                    self._genesis_p77_last_num_tokens = None\n"
    "        except Exception as _genesis_p77_init_err:\n"
    "            import logging as _genesis_p77_logmod\n"
    "            _genesis_p77_logmod.getLogger('genesis.kernels.p77').warning(\n"
    "                '[Genesis P77] init failed (%s); using upstream K=%d',\n"
    "                _genesis_p77_init_err, self.k,\n"
    "            )\n"
    "\n"
    "        # K=0: short-circuit. Return empty drafts so verify pass becomes\n"
    "        # essentially no-spec (1 forward pass per token).\n"
    "        if _genesis_p77_active and _genesis_p77_K == 0:\n"
    "            try:\n"
    "                # Feed approximate accept signal even when K=0 so probe\n"
    "                # interval can detect workload change.\n"
    "                if (\n"
    "                    self._genesis_p77_last_num_tokens is not None\n"
    "                    and len(num_tokens_no_spec) == len(self._genesis_p77_last_num_tokens)\n"
    "                ):\n"
    "                    _genesis_p77_accept = [\n"
    "                        max(0, int(num_tokens_no_spec[_i] - self._genesis_p77_last_num_tokens[_i]) - 1)\n"
    "                        for _i in range(len(num_tokens_no_spec))\n"
    "                        if len(sampled_token_ids[_i]) > 0\n"
    "                    ]\n"
    "                    _genesis_p77_drafted = [0] * len(_genesis_p77_accept)\n"
    "                    _genesis_p77_controller.update(_genesis_p77_accept, _genesis_p77_drafted)\n"
    "                self._genesis_p77_last_num_tokens = num_tokens_no_spec.copy()\n"
    "            except Exception:\n"
    "                pass\n"
    "            return [[] for _ in range(len(sampled_token_ids))]\n"
    "\n"
    "        # K > 0: temporarily override self.k for this propose call.\n"
    "        if _genesis_p77_active and _genesis_p77_K != self.k:\n"
    "            self.k = _genesis_p77_K\n"
    "\n"
    "        # find which requests need ngram proposals\n"
    "        valid_ngram_requests = []\n"
    "        for i, sampled_ids in enumerate(sampled_token_ids):\n"
    "            num_sampled_ids = len(sampled_ids)\n"
    "            if not num_sampled_ids:\n"
    "                # Skip speculative decoding.\n"
    "                continue\n"
    "\n"
    "            num_tokens = num_tokens_no_spec[i]\n"
    "            if num_tokens >= self.max_model_len:\n"
    "                # Skip requests that have already reached the max model length.\n"
    "                continue\n"
    "\n"
    "            valid_ngram_requests.append(i)\n"
    "\n"
    "        draft_token_ids = self.batch_propose(\n"
    "            len(sampled_token_ids),\n"
    "            valid_ngram_requests,\n"
    "            num_tokens_no_spec,\n"
    "            token_ids_cpu,\n"
    "        )\n"
    "\n"
    "        # Restore self.k AFTER batch_propose (so static config doesn't drift).\n"
    "        if _genesis_p77_active and self.k != _genesis_p77_orig_k:\n"
    "            self.k = _genesis_p77_orig_k\n"
    "\n"
    "        # Update controller with approximate acceptance signal.\n"
    "        if _genesis_p77_active and _genesis_p77_controller is not None:\n"
    "            try:\n"
    "                if (\n"
    "                    self._genesis_p77_last_num_tokens is not None\n"
    "                    and len(num_tokens_no_spec) == len(self._genesis_p77_last_num_tokens)\n"
    "                ):\n"
    "                    _genesis_p77_accept = [\n"
    "                        max(0, int(num_tokens_no_spec[_i] - self._genesis_p77_last_num_tokens[_i]) - 1)\n"
    "                        for _i in range(len(num_tokens_no_spec))\n"
    "                        if len(sampled_token_ids[_i]) > 0\n"
    "                    ]\n"
    "                    _genesis_p77_drafted = [_genesis_p77_K] * len(_genesis_p77_accept)\n"
    "                    _genesis_p77_controller.update(_genesis_p77_accept, _genesis_p77_drafted)\n"
    "                self._genesis_p77_last_num_tokens = num_tokens_no_spec.copy()\n"
    "            except Exception as _genesis_p77_upd_err:\n"
    "                import logging as _genesis_p77_logmod2\n"
    "                _genesis_p77_logmod2.getLogger('genesis.kernels.p77').debug(\n"
    "                    '[Genesis P77] update failed (%s); controller will retry next batch',\n"
    "                    _genesis_p77_upd_err,\n"
    "                )\n"
    "\n"
    "        return draft_token_ids\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/ngram_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P77 v1/spec_decode/ngram_proposer.py — adaptive K wrapper",
        target_file=str(target),
        marker=GENESIS_P77_MARKER,
        sub_patches=[
            TextPatch(
                name="p77_adaptive_propose",
                anchor=P77_OLD,
                replacement=P77_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P77",
            "_genesis_p77_controller",
            "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P77 — adaptive ngram K wiring."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P77")
    log_decision("P77", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/spec_decode/ngram_proposer.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P77] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P77" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed adaptive K",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "P77 applied: NgramProposer.propose() wrapped with adaptive K controller. "
        "Activates when GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1 + ngram method active. "
        "K dynamically chosen from {0,1,3,5} based on EMA of acceptance, with "
        "auto-disable to K=0 (no-spec mode) on accept_rate < 30%. Probe every 100 "
        "batches re-tests in case workload shifted."
    )
