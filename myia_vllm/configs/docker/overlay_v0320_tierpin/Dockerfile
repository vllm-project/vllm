# Overlay image: stock vllm/vllm-openai:v0.32.0 + our fix for the tiering
# race "AssertionError: Chunk ... not found in cache" (cpu/manager.py:221).
#
# PROD stopgap (2026-10-24, user GO "construire une image dédiée le temps que
# ça lande dans main"): 5 engine crashes that day on the TieringOffloadingSpec
# profile (RAM primary + fs secondary). Remove this image as soon as an
# upstream release carries an equivalent fix.
#
# Bug: TieringOffloadingManager.lookup() reports a primary HIT without pinning
# it (the scheduler pins later, in update_state_after_alloc -> prepare_load).
# A later lookup in the same scheduler step that hits a secondary tier starts
# a promotion -> primary.prepare_write([key]) -> eviction protects only the
# promoted key, so it can evict the HIT already counted. prepare_load() then
# asserts. Reachable from the scheduler (partial-tail boundary walk, sliding
# window walk after a MISS reset), not only from P2P (vllm#62615 covers P2P).
# Upstream: issue #54914, PRs #60729 (diagnostic) and #62615 (P2P), all open.
#
# Fix (PATCH.diff, 2 files): the manager records primary HITs reported during
# the current step and passes them as extra_protected to the promotion's
# prepare_write(); cleared at on_schedule_end() and reset_cache(). If no idle
# chunk is left, the promotion fails -> MISS -> recompute (no crash).
#
# Tests (tests/, run in a throwaway container of this base image):
#   stock   -> test_promotion_does_not_evict_a_hit_reported_this_step raises
#              the production assert verbatim;
#   patched -> 3 regression tests + 113 upstream v0.32.0 tests
#              (tiering/test_tiering_offloading.py, cpu/test_manager.py) pass.
#
# Base pinned by digest (release tag, not subject to nightly GC).

FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
COPY vllm/v1/kv_offload/cpu/manager.py /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/manager.py
COPY vllm/v1/kv_offload/tiering/manager.py /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/manager.py
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload \
 && python3 -c "from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager as M; import inspect; assert '_step_hit_keys' in inspect.getsource(M)"
LABEL org.myia.overlay="v0320-tierpin" org.myia.base="vllm/vllm-openai:v0.32.0"
