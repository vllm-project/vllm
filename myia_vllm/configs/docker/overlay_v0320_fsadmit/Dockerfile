# Overlay image: vllm-openai-v0320-tierpin:v1 + our fs-tier write-admission
# filter (jsboige/vllm#198, Phase 1a).
#
# NOT deployed (built only for testing until the user GO). Stacks ON TOP of
# the production tierpin overlay: the base image already carries stock
# v0.32.0 + the tiering-race fix ("Chunk ... not found in cache",
# cpu/manager.py extra_protected + manager.py _step_hit_keys, PR #176).
# This overlay replaces exactly ONE file, tiering/manager.py, with a version
# that contains BOTH the tierpin fix and the admission filter — see PATCH.diff
# (diffed against the tierpin file, not against stock).
#
# What the filter changes (see README.md for the full design):
#   cascades to STORAGE (fs/SSD) secondary tiers are frequency-gated — a
#   chunk is written to fs only once it has been found (lookup HIT) in the
#   primary/RAM tier at least MYIA_FS_ADMIT_MIN_HITS times (default 2).
#   Production motivation: 427.5 GB written to the fs tier in 8.9 h with
#   only 9.3% ever read back (measured 2026-10-25). Prior art: NVIDIA Dynamo
#   disk-offload filter (frequency >= 2, on by default), SGLang HiCache
#   write_through_selective. A refused store costs a recompute later, never
#   wrong output.
#
#   MYIA_FS_ADMIT_MIN_HITS=0 restores exact upstream write-through for every
#   tier (the rollback switch; read once at manager construction, so it needs
#   a container recreate).
#   MYIA_FS_ADMIT_GDN_LATEST=1 only logs: the GDN latest-state rule is a
#   documented no-op stub at this layer (no conversation-chain identity in
#   the tiering manager — see README for where the rule belongs).
#
# Tests (tests/, run in a throwaway CPU-only container of the base image):
#   stock   (tierpin base) -> filter-on tests fail: first store IS cascaded
#                             (the behaviour this overlay removes);
#   patched -> 7 admission tests + upstream v0.32.0 tiering suites pass
#             (test_tiering_offloading.py at default env — CPU-medium tiers
#             are unfiltered; test_fs_tier.py + cpu/test_manager.py with
#             MYIA_FS_ADMIT_MIN_HITS=0, which the rollback mode must keep
#             byte-identical to upstream, fs events included).
#
# Base is the production tierpin image (itself pinned to the stock v0.32.0
# digest), so this overlay never reverts the race fix.

FROM vllm-openai-v0320-tierpin:v1
COPY vllm/v1/kv_offload/tiering/manager.py /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/manager.py
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload \
 && python3 -c "from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager as M; import inspect; src = inspect.getsource(M); assert '_step_hit_keys' in src, 'tierpin fix lost'; assert '_record_fs_admit_hit' in src, 'fs-admit overlay missing'"
LABEL org.myia.overlay="v0320-fsadmit" org.myia.base="vllm-openai-v0320-tierpin:v1"
