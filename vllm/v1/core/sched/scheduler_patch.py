# -*- coding: utf-8 -*-
"""Fix for #46453: per-group prefix-hit divergence in hybrid Mamba + KV connector

Root cause: scheduler.py L715 uses max(per_group_hits) which picks the deepest
hit (usually Mamba state blocks that survive longer than FA blocks). When FA
blocks are evicted but Mamba state survives, max() over-reports the local
computed length → connector accesses beyond valid FA block range → engine crash
or silent correctness bug.

Fix:
1. Use FA (FullAttention) group hit as the safe boundary (per comment intent)
2. Trim non-FA groups' block lists to the FA boundary to maintain Mamba state
   consistency (prevents has_initial_states misjudgment from #43090)

FA group is guaranteed to be attention_groups[0] by the sort at
kv_cache_coordinator.py L593-594.

References: #43090, #43884, #47491
"""

# ═══════════════════════════════════════════════════════════
# Patch: scheduler.py L715
# ═══════════════════════════════════════════════════════════

# OLD (buggy):
#     num_new_local_computed_tokens = max(per_group_hits)

# NEW (correct):
FA_HIT_FIX = """\
                        # NOTE(ZhanqiuHu): For Mamba hybrid models,
                        # num_new_local_computed_tokens should be the FA hit
                        # because only FA blocks are guaranteed GPU-resident.
                        # Using max() picked the deeper Mamba hit and caused
                        # OOB access when FA blocks were evicted (#46453).
                        # attention_groups[0] is always FullAttention by sort
                        # in kv_cache_coordinator.py L593-594.
                        fa_group_idx = 0  # FA is always first after sort
                        num_new_local_computed_tokens = per_group_hits[fa_group_idx]
"""

# ═══════════════════════════════════════════════════════════
# Patch: Mamba state trimming after min boundary
# ═══════════════════════════════════════════════════════════

# After computing num_new_local_computed_tokens, trim non-FA per-group
# block lists to the safe boundary to prevent Mamba from believing
# it has state beyond the FA boundary (has_initial_states class bug).
MAMBA_TRIM_FIX = """\
                        # Trim Mamba/SSM group block lists to FA boundary.
                        # Mamba state blocks beyond the FA hit are not
                        # GPU-resident and would cause has_initial_states
                        # misjudgment (#43090) or connector crash (#46453).
                        fa_boundary = num_new_local_computed_tokens
                        for gid in range(len(new_computed_blocks)):
                            if gid == fa_group_idx:
                                continue
                            # Trim blocks beyond FA boundary
                            blocks = new_computed_blocks[gid]
                            trimmed = tuple(
                                b for b in blocks
                                if b is not None and b.start_idx < fa_boundary
                            )
                            new_computed_blocks = (
                                new_computed_blocks[:gid]
                                + (trimmed,)
                                + new_computed_blocks[gid + 1:]
                            )
"""
