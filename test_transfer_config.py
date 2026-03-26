# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick sanity check for HeteroTPTransferConfig.
Run: python test_transfer_config.py
"""

import sys

sys.path.insert(0, ".")

from vllm.distributed.kv_transfer.kv_connector.v1.hma_transfer_config import (
    HeteroTPTransferConfig,
)


def test_config(
    label,
    *,
    d_tp,
    p_tp,
    d_rank=0,
    K=2,
    d_block_len=8192,
    p_block_len=4096,
    has_mamba=True,
    tp_ratio=None,
):
    if tp_ratio is None:
        tp_ratio = -(p_tp // d_tp) if p_tp > d_tp else (d_tp // p_tp)
    cfg = HeteroTPTransferConfig(
        tp_ratio=tp_ratio,
        K=K,
        d_tp=d_tp,
        p_tp=p_tp,
        d_rank=d_rank,
        has_mamba=has_mamba,
        use_mla=False,
        d_block_len=d_block_len,
        p_block_len=p_block_len,
        is_blocks_first=True,
    )
    print(f"\n{'=' * 60}")
    print(f"  {label}  (d_rank={d_rank})")
    print(f"{'=' * 60}")
    print(f"  d_tp={cfg.d_tp}, p_tp={cfg.p_tp}, tp_ratio={cfg.tp_ratio}")
    print(f"  K={cfg.K}")
    print(
        f"  d_physical_heads={cfg.d_physical_heads}, "
        f"p_physical_heads={cfg.p_physical_heads}"
    )
    print(
        f"  is_d_replicated={cfg.is_d_replicated}, "
        f"is_p_replicated={cfg.is_p_replicated}"
    )
    print(f"  physical_fa_num_reads={cfg.physical_fa_num_reads}")
    print(f"  mamba_num_reads={cfg.mamba_num_reads}")
    print(f"  fa_read_targets={cfg.fa_read_targets}")
    print(f"  transfer_targets={cfg.transfer_targets}")
    print(f"  fa_entry_size={cfg.fa_entry_size}")
    print(f"  needs_split_handles={cfg.needs_split_handles}")
    print(f"  D_K_half={cfg.d_block_len // 2}, P_K_half={cfg.p_block_len // 2}")

    # Validate FA sizes match
    d_k_half = cfg.d_block_len // 2
    p_k_half = cfg.p_block_len // 2
    if cfg.tp_ratio < 0:
        local_chunk = d_k_half // max(1, cfg.physical_fa_num_reads)
        remote_chunk = p_k_half
    elif cfg.tp_ratio > 0:
        local_chunk = d_k_half
        remote_chunk = d_k_half  # D indexes into P
    else:
        local_chunk = d_k_half
        remote_chunk = p_k_half
    sizes_ok = "OK" if local_chunk == remote_chunk else "MISMATCH!"
    entry_ok = "OK" if cfg.fa_entry_size == min(d_k_half, p_k_half) else "BAD!"
    print(f"  D_K_half={d_k_half}, P_K_half={p_k_half}")
    print(
        f"  Local FA chunk={local_chunk}, Remote FA chunk={remote_chunk}  [{sizes_ok}]"
    )
    print(
        f"  fa_entry_size={cfg.fa_entry_size} == "
        f"min(D,P)={min(d_k_half, p_k_half)}  [{entry_ok}]"
    )

    # Check FA targets subset of transfer_targets
    tt_set = set(cfg.transfer_targets)
    orphans = [r for r in cfg.fa_read_targets if r not in tt_set]
    if orphans:
        print(f"  *** ORPHANED FA TARGETS: {orphans} not in transfer_targets! ***")
    else:
        print("  FA targets ⊆ transfer_targets: OK")

    # Show should_skip_fa for each transfer target
    for r in cfg.transfer_targets:
        skip = cfg.should_skip_fa(r)
        slot = cfg.fa_head_slot(r) if not skip else "N/A"
        print(f"    P rank {r}: skip_fa={skip}, fa_slot={slot}")

    return cfg


print("Nemotron-Nano-30B parameters: K=2, head_dim=128, page_size=16, dtype=bf16")
print("  D block_len=8192 (2 heads), P block_len=4096 (1 head) when P_TP≥K")

# 2p1d: D_TP=1, P_TP=2  (the known-good baseline)
test_config("2p1d", d_tp=1, p_tp=2, d_rank=0, d_block_len=8192, p_block_len=4096)

# 4p1d: D_TP=1, P_TP=4  (the main bug target)
test_config("4p1d", d_tp=1, p_tp=4, d_rank=0, d_block_len=8192, p_block_len=4096)

# 4p2d: D_TP=2, P_TP=4
test_config("4p2d", d_tp=2, p_tp=4, d_rank=0, d_block_len=4096, p_block_len=4096)
test_config("4p2d", d_tp=2, p_tp=4, d_rank=1, d_block_len=4096, p_block_len=4096)

# 1p4d: D_TP=4, P_TP=1  (tp_ratio > 0)
test_config("1p4d", d_tp=4, p_tp=1, d_rank=0, d_block_len=4096, p_block_len=8192)
test_config("1p4d", d_tp=4, p_tp=1, d_rank=1, d_block_len=4096, p_block_len=8192)

# 1p1d: D_TP=1, P_TP=1  (trivial)
test_config(
    "1p1d", d_tp=1, p_tp=1, d_rank=0, tp_ratio=0, d_block_len=8192, p_block_len=8192
)

# 2p2d: D_TP=2, P_TP=2  (equal TP)
test_config(
    "2p2d", d_tp=2, p_tp=2, d_rank=0, tp_ratio=0, d_block_len=4096, p_block_len=4096
)

print("\n\nDone. Check for any MISMATCH above.")
