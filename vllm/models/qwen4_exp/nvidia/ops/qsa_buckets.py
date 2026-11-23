# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA configs by integer work count, architecture, dtype and query profile."""

from vllm.kernel_buckets import MBucketTable, PhaseBucketTable

QSA_CONFIG_BUCKETS = {
    ("default", False, False): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513),
        values=(
            (32, 64, 4),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
        ),
    ),
    ("default", False, True): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513, 2049),
        values=(
            (32, 64, 4),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
            (32, 1, 1),
        ),
    ),
    ("default", True, False): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513),
        values=(
            (32, 64, 4),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
        ),
    ),
    ("default", True, True): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513, 2049),
        values=(
            (32, 64, 4),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
            (32, 1, 1),
        ),
    ),
    ("sm90", False, False): MBucketTable(
        stops=(33, 65, 257, 2049),
        values=((64, 64, 2), (32, 16, 1), (32, 8, 1), (32, 4, 1), (64, 1, 2)),
    ),
    ("sm90", False, True): MBucketTable(
        stops=(33, 65, 257, 2049),
        values=((64, 64, 2), (32, 16, 1), (32, 8, 1), (32, 4, 1), (32, 1, 1)),
    ),
    ("sm90", True, False): MBucketTable(
        stops=(25, 33, 65, 129, 257, 2049),
        values=(
            (64, 64, 2),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (32, 4, 1),
            (64, 1, 2),
        ),
    ),
    ("sm90", True, True): MBucketTable(
        stops=(25, 33, 65, 129, 257, 2049),
        values=(
            (64, 64, 2),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 1, 1),
        ),
    ),
    ("sm100", False, False): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513),
        values=(
            (64, 64, 2),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
        ),
    ),
    ("sm100", False, True): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513, 2049),
        values=(
            (64, 64, 2),
            (32, 16, 1),
            (32, 8, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
            (32, 1, 1),
        ),
    ),
    ("sm100", True, False): MBucketTable(
        stops=(5, 25, 33, 65, 129, 257, 513),
        values=(
            (32, 64, 4),
            (64, 64, 2),
            (32, 32, 1),
            (32, 16, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
        ),
    ),
    ("sm100", True, True): MBucketTable(
        stops=(5, 25, 33, 65, 129, 257, 513, 2049),
        values=(
            (32, 64, 4),
            (64, 64, 2),
            (32, 32, 1),
            (32, 16, 1),
            (32, 4, 1),
            (32, 8, 1),
            (64, 4, 2),
            (64, 1, 2),
            (32, 1, 1),
        ),
    ),
    ("sm120", False, False): MBucketTable(
        stops=(25, 65, 129, 257, 513, 2049),
        values=(
            (64, 64, 8),
            (64, 8, 8),
            (32, 8, 8),
            (32, 8, 1),
            (64, 4, 2),
            (32, 4, 1),
            (64, 2, 2),
        ),
    ),
    ("sm120", False, True): MBucketTable(
        stops=(25, 65, 129, 257, 513, 2049),
        values=(
            (64, 64, 8),
            (64, 8, 8),
            (32, 8, 8),
            (32, 8, 1),
            (64, 4, 2),
            (32, 4, 1),
            (64, 2, 2),
        ),
    ),
    ("sm120", True, False): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513, 2049),
        values=(
            (64, 64, 8),
            (128, 8, 4),
            (64, 8, 4),
            (32, 4, 4),
            (128, 4, 4),
            (64, 4, 4),
            (32, 2, 1),
            (64, 1, 2),
        ),
    ),
    ("sm120", True, True): MBucketTable(
        stops=(25, 33, 65, 129, 257, 513),
        values=(
            (64, 64, 8),
            (128, 8, 4),
            (64, 8, 4),
            (32, 4, 4),
            (128, 4, 4),
            (64, 4, 4),
            (32, 2, 1),
        ),
    ),
}

QSA_PHASE_BUCKETS = {
    key: PhaseBucketTable(
        operator="vllm.models.qwen4_exp.nvidia.ops.qsa.qsa_sparse_paged_attention",
        axis="M * num_kv_heads",
        fallback=table,
        prefill=table,
        decode=table,
        verify=table,
    )
    for key, table in QSA_CONFIG_BUCKETS.items()
}
