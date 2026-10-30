# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Finite UE8M0 group-scale variation, applied before observation/timing."""


def apply_scale_profile(torch, x, profile):
    if profile == "unit":
        return
    if profile != "powers":
        raise ValueError(profile)
    raw_q = x["qs"].view(torch.uint8)
    raw_q.copy_(
        torch.randint(
            124,
            131,
            raw_q.shape,
            device=raw_q.device,
            dtype=torch.uint8,
            generator=torch.Generator(device=raw_q.device).manual_seed(1743),
        )
    )
    page = x["kv"].shape[1]
    raw_k = x["kv"].squeeze(2).view(x["kv"].shape[0], page * 68)
    scales = raw_k[:, page * 64 :]
    scales.copy_(
        torch.randint(
            124,
            131,
            scales.shape,
            device=scales.device,
            dtype=torch.uint8,
            generator=torch.Generator(device=scales.device).manual_seed(1744),
        )
    )


def scale_receipt(torch, x, profile):
    """Read scale extrema only during input setup, before graph or timing."""
    page = x["kv"].shape[1]
    q = x["qs"].view(torch.uint8)
    k = x["kv"].squeeze(2).view(x["kv"].shape[0], page * 68)[:, page * 64 :]
    q_range = [int(q.amin().item()), int(q.amax().item())]
    k_range = [int(k.amin().item()), int(k.amax().item())]
    expected = [127, 127] if profile == "unit" else [124, 130]
    assert q_range == k_range == expected
    return dict(
        profile=profile,
        query_scale_range=q_range,
        key_scale_range=k_range,
        scope="Observed during input setup outside capture/timing",
    )
