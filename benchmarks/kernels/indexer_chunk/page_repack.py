# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep logical MXFP4 keys while changing only physical page grouping64→128."""


def repack_page128(torch, x, q_len, quant_view, schedule, num_sms):
    rows = x["q"].shape[0]
    first = torch.arange(0, rows, q_len, device=x["q"].device)
    table = x["bt"][first].long()
    requests, old_pages = table.shape
    new_pages = (old_pages + 1) // 2
    raw = x["kv"].squeeze(2).contiguous().view(x["kv"].shape[0], 64 * 68)
    old_data = raw[:, : 64 * 64].view(-1, 64, 64)[table].reshape(requests, -1, 64)
    old_scales = raw[:, 64 * 64 :].view(-1, 64, 4)[table].reshape(requests, -1, 4)
    data = torch.zeros(
        (requests, new_pages * 128, 64), dtype=torch.uint8, device=raw.device
    )
    scales = torch.full(
        (requests, new_pages * 128, 4), 127, dtype=torch.uint8, device=raw.device
    )
    data[:, : old_pages * 64].copy_(old_data)
    scales[:, : old_pages * 64].copy_(old_scales)
    logical = torch.empty(
        (requests * new_pages, 128 * 68), dtype=torch.uint8, device=raw.device
    )
    logical[:, : 128 * 64].copy_(data.reshape(-1, 128 * 64))
    logical[:, 128 * 64 :].copy_(scales.reshape(-1, 128 * 4))
    order = torch.randperm(
        requests * new_pages,
        device=raw.device,
        generator=torch.Generator(device=raw.device).manual_seed(1742),
    )
    allocation = torch.empty_like(logical)
    allocation[order] = logical
    counts = torch.full((requests,), q_len, device=raw.device, dtype=torch.int64)
    counts[-1] = rows - (requests - 1) * q_len
    result = dict(x)
    result["kv"] = quant_view(allocation.view(-1, 128, 68), 128, True)
    result["bt"] = (
        order.to(torch.int32)
        .view(requests, new_pages)
        .repeat_interleave(counts, dim=0)
        .contiguous()
    )
    result["schedule"] = schedule(x["ctx"], 128, num_sms, indices=x["indices"])
    return result


def capacity_block_table(torch, x, width):
    """Use the configured compressed capacity's stride; invalid columns are0."""
    page = x["kv"].shape[1]
    columns = (width + page - 1) // page
    old = x["bt"]
    assert old.shape[1] <= columns
    padded = torch.zeros((old.shape[0], columns), dtype=old.dtype, device=old.device)
    padded[:, : old.shape[1]].copy_(old)
    result = dict(x)
    result["bt"] = padded
    return result


def align_cache_pages(torch, x, quant_view):
    """Use the indexer spec's minimum512-byte page alignment, not slab packing."""
    page = x["kv"].shape[1]
    content = page * 68
    stride = (content + 511) // 512 * 512
    storage = torch.zeros(
        (x["kv"].shape[0], stride), device=x["kv"].device, dtype=torch.uint8
    )
    storage[:, :content].copy_(x["kv"].squeeze(2).view(x["kv"].shape[0], content))
    result = dict(x)
    result["page_storage"] = storage
    result["kv"] = quant_view(storage[:, :content].view(-1, page, 68), 128, True)
    return result
