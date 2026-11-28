# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agree on eligibility and map TP2 IPC using Unix socket fd transfer."""

import os
import socket
import tempfile

import torch
import torch.distributed as dist


def agree(reasons, group, what):
    votes: list[list[str] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(votes, reasons, group=group)
    errors: list[str] = []
    for rank, rows in enumerate(votes):
        if rows is None:
            raise RuntimeError("MiniCPM5 mono incomplete TP agreement")
        errors.extend(f"rank{rank}: {reason}" for reason in rows)
    if errors:
        raise ValueError(f"MiniCPM5 mono {what}: " + "; ".join(errors))


def exchange_fd(fd, rank, group):
    paths: list[str | None] = [None, None]
    server = path = directory = None
    try:
        reasons = []
        if rank == 0:
            try:
                directory = tempfile.mkdtemp(prefix="vllm_minicpm5_")
                path = os.path.join(directory, "peer.sock")
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.settimeout(30)
                server.bind(path)
                server.listen(1)
            except OSError as error:
                reasons.append(str(error))
        agree(reasons, group, "IPC socket setup")
        dist.all_gather_object(paths, path, group=group)
        if server is not None:
            connection, _ = server.accept()
        else:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(30)
            assert paths[0] is not None
            connection.connect(paths[0])
        with connection:
            connection.settimeout(30)
            socket.send_fds(connection, [b"f"], [fd])
            _, fds, _, _ = socket.recv_fds(connection, 1, 1)
        if len(fds) != 1:
            raise RuntimeError("MiniCPM5 IPC fd transfer failed")
        return fds[0]
    finally:
        if server is not None:
            server.close()
        if path is not None and os.path.exists(path):
            os.unlink(path)
        if directory is not None:
            os.rmdir(directory)


def region(layers, tp_group, ops):
    result = None
    reasons = []
    try:
        result = torch.classes.minicpm5_mono.Region(layers * ops.mailbox_bytes())
    except RuntimeError as error:
        reasons.append(str(error))
    agree(reasons, tp_group.cpu_group, "IPC allocation")
    assert result is not None
    if tp_group.world_size == 1:
        return result
    reasons = []
    info = None
    try:
        info = result.export_info()
    except RuntimeError as error:
        reasons.append(str(error))
    agree(reasons, tp_group.cpu_group, "IPC export")
    assert info is not None
    peers = [None, None]
    dist.all_gather_object(peers, info, group=tp_group.cpu_group)
    fd = None
    reasons = []
    try:
        fd = exchange_fd(int(info[2]), tp_group.rank_in_group, tp_group.cpu_group)
        result.open(peers[1 - tp_group.rank_in_group], fd)
    except (OSError, RuntimeError) as error:
        if fd is not None:
            os.close(fd)
        reasons.append(str(error))
    agree(reasons, tp_group.cpu_group, "IPC mapping")
    return result
