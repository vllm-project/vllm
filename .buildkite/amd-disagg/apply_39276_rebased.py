#!/usr/bin/env python3
"""Rebased vLLM PR #39276 fixes for multi-node DP MoRIIO disaggregation.

This runtime patch script:
  - Applies the 5 hunks from upstream PR #39276 (engine_id, moriio_common,
    moriio_connector, etc.) - idempotent: skipped if image already has them baked.
  - Applies PR#39276 H1a + H1b: skip dummy MoE forwards during boot when
    enforce_eager + preset kv_cache_memory_bytes (avoids MoRI internode
    lanePe device assert on cold buffers at EP > local).
  - Strips legacy pre-H1 boot extras (_ep_profile_barrier method, the
    VLLM_SKIP_PROFILE_SAMPLER and VLLM_EP_MOE_PROFILE_BARRIER env-gated
    hunks) if the image baked them in.

Idempotent and safe to re-run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Idempotent anchor replacement.

    If the anchor isn't found we WARN and return the text unchanged, rather
    than aborting the whole patcher.  This lets the bulk applicator handle
    images where some hunks are already baked in (e.g. v1.1.4_pr366_v2_v3
    bakes H1 + H11 with different syntax) and still apply the remaining
    H2-H9 hunks that haven't been baked in.
    """
    if old not in text:
        print(f"[39276-rebased] WARN: anchor not found for {label} (skipping; likely already baked in or vllm refactored)")
        return text
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# PR #39276 hunks (idempotent: skip if already present)
# ---------------------------------------------------------------------------

def patch_core_py(path: Path) -> None:
    text = path.read_text()
    if "engine_id}_dp{dp_rank}" in text:
        print(f"  {path.name}: already patched, skipping")
        return
    text = replace_once(
        text,
        """            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )""",
        """            if data_parallel and vllm_config.kv_transfer_config is not None:
                # Use global dp_rank (not local_dp_rank) so engine_ids are unique
                # across nodes in multi-node data parallel deployments.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
                )""",
        "core.py engine_id",
    )
    if "HANDSHAKE_TIMEOUT_MINS = 5" in text:
        text = replace_once(
            text,
            "HANDSHAKE_TIMEOUT_MINS = 5",
            'HANDSHAKE_TIMEOUT_MINS = int(os.environ.get("VLLM_HANDSHAKE_TIMEOUT_MINS", "5"))',
            "core.py handshake timeout",
        )
    path.write_text(text)
    print(f"  patched {path.name}")


def patch_moriio_common_py(path: Path) -> None:
    text = path.read_text()
    if "data_parallel_size_local" in text and "dp_rank = (" in text:
        print(f"  {path.name}: already patched, skipping")
        return
    text = replace_once(
        text,
        """        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        base_notify_port = int(extra_config["notify_port"])
        dp_size = vllm_config.parallel_config.data_parallel_size""",
        """        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = (
            vllm_config.parallel_config.data_parallel_rank
            % vllm_config.parallel_config.data_parallel_size_local
        )
        base_notify_port = int(extra_config["notify_port"])
        dp_size = vllm_config.parallel_config.data_parallel_size_local""",
        "moriio_common local DP",
    )
    path.write_text(text)
    print(f"  patched {path.name}")


def patch_is_kv_master_local_rank(path: Path) -> None:
    """Surgical fix for the v1.1.4_pr366_v2_v3 image (and any newer baked-in
    variant) where `_is_kv_master` already exists but still uses the GLOBAL
    `data_parallel_rank`.  At 2P/2D EP=16 the decode child node has global
    ranks 8-15 -> _is_kv_master=False -> `send_notify_block` is suppressed ->
    prefill side waits 120s for block notify that decode never sends ->
    "Deferred write task expired (remote blocks never arrived)".

    Fix: replace with NODE-LOCAL rank check so every local rank 0..7 on every
    pod becomes a kv_master and can send notify.

    Anchor: the broken block as baked into the image.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# 2P/2D LANEPE FIX: use node-local rank for _is_kv_master"
    if marker in text:
        print(f"  {path.name}: _is_kv_master local-rank fix already applied")
        return
    old = (
        "        self._is_kv_master = (\n"
        "            self.vllm_config.parallel_config.data_parallel_rank\n"
        "            < self.vllm_config.parallel_config.data_parallel_size_local\n"
        "        )"
    )
    new = (
        "        # 2P/2D LANEPE FIX: use node-local rank for _is_kv_master.  The\n"
        "        # baked image still checks GLOBAL data_parallel_rank, which\n"
        "        # leaves decode child nodes (global ranks >= local_size) as\n"
        "        # non-masters - their send_notify_block is suppressed and\n"
        "        # prefill writers time out waiting for remote block notify.\n"
        "        _local_rank = self.vllm_config.parallel_config.data_parallel_rank_local\n"
        "        if _local_rank is None:\n"
        "            _local_rank = (\n"
        "                self.vllm_config.parallel_config.data_parallel_rank\n"
        "                % self.vllm_config.parallel_config.data_parallel_size_local\n"
        "            )\n"
        "        self._is_kv_master = (\n"
        "            _local_rank\n"
        "            < self.vllm_config.parallel_config.data_parallel_size_local\n"
        "        )"
    )
    if old not in text:
        print(f"  {path.name}: _is_kv_master broken anchor not found, skipping (likely upstream-different)")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (2P/2D LANEPE: _is_kv_master node-local rank)")


def patch_moriio_connector_py(path: Path) -> None:
    text = path.read_text()
    if "_req_kv_params" in text and "_is_kv_master" in text and "moriio_config.dp_rank == 0" in text:
        print(f"  {path.name}: already patched, skipping")
        return

    text = replace_once(
        text,
        """        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = self.kv_transfer_config.kv_role == "kv_producer"
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}""",
        """        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.dp_rank = (
            self.vllm_config.parallel_config.data_parallel_rank
            % self.vllm_config.parallel_config.data_parallel_size_local
        )
        # Node-local rank: headless child nodes (decode DP8-15) must send notifies;
        # global rank >= local_size on those nodes — PR #39276 global check breaks 2P/2D.
        _local_rank = self.vllm_config.parallel_config.data_parallel_rank_local
        if _local_rank is None:
            _local_rank = self.dp_rank
        self._is_kv_master = (
            _local_rank < self.vllm_config.parallel_config.data_parallel_size_local
        )
        self._global_dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = self.kv_transfer_config.kv_role == "kv_producer"
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}
        self._req_kv_params: dict[ReqId, dict] = {}""",
        "moriio_connector scheduler init",
    )

    text = replace_once(
        text,
        """        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (request, local_block_ids)

        if params is not None and params.get("do_remote_prefill"):""",
        """        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (request, local_block_ids)
            self._req_kv_params[request.request_id] = dict(params)

        if params is not None and params.get("do_remote_prefill"):""",
        "moriio_connector cache kv params (decode)",
    )

    text = replace_once(
        text,
        """                        self._reqs_need_recv[request.request_id] = (
                            request,
                            local_block_ids,
                        )
                    else:
                        logger.warning(
                            "Got invalid KVTransferParams: %s. This "
                            "request will not utilize KVTransfer",
                            params,
                        )

            else:
                # WRITE mode, decode side: notify P that blocks are ready
                assert request.kv_transfer_params is not None, (
                    "kv_transfer_params should not be None"
                )

                remote_dp_rank = request.kv_transfer_params.get("remote_dp_rank", 0)

                peer_zmq = get_peer_zmq_from_request_id(
                    request.request_id, is_producer=False
                )
                remote_host, _, remote_notify_port = parse_moriio_zmq_address(peer_zmq)

                for tp_index in range(self.tp_size):
                    target_port = remote_notify_port + get_port_offset(
                        remote_dp_rank, tp_index
                    )

                    self.send_notify_block(
                        req_id=request.request_id,
                        transfer_id=request.kv_transfer_params["transfer_id"],
                        block_notify_list=blocks.get_block_ids()[0],
                        host=remote_host,
                        port=target_port,
                    )""",
        """                        self._reqs_need_recv[request.request_id] = (
                            request,
                            local_block_ids,
                        )
                        self._req_kv_params[request.request_id] = dict(params)
                    else:
                        logger.warning(
                            "Got invalid KVTransferParams: %s. This "
                            "request will not utilize KVTransfer",
                            params,
                        )

            else:
                # WRITE mode, decode side: notify P that blocks are ready
                assert request.kv_transfer_params is not None, (
                    "kv_transfer_params should not be None"
                )

                remote_dp_rank = request.kv_transfer_params.get("remote_dp_rank", 0)

                if self._is_kv_master:
                    peer_zmq = get_peer_zmq_from_request_id(
                        request.request_id, is_producer=False
                    )
                    remote_host, _, remote_notify_port = parse_moriio_zmq_address(
                        peer_zmq
                    )

                    for tp_index in range(self.tp_size):
                        target_port = remote_notify_port + get_port_offset(
                            remote_dp_rank, tp_index
                        )

                        self.send_notify_block(
                            req_id=request.request_id,
                            transfer_id=request.kv_transfer_params["transfer_id"],
                            block_notify_list=blocks.get_block_ids()[0],
                            host=remote_host,
                            port=target_port,
                        )""",
        "moriio_connector master-only notify + kv cache",
    )

    text = replace_once(
        text,
        """                        ):
                            meta.add_new_req(
                                request_id=req_id,
                                local_block_ids=self._reqs_need_pending_save[req_id][1],
                                kv_transfer_params=req.kv_transfer_params or {},
                                write_mode=True,
                            )
                            del self._reqs_need_pending_save[req_id]

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            if req.num_prompt_tokens > len(block_ids) * self.block_size:
                # not last chunk prefill
                self._reqs_need_pending_save[req_id] = (req, block_ids)
                continue
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                write_mode=True,
            )
        # Clear the list once workers start the transfers

        meta.reqs_to_send = self._reqs_need_send

        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_need_send = {}""",
        """                        ):
                            kv_params = self._req_kv_params.pop(
                                req_id, req.kv_transfer_params or {}
                            )
                            meta.add_new_req(
                                request_id=req_id,
                                local_block_ids=self._reqs_need_pending_save[req_id][1],
                                kv_transfer_params=kv_params,
                                write_mode=True,
                            )
                            del self._reqs_need_pending_save[req_id]

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=kv_params,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})
            if req.num_prompt_tokens > len(block_ids) * self.block_size:
                # not last chunk prefill
                self._reqs_need_pending_save[req_id] = (req, block_ids)
                continue
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=kv_params,
                write_mode=True,
            )
        # Clear the list once workers start the transfers

        meta.reqs_to_send = self._reqs_need_send

        for req_id in self._reqs_need_recv:
            self._req_kv_params.pop(req_id, None)
        for req_id in self._reqs_need_save:
            if req_id not in self._reqs_need_pending_save:
                self._req_kv_params.pop(req_id, None)
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_need_send = {}""",
        "moriio_connector build_connector_meta kv cache",
    )

    if "self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}" in text:
        text = replace_once(
            text,
            """        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}""",
            """        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}
        self._recving_transfers_start: dict[str, float] = {}""",
            "worker _recving_transfers_start",
        )

    if "assert block_size == self.block_size" in text:
        text = replace_once(
            text,
            "        assert block_size == self.block_size",
            """        if block_size != self.block_size:
            logger.info(
                "KV cache block_size=%d differs from config block_size=%d; "
                "using actual tensor shape (attention backend override).",
                block_size,
                self.block_size,
            )
            self.block_size = block_size""",
            "worker register_kv_caches block_size",
        )

    text = replace_once(
        text,
        """        if self._rank == 0 and self.moriio_config.proxy_ip:
            self._ping_thread = threading.Thread(
                target=self._ping, args=(self.zmq_context,), daemon=True
            )
            self._ping_thread.start()""",
        """        if (
            self.moriio_config.dp_rank == 0
            and self.tp_rank == 0
            and self.moriio_config.proxy_ip
        ):
            self._ping_thread = threading.Thread(
                target=self._ping, args=(self.zmq_context,), daemon=True
            )
            self._ping_thread.start()""",
        "moriio worker node-local proxy ping",
    )

    path.write_text(text)
    print(f"  patched {path.name}")


def patch_connector_remote_dp_rank(path: Path) -> None:
    """PR#39276 H2: producer request_finished must advertise its own (node-local)
    DP rank so the decode side routes the KV-write block-notify to the correct
    prefill notify port.

    Upstream omits remote_dp_rank from the kv_transfer_params returned by the
    producer's request_finished, so the decode's
        remote_dp_rank = kv_transfer_params.get("remote_dp_rank", 0)
    always defaults to 0 and every notify is sent to prefill DP0's port
    (base_notify_port + get_port_offset(0, tp)). Any request the DP load
    balancer schedules on a prefill rank != DP0 then never receives its
    block-notify; the deferred write expires after defer_timeout and the
    decode hangs in WAITING_FOR_REMOTE_KVS. This only manifests once
    concurrency exceeds ~1-per-rank (e.g. con16 over 8 local ranks), where
    requests actually spread off DP0.

    Applied as a standalone, independently-idempotent step so it lands even if
    an image already baked in the other moriio_connector hunks.
    """
    if not path.is_file():
        return
    text = path.read_text()
    if "remote_dp_rank=self.dp_rank" in text:
        print(f"  {path.name}: remote_dp_rank already patched, skipping")
        return
    text = replace_once(
        text,
        """        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            transfer_id=params["transfer_id"],
        )""",
        """        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            transfer_id=params["transfer_id"],
            # PR#39276 H2: node-local prefill DP rank so the decode routes the
            # KV-write notify to THIS rank's notify port, not always DP0's.
            remote_dp_rank=self.dp_rank,
        )""",
        "moriio_connector advertise prefill remote_dp_rank",
    )
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H2: advertise prefill remote_dp_rank)")


# ---------------------------------------------------------------------------
# PR #39276 H1: skip dummy MoE forwards during boot
# (avoids MoRI internode lanePe assert when EP > local on cold buffers)
# ---------------------------------------------------------------------------

def patch_moriio_engine_py(path: Path) -> None:
    """PR #39276 hunks in moriio_engine.py:
      - add `import os` (needed for downstream env reads)
      - replace ZMQ send block with retry loop using zmq.NOBLOCK
        to prevent permanent stalls from transient ZMQ failures.
    Idempotent: skips if already patched.
    """
    if not path.is_file():
        return
    text = path.read_text()
    if "_MAX_RETRIES" in text and "zmq.NOBLOCK" in text:
        print(f"  {path.name}: already patched, skipping")
        return
    # Add `import os` after the SPDX header (idempotent)
    if "import os\nimport threading" not in text:
        text = replace_once(
            text,
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\nimport threading",
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\nimport os\nimport threading",
            "moriio_engine import os",
        )
    text = replace_once(
        text,
        """        sock = self.paths[path]
        try:
            for req_id in req_list:
                if not isinstance(req_id, str):
                    logger.warning(
                        "Invalid req_id type: %s, expected str", type(req_id)
                    )
                    continue
                sock.send(req_id.encode("utf-8"))
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", path, e)
            self.paths.pop(path, None)
            raise""",
        """        sock = self.paths[path]
        _MAX_RETRIES = 3
        for req_id in req_list:
            if not isinstance(req_id, str):
                logger.warning(
                    "Invalid req_id type: %s, expected str", type(req_id))
                continue
            for _attempt in range(_MAX_RETRIES):
                try:
                    sock.send(req_id.encode("utf-8"), zmq.NOBLOCK)
                    break
                except zmq.Again:
                    if _attempt < _MAX_RETRIES - 1:
                        time.sleep(0.01 * (_attempt + 1))
                        logger.warning(
                            "ZMQ send retry %d for req %s to %s",
                            _attempt + 1, req_id, path)
                    else:
                        logger.error(
                            "ZMQ send FAILED after %d retries "
                            "for req %s to %s",
                            _MAX_RETRIES, req_id, path)
                except Exception as e:
                    logger.error(
                        "Failed to send notification to %s: %s",
                        path, e)
                    self.paths.pop(path, None)
                    raise""",
        "moriio_engine ZMQ retry",
    )
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276: ZMQ retry + import os)")


def _strip_old_boot_extras_worker(text: str) -> str:
    """Remove pre-H1 boot extras baked into the docker image:
    _ep_profile_barrier method/calls, VLLM_SKIP_PROFILE_SAMPLER guard.
    Idempotent: returns original text if not present.
    """
    text = text.replace(
        "            self._ep_profile_barrier()\n            self.model_runner.profile_run()",
        "            self.model_runner.profile_run()",
    )
    text = text.replace(
        "        self._ep_profile_barrier()\n        if kv_cache_memory_bytes",
        "        if kv_cache_memory_bytes",
    )
    text = re.sub(
        r"    def _ep_profile_barrier\(self\) -> None:\n(?:        .*\n|\n)+?"
        r"(?=    @torch\.inference_mode\(\)\n    def determine_available_memory)",
        "",
        text,
    )
    text = text.replace(
        '            else:\n'
        '                import os as _os\n'
        '                if _os.environ.get("VLLM_SKIP_PROFILE_SAMPLER", "0") != "1":\n'
        '                    self.model_runner._dummy_sampler_run(\n'
        '                        hidden_states=last_hidden_states\n'
        '                    )',
        '            else:\n'
        '                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)',
    )
    return text


def _strip_runner_boot_extras(path: Path) -> None:
    """Strip image-baked VLLM_EP_MOE_PROFILE_BARRIER + VLLM_SKIP_PROFILE_SAMPLER
    hunks from gpu_model_runner.py if present. Idempotent.
    """
    if not path.is_file():
        return
    text = path.read_text()
    original = text
    text = text.replace(
        '        hidden_states, last_hidden_states = self._dummy_run(\n'
        '            self.max_num_tokens, is_profile=True\n'
        '        )\n'
        '        import os as _os\n'
        '        if _os.environ.get("VLLM_EP_MOE_PROFILE_BARRIER", "1") != "0":\n'
        '            _pc = self.vllm_config.parallel_config\n'
        '            if (\n'
        '                _pc.data_parallel_size > _pc.data_parallel_size_local\n'
        '                and torch.distributed.is_initialized()\n'
        '            ):\n'
        '                from vllm.distributed.parallel_state import get_dp_group\n'
        '\n'
        '                get_dp_group().barrier()\n'
        '        if get_pp_group().is_last_rank:',
        '        hidden_states, last_hidden_states = self._dummy_run(\n'
        '            self.max_num_tokens, is_profile=True\n'
        '        )\n'
        '        if get_pp_group().is_last_rank:',
    )
    text = text.replace(
        '            else:\n'
        '                import os as _os\n'
        '                if _os.environ.get("VLLM_SKIP_PROFILE_SAMPLER", "0") != "1":\n'
        '                    output = self._dummy_sampler_run(last_hidden_states)\n'
        '                else:\n'
        '                    output = None\n'
        '        else:\n'
        '            output = None',
        '            else:\n'
        '                output = self._dummy_sampler_run(last_hidden_states)\n'
        '        else:\n'
        '            output = None',
    )
    if text != original:
        path.write_text(text)
        print(f"  stripped legacy boot extras from {path.name}")


def patch_skip_profile_run(path: Path) -> None:
    """PR#39276 H1a + H1b: skip dummy MoE forwards during boot when
    kv_cache_memory_bytes preset AND enforce_eager=True.

    H1a: profile_run() in determine_available_memory (kv_cache_memory_bytes branch)
    H1b: _dummy_run() in compile_or_warm_up_model (sampler warmup branch)

    Both call MoE all2all internode kernel which asserts lanePe on cold buffers
    when EP > local. With enforce_eager + preset KV memory, both runs are pure
    cudagraph/fragmentation optimizations; safe to skip.
    """
    text = path.read_text()
    text = _strip_old_boot_extras_worker(text)

    h1a_present = "PR#39276 H1: profile_run skipped" in text
    h1b_present = "PR#39276 H1b" in text

    if not h1a_present:
        text = replace_once(
            text,
            """        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()""",
            """        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # PR#39276 H1: profile_run skipped when kv_cache_memory_bytes is preset.
            # The profile dummy forward runs at max_num_batched_tokens and compiles
            # a graph that, on the v1.1.5 image, calls a broken AITER group fp8 quant
            # (_rocm_aiter_group_fp8_quant_impl -> per_group_quant_hip ->
            # dynamic_per_token_scaled_quant, torch_guard rejects aiter_tensor_t) and
            # crashes engine init. The model compiles lazily / during cudagraph
            # capture anyway, so skipping is safe (matches the validated v1.1.5 run).
            logger.info(
                "[PR#39276 H1] Skipping profile_run: kv_cache_memory_bytes preset "
                "(avoids broken AITER group fp8 quant on this image)."
            )""",
            "gpu_worker H1a skip profile_run",
        )

    if not h1b_present:
        text = replace_once(
            text,
            """        elif get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)""",
            """        elif get_pp_group().is_last_rank:
            # PR#39276 H1b: skip sampler warmup _dummy_run unconditionally. The dummy
            # forward at max_num_reqs compiles/runs a graph that, on the v1.1.5 image,
            # calls a broken AITER group fp8 quant (_rocm_aiter_group_fp8_quant_impl,
            # torch_guard rejects aiter_tensor_t) and crashes. Sampler tensor
            # pre-allocation is a fragmentation optimization only; the first real
            # request allocates them at small cost.
            logger.info(
                "[PR#39276 H1b] Skipping sampler warmup _dummy_run "
                "(avoids broken AITER group fp8 quant on this image)."
            )""",
            "gpu_worker H1b skip sampler warmup",
        )

    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H1a/H1b: skip dummy MoE forwards)")


# ---------------------------------------------------------------------------
# H3: Scheduler assert crash when deferred-send reaper races with worker
# ---------------------------------------------------------------------------

def patch_scheduler_kv_xfer_guard(path: Path) -> None:
    """PR#39276 H3: convert fatal assert to guard in _update_from_kv_xfer_finished.

    When update_connector_output reaps a deferred send (deadline expired) it
    injects req_id into finished_sending and removes it from self.requests.
    The worker can also report the same req_id via get_finished in a later step.
    The assert at scheduler.py:2178 then crashes the engine core:
        assert req_id in self.requests  # ← already gone

    Fix: skip unknown req_ids with a warning instead of asserting.  Blocks were
    already freed by the reaper so there is nothing left to do.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H3: guard finished_sending"
    if marker in text:
        print(f"  {path.name}: H3 already patched, skipping")
        return

    old = """\
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])"""
    new = """\
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            # H3: guard finished_sending — the deferred-send reaper in
            # update_connector_output may have already freed this request's
            # blocks and removed it from self.requests.  Skip gracefully
            # instead of crashing the engine core.
            if req_id not in self.requests:
                logger.warning(
                    "finished_sending for unknown req %s "
                    "(already reaped by deferred-send deadline); skipping",
                    req_id,
                )
                continue
            self._free_blocks(self.requests[req_id])"""
    if old not in text:
        print(f"  {path.name}: H3 anchor not found (maybe different vllm version), skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H3: guard finished_sending assert)")


# ---------------------------------------------------------------------------
# H4: Notify-listener thread resilience
# ---------------------------------------------------------------------------

def patch_notify_listener_resilience(path: Path) -> None:
    """PR#39276 H4: prevent notify-listener thread from dying on transient errors.

    The async_wait_reqid listener re-raises any exception from _handle_message,
    which kills the thread.  All subsequent notifies for that rank are lost and
    every deferred write expires, hanging the prefill instance permanently.

    Fix: catch exceptions inside the while-True loop, log, and continue.
    Only HandshakeError (a broken ZMQ socket) should propagate.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H4: resilient notify listener"
    if marker in text:
        print(f"  {path.name}: H4 already patched, skipping")
        return

    old = """\
            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    try:
                        identity, msg = sock.recv_multipart()
                        self._handle_message(msg)
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
                        raise HandshakeError(f"Error processing message: {e}") from e"""
    new = """\
            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    # H4: resilient notify listener — transient message
                    # processing errors must not kill the listener thread;
                    # only a fatal ZMQ/socket error should propagate.
                    try:
                        identity, msg = sock.recv_multipart()
                    except Exception as e:
                        logger.error(
                            "Fatal ZMQ recv error on notify listener: %s", e)
                        raise HandshakeError(
                            f"Notify listener recv failed: {e}") from e
                    try:
                        self._handle_message(msg)
                    except (HandshakeError, zmq.ZMQError):
                        raise
                    except Exception as e:
                        logger.error(
                            "Error processing notify message (continuing): %s",
                            e, exc_info=True)"""
    if old not in text:
        print(f"  {path.name}: H4 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H4: resilient notify listener)")


# ---------------------------------------------------------------------------
# H5: send_notify_block — non-blocking ZMQ with retry
# ---------------------------------------------------------------------------

def patch_send_notify_block_robustness(path: Path) -> None:
    """PR#39276 H5: send_notify_block uses blocking send with no timeout.

    At high concurrency the ZMQ HWM can be hit and the blocking send stalls
    the scheduler thread (update_state_after_alloc runs in the scheduler
    process). Switch to zmq.NOBLOCK with retries, matching the engine's
    send_notify already patched by the ZMQ-retry hunk.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H5: non-blocking send_notify_block"
    if marker in text:
        print(f"  {path.name}: H5 already patched, skipping")
        return

    old = """\
        serialized_data = msgpack.dumps(data)
        self.paths[path].send(serialized_data)"""
    new = """\
        serialized_data = msgpack.dumps(data)
        # H5: non-blocking send_notify_block with retry — prevents the
        # scheduler from stalling when ZMQ HWM is hit at high concurrency.
        _MAX_RETRIES = 5
        for _attempt in range(_MAX_RETRIES):
            try:
                self.paths[path].send(serialized_data, zmq.NOBLOCK)
                break
            except zmq.Again:
                if _attempt < _MAX_RETRIES - 1:
                    import time as _time
                    _time.sleep(0.01 * (_attempt + 1))
                    logger.warning(
                        "send_notify_block ZMQ retry %d for req %s to %s",
                        _attempt + 1, req_id, path)
                else:
                    logger.error(
                        "send_notify_block FAILED after %d retries "
                        "for req %s to %s — decode will not receive "
                        "block-notify for this request",
                        _MAX_RETRIES, req_id, path)"""
    if old not in text:
        print(f"  {path.name}: H5 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H5: non-blocking send_notify_block)")


# ---------------------------------------------------------------------------
# H6: Scheduler dp_rank should be node-local for multi-node DP (4P/4D ready)
# ---------------------------------------------------------------------------

def patch_scheduler_local_dp_rank(path: Path) -> None:
    """PR#39276 H6: MoRIIOConnectorScheduler uses global dp_rank.

    In 2P/2D the scheduler-side dp_rank controls port offset and the
    decode_rank field in send_notify_block.  Using the global rank makes
    decode nodes with dp_rank >= dp_size_local compute wrong port offsets
    for the prefill notify listener.

    Fix: use dp_rank % dp_size_local (same as the worker side).  This is
    forward-compatible with 4P/4D where dp_size_local < dp_size.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H6: node-local scheduler dp_rank"
    if marker in text:
        print(f"  {path.name}: H6 already patched, skipping")
        return

    old = """\
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = self.kv_transfer_config.kv_role == "kv_producer\""""
    new = """\
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        # H6: node-local scheduler dp_rank — use modular rank so port
        # offsets and decode_rank in send_notify_block are correct on
        # headless child nodes (DP8-15 in 2P/2D, future 4P/4D).
        self.dp_rank = (
            self.vllm_config.parallel_config.data_parallel_rank
            % self.vllm_config.parallel_config.data_parallel_size_local
        )
        self.is_producer = self.kv_transfer_config.kv_role == "kv_producer\""""
    if old not in text:
        print(f"  {path.name}: H6 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H6: node-local scheduler dp_rank)")


# ---------------------------------------------------------------------------
# H7: EP warmup barrier — pre-load AITER fused-MoE kernels before serving
# ---------------------------------------------------------------------------

def patch_ep_warmup_barrier(path: Path) -> None:
    """PR#39276 H7: run a minimal 1-token dummy forward after boot to trigger
    lazy AITER fused-MoE kernel loading (hsaco LoadKernel) on all EP ranks,
    then barrier so every rank is ready before serving starts.

    Without this, the first real request triggers unsynchronized kernel loading
    across 16 EP ranks.  Fast ranks enter the MoRI EP all-to-all collective
    while slow ranks are still in hipModuleLoad, creating a permanent deadlock.

    num_tokens=1 is small enough to avoid the lanePe assert that H1a/H1b guard
    against (that only fires at max_num_tokens sized batches on cold buffers).
    """
    if not path.is_file():
        return
    text = path.read_text()
    # H7v2 marker — extended to cover attention kernels in addition to MoE.
    marker_v2 = "EP+attn warmup forward"
    if marker_v2 in text:
        print(f"  {path.name}: H7v2 already patched, skipping")
        return

    # Detect old H7 (MoE-only warmup) and revert it so we can re-apply v2.
    old_h7_start = "# H7: EP warmup barrier — run a minimal 1-token dummy forward"
    if old_h7_start in text:
        # Find the old H7 block and revert it back to the original anchor so
        # the v2 patch can be applied cleanly.
        idx_start = text.find("        # H7: EP warmup barrier — run a minimal 1-token")
        idx_end = text.find("        # Reset the seed", idx_start)
        if idx_start != -1 and idx_end != -1:
            text = text[:idx_start] + text[idx_end:]
            print(f"  {path.name}: reverted old H7 (MoE-only) before applying v2")

    old = """\
        elif get_pp_group().is_last_rank:
            logger.info(
                "[PR#39276 H1b] Skipping sampler warmup _dummy_run: enforce_eager=True "
                "(avoids MoRI lanePe on internode EP cold buffers)."
            )

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)"""
    new = """\
        elif get_pp_group().is_last_rank:
            logger.info(
                "[PR#39276 H1b] Skipping sampler warmup _dummy_run: enforce_eager=True "
                "(avoids MoRI lanePe on internode EP cold buffers)."
            )

        # H7: EP warmup barrier + attention kernel JIT prewarm. The original
        # 1-token forward only triggered AITER MoE kernel loading, but the
        # decode-side Triton attention/MLA kernels (_fwd_grouped_kernel_stage1,
        # _fwd_kernel_stage2, _compute_slot_mapping_kernel) still JIT-compile
        # on the first real request to each rank. That cold-JIT can take 15+
        # seconds, exceeding defer_timeout and causing the MoRIIO write task
        # to expire (request hangs, blocks freed prematurely). Run multiple
        # token counts so common attention shape buckets are autotuned and
        # cached before serving.
        _ep_warmup_tokens_env = os.environ.get("VLLM_EP_WARMUP_TOKENS", "1,256")
        _ep_warmup_token_counts = []
        for _tok in _ep_warmup_tokens_env.split(","):
            _tok = _tok.strip()
            if _tok and _tok.isdigit() and int(_tok) > 0:
                _ep_warmup_token_counts.append(int(_tok))
        if (
            self.vllm_config.parallel_config.enable_expert_parallel
            and _ep_warmup_token_counts
        ):
            for _num_tokens in _ep_warmup_token_counts:
                logger.info(
                    "[PR#39276 H7] Running EP+attn warmup forward "
                    "(%d token(s)) to JIT-compile MoE and attention kernels.",
                    _num_tokens,
                )
                self.model_runner._dummy_run(
                    num_tokens=_num_tokens,
                    skip_eplb=True,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            logger.info(
                "[PR#39276 H7] EP+attn warmup barrier complete (sizes=%s).",
                _ep_warmup_token_counts,
            )

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)"""
    if old not in text:
        print(f"  {path.name}: H7 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H7: EP warmup barrier)")


# ---------------------------------------------------------------------------
# H8: dedupe deferred-task expiry log spam
# ---------------------------------------------------------------------------

def patch_deferred_expiry_dedup(path: Path) -> None:
    """PR#39276 H8: only log the deferred-task expiry ERROR once per request_id.

    A single failed KV transfer can enqueue many WriteTask objects (one per
    block/layer chunk) with the same request_id.  When the remote allocation
    never arrives, ALL of them expire together and the ERROR log fires
    hundreds of times — a 183-line burst was observed for one stuck request.
    This drowns the logs and makes triage hard.

    Fix: keep a set of request_ids already logged in this expiry pass and skip
    duplicates.  The deferred queue itself still drains correctly.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H8: dedupe expiry log"
    if marker in text:
        print(f"  {path.name}: H8 already patched, skipping")
        return

    old = """\
        defer_timeout = self._defer_timeout
        now = time.perf_counter()
        still_deferred: list[WriteTask] = []

        for task in self._deferred_tasks:
            if now - task.enqueue_time > defer_timeout:
                logger.error(
                    "Deferred write task for request %s expired after %.1fs "
                    "(remote blocks never arrived), marking done",
                    task.request_id,
                    now - task.enqueue_time,
                )
                self._mark_request_done(task.transfer_id)
                continue"""
    new = """\
        defer_timeout = self._defer_timeout
        now = time.perf_counter()
        still_deferred: list[WriteTask] = []
        # H8: dedupe expiry log — same request_id can have many block-chunk
        # tasks; log only once per req_id per pass to avoid drowning the log.
        _expired_logged: set = set()

        for task in self._deferred_tasks:
            if now - task.enqueue_time > defer_timeout:
                if task.request_id not in _expired_logged:
                    logger.error(
                        "Deferred write task for request %s expired after %.1fs "
                        "(remote blocks never arrived), marking done",
                        task.request_id,
                        now - task.enqueue_time,
                    )
                    _expired_logged.add(task.request_id)
                self._mark_request_done(task.transfer_id)
                continue"""
    if old not in text:
        print(f"  {path.name}: H8 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H8: dedupe expiry log)")


# ---------------------------------------------------------------------------
# H9 (diagnostic): trace notify send/recv to localize where the hang is
# ---------------------------------------------------------------------------

def patch_diag_notify_logging(connector_path: Path, engine_path: Path) -> None:
    """PR#39276 H9 (DIAGNOSTIC): log every send_notify_block call on decode
    and every _handle_structured_message receipt on prefill, so we can prove
    whether notifies are being sent, are arriving, and are matched correctly.

    Without this, the happy-path is silent and we cannot distinguish:
      - decode scheduler not calling update_state_after_alloc
      - decode calling send_notify but ZMQ dropping the message
      - prefill receiving but failing to match transfer_id

    This is INTENTIONALLY INFO level (not DEBUG) so it shows up in
    production-style logs without env tweaks. Volume is bounded by request
    rate (one log per notify, not per block-chunk).
    """
    marker = "# H9 diag notify"
    # ---- decode side: log send_notify_block ----
    if connector_path.is_file():
        text = connector_path.read_text()
        if marker in text:
            print(f"  {connector_path.name}: H9 already patched, skipping")
        else:
            old = """\
        data = {
            "req_id": req_id,
            "transfer_id": transfer_id,
            "block_notify_list": block_notify_list or [],
            "decode_rank": self.dp_rank,
            "type": "remote_blocks",
        }"""
            new = """\
        data = {
            "req_id": req_id,
            "transfer_id": transfer_id,
            "block_notify_list": block_notify_list or [],
            "decode_rank": self.dp_rank,
            "type": "remote_blocks",
        }
        # H9 diag notify: trace send so we can correlate with prefill recv
        logger.info(
            "[H9 NOTIFY-SEND] transfer_id=%s decode_dp_rank=%s -> %s blocks=%d",
            transfer_id, self.dp_rank, path, len(block_notify_list or []),
        )"""
            if old in text:
                text = text.replace(old, new, 1)
                connector_path.write_text(text)
                print(f"  patched {connector_path.name} (H9 diag: notify-send)")
            else:
                print(f"  {connector_path.name}: H9 send anchor not found")

    # ---- prefill side: log _handle_structured_message ----
    if engine_path.is_file():
        text = engine_path.read_text()
        if marker in text:
            print(f"  {engine_path.name}: H9 already patched, skipping")
            return
        old = """\
    def _handle_structured_message(self, data: dict):
        assert get_role() == ROLE.PRODUCER, "Only prefill can get block messages"
        transfer_id = data["transfer_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)"""
        new = """\
    def _handle_structured_message(self, data: dict):
        assert get_role() == ROLE.PRODUCER, "Only prefill can get block messages"
        transfer_id = data["transfer_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)
        # H9 diag notify: trace receipt of allocation notify from decode
        logger.info(
            "[H9 NOTIFY-RECV] transfer_id=%s from_decode_dp=%s blocks=%d",
            transfer_id, decode_dp_rank, len(block_notify_list),
        )"""
        if old in text:
            text = text.replace(old, new, 1)
            engine_path.write_text(text)
            print(f"  patched {engine_path.name} (H9 diag: notify-recv)")
        else:
            print(f"  {engine_path.name}: H9 recv anchor not found")


# ---------------------------------------------------------------------------
# H10: backport vllm-project/vllm#41213 — clamp negative dequeue_timeout
# (ported from Shiksha Patel's r13-vllm-r13-rpc-timeout-fix.py)
# ---------------------------------------------------------------------------

def patch_multiproc_executor_dequeue_timeout(path: Path) -> None:
    """PR#39276 H10: clamp negative deadline in multiproc_executor.collective_rpc().

    Bug: deadline computed at enqueue time; by the time .result() drains
    pending async futures, ``deadline - time.monotonic()`` can be NEGATIVE.
    Negative float flows into ZMQ poll(timeout) which interprets it as
    "poll forever" -> worker hangs in shm_broadcast wait, eventually
    times out via 1800s gloo recv with mysterious "Worker failed" errors.

    Fix: max(0.0, deadline - now).  Stale deadline yields an immediate
    TimeoutError (the intended semantics) rather than a poll-forever bomb.

    Upstream: https://github.com/vllm-project/vllm/pull/41213
    Source: Shiksha Patel's r13 patch (vllm-r13-rpc-timeout-fix.py).
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H10: clamp expired deadline"
    if marker in text:
        print(f"  {path.name}: H10 already patched, skipping")
        return
    old = """\
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )"""
    new = """\
                # H10: clamp expired deadline to 0 (PR #41213 backport / r13)
                dequeue_timeout = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )"""
    if old not in text:
        print(f"  {path.name}: H10 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H10: dequeue_timeout clamp)")


# ---------------------------------------------------------------------------
# H11: DP coordinator wave-start broadcast fix (THE CRITICAL FIX)
# (ported from Shiksha Patel's r14-vllm-r14-wave-start-fix.py)
# ---------------------------------------------------------------------------

def patch_dp_wave_start(path: Path) -> None:
    """PR#39276 H11: fix DP coordinator first-wave deadlock.

    THE root cause of the multi-node cross-node EP all2all hang.

    Bug: In ``DPEngineCoreProc.add_request()``, the engines_running flag
    and the start_wave broadcast are gated on
        ``if self.has_coordinator and request_wave != self.current_wave:``
    Both default to 0, so on the FIRST request after engine init,
    ``0 != 0`` is False -> early-return fires, engines_running stays False,
    start_wave broadcast never sent.

    Consequence: the rank that received the first request enters its
    forward, calling EP all2all (which requires all 16 DP ranks).  The
    other 15 ranks see (engines_running=False AND local_unfinished_reqs=
    False) at run_busy_loop and hit the ``continue`` path, skipping
    execute_dummy_batch.  They never enter the all2all.  The single busy
    rank's execute_model hangs forever on the EP all2all collective until
    the multiproc_executor 1800s timeout fires.

    On AMD GPUs this manifests as ``HSA_STATUS_ERROR_EXCEPTION 0x1016``
    because the GPU watchdog catches the hang before the software timeout.

    Fix: drop the ``request_wave != self.current_wave`` gate.  When a
    request arrives and ``not engines_running`` AND scheduler is unpaused,
    set engines_running=True and broadcast start_wave to the coordinator
    regardless of wave-number equality.  In steady-state this is a no-op
    (engines_running is already True); it only matters on the first
    request of the first wave (cold start).

    Source: Shiksha Patel's r14 patch (vllm-r14-wave-start-fix.py).
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H11: DP coordinator wave-start broadcast fix"
    if marker in text:
        print(f"  {path.name}: H11 already patched, skipping")
        return
    old = (
        "    def add_request(self, request: Request, request_wave: int = 0):\n"
        "        super().add_request(request, request_wave)\n"
        "        if self.has_coordinator and request_wave != self.current_wave:\n"
        "            if request_wave > self.current_wave:\n"
        "                self.current_wave = request_wave\n"
        "            elif (\n"
        "                not self.engines_running\n"
        "                and self.scheduler.pause_state == PauseState.UNPAUSED\n"
        "            ):\n"
        "                # Request received for an already-completed wave, notify\n"
        "                # front-end that we need to start the next one.\n"
        "                self.engines_running = True\n"
        "                self.output_queue.put_nowait(\n"
        "                    (-1, EngineCoreOutputs(start_wave=self.current_wave))\n"
        "                )"
    )
    new = (
        "    def add_request(self, request: Request, request_wave: int = 0):\n"
        "        super().add_request(request, request_wave)\n"
        "        # H11: DP coordinator wave-start broadcast fix (r14).  Wake other\n"
        "        # DP engines on FIRST request of any wave, not only when\n"
        "        # request_wave != current_wave (both default to 0 -> early-return\n"
        "        # bug caused cross-node EP all2all deadlock on cold start).\n"
        "        if self.has_coordinator:\n"
        "            if request_wave > self.current_wave:\n"
        "                self.current_wave = request_wave\n"
        "            if (\n"
        "                not self.engines_running\n"
        "                and self.scheduler.pause_state == PauseState.UNPAUSED\n"
        "            ):\n"
        "                # Notify coordinator so other DP engines set engines_running=True\n"
        "                # and step in lockstep on collectives (EP all2all, etc.).\n"
        "                self.engines_running = True\n"
        "                self.output_queue.put_nowait(\n"
        "                    (-1, EngineCoreOutputs(start_wave=self.current_wave))\n"
        "                )"
    )
    if old not in text:
        print(f"  {path.name}: H11 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (PR#39276 H11: DP wave-start broadcast)")


def patch_respect_remote_dp_rank(path: Path) -> None:
    """H13: respect proxy-set ``remote_dp_rank`` in kv_transfer_params.

    The v1.1.4_pr366_v2_v3 image's connector reads
    ``kv_transfer_params["remote_dp_rank"]`` then OVERRIDES it with
    ``blake2s(rid) % dp_size`` unless ``remote_dp_rank_override`` is also
    present.  The upstream toy_proxy sets ``remote_dp_rank`` (round-robin
    choice) but DOES NOT set ``remote_dp_rank_override`` -> connector
    overrides with a hash that disagrees with the proxy's choice ->
    decode's notify goes to wrong prefill DP rank -> prefill waiting
    DP rank never sees notify -> deferred write expires at 120s.

    Fix: treat the EXPLICIT PRESENCE of ``remote_dp_rank`` key as the
    "proxy chose, do not override" signal.  Only fall through to blake2s
    when the proxy never set anything at all.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H13: respect proxy-set remote_dp_rank"
    if marker in text:
        print(f"  {path.name}: H13 already applied")
        return
    old = (
        "                if (\n"
        "                    _dp_size > 1\n"
        '                    and "remote_dp_rank_override" not in request.kv_transfer_params\n'
        "                ):"
    )
    new = (
        "                # H13: respect proxy-set remote_dp_rank.  When the\n"
        "                # upstream proxy (toy_proxy / vllm-router) explicitly\n"
        "                # set remote_dp_rank in kv_transfer_params, honour it.\n"
        "                # Only fall through to the blake2s hash when no\n"
        "                # remote_dp_rank was provided at all.\n"
        "                if (\n"
        "                    _dp_size > 1\n"
        '                    and "remote_dp_rank_override" not in request.kv_transfer_params\n'
        '                    and "remote_dp_rank" not in request.kv_transfer_params\n'
        "                ):"
    )
    if old not in text:
        print(f"  {path.name}: H13 anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (H13: respect proxy-set remote_dp_rank)")


def patch_get_peer_zmq_tolerant(path: Path) -> None:
    """H12a: make get_peer_zmq_from_request_id return None instead of raising
    when the request_id lacks the `___prefill_addr_X___decode_addr_Y` envelope.

    The vllm bench tool (and various internal vllm code paths) create requests
    with bare UUID-style ids.  When those reach the MoRIIOConnector scheduler,
    the connector crashes the whole EngineCore.  Returning None lets the
    caller (add_new_req) decide what to do.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H12a: tolerate unparseable request_id"
    if marker in text:
        print(f"  {path.name}: H12a already applied")
        return
    old = (
        "    if m is None:\n"
        "        raise ValueError(\n"
        '            f"Cannot parse peer zmq_address from request_id: {request_id!r}"\n'
        "        )\n"
        "    return m.group(1)"
    )
    new = (
        "    if m is None:\n"
        "        # H12a: tolerate unparseable request_id (e.g. vllm bench tool\n"
        "        # or internal heartbeat requests without proxy envelope).\n"
        "        # Returning None lets add_new_req skip the request instead of\n"
        "        # crashing the whole EngineCore.\n"
        "        return None\n"
        "    return m.group(1)"
    )
    if old not in text:
        print(f"  {path.name}: H12a anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (H12a: get_peer_zmq tolerant)")


def patch_add_new_req_skip_unparseable(path: Path) -> None:
    """H12b: in add_new_req, if get_peer_zmq returns None or
    parse_moriio_zmq_address raises, log a warning and SKIP this request
    (don't add it to the meta).  Crashing the EngineCore for one bad
    request poisons all in-flight requests.
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# H12b: skip requests with unparseable peer zmq"
    if marker in text:
        print(f"  {path.name}: H12b already applied")
        return
    old = (
        "        # Parse host/ports from the request_id. The router embeds both zmq_addresses\n"
        "        # in the request_id\n"
        "        peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=write_mode)\n"
        "        remote_host, remote_handshake_port, remote_notify_port = (\n"
        "            parse_moriio_zmq_address(peer_zmq)\n"
        "        )"
    )
    new = (
        "        # Parse host/ports from the request_id. The router embeds both zmq_addresses\n"
        "        # in the request_id\n"
        "        # H12b: skip requests with unparseable peer zmq.  Some requests\n"
        "        # (vllm bench, internal heartbeats) reach the connector without\n"
        "        # the proxy-injected envelope; raising here would tear down the\n"
        "        # entire EngineCore and poison healthy concurrent requests.\n"
        "        peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=write_mode)\n"
        "        if peer_zmq is None:\n"
        "            import logging as _h12_logging\n"
        "            _h12_logging.getLogger(__name__).warning(\n"
        "                'MoRIIO add_new_req: skipping request %r - no proxy envelope in request_id (likely bench tool or internal request)',\n"
        "                request_id,\n"
        "            )\n"
        "            return\n"
        "        try:\n"
        "            remote_host, remote_handshake_port, remote_notify_port = (\n"
        "                parse_moriio_zmq_address(peer_zmq)\n"
        "            )\n"
        "        except ValueError as _h12_e:\n"
        "            import logging as _h12_logging\n"
        "            _h12_logging.getLogger(__name__).warning(\n"
        "                'MoRIIO add_new_req: skipping request %r - malformed peer_zmq %r: %s',\n"
        "                request_id, peer_zmq, _h12_e,\n"
        "            )\n"
        "            return"
    )
    if old not in text:
        print(f"  {path.name}: H12b anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (H12b: add_new_req skip unparseable)")


def patch_writemode_notify_is_producer(path: Path) -> None:
    """PR#39276 notify-direction fix, ported to the 71725f673 connector.

    The 71725f673 WRITE-mode block resolves the notify peer with
    is_producer=True, which returns the DECODE address.  On the decode side
    this makes it send the "blocks ready" notify to ITS OWN host:notify_port,
    so the receiver trips `assert get_role()==PRODUCER` ("Only prefill can get
    block messages") and the KV transfer hangs.  The consumer (decode) must
    target the PRODUCER (prefill) address -> is_producer=False.  (#39276 does
    exactly this; its hunk did not anchor on 71725f673 because the surrounding
    block was reworded, so this is the minimal equivalent.)
    """
    if not path.is_file():
        return
    text = path.read_text()
    marker = "# 39276 NOTIFY-DIR FIX"
    if marker in text:
        print(f"  {path.name}: notify is_producer fix already applied")
        return
    old = (
        "                peer_zmq = get_peer_zmq_from_request_id(\n"
        "                    request.request_id, is_producer=True\n"
        "                )"
    )
    new = (
        "                # 39276 NOTIFY-DIR FIX: decode (consumer) must notify the\n"
        "                # PREFILL (producer) address -> is_producer=False.  With\n"
        "                # is_producer=True the decode side sent the block-ready\n"
        "                # notify to its OWN address ('Only prefill can get block\n"
        "                # messages' -> hang).\n"
        "                peer_zmq = get_peer_zmq_from_request_id(\n"
        "                    request.request_id, is_producer=False\n"
        "                )"
    )
    if old not in text:
        print(f"  {path.name}: notify is_producer anchor not found, skipping")
        return
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name} (39276 notify-dir: is_producer=False)")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1:
        raise SystemExit(f"usage: {sys.argv[0]} <vllm_install_dir> [--core-only]")
    root = Path(args[0])

    # --core-only: apply ONLY the core/engine multi-DP + robustness fixes that
    # are NOT in the (native) connector. Use this on a vLLM whose MoRIIO connector
    # already has the #39276 notify/remote_dp_rank logic natively (e.g. the
    # June-12 nightly): layering the old connector hunks (H2/H5/H9/H12) on top of
    # the evolved native connector breaks multi-DP notify routing (writes expire).
    # Core/engine fixes still needed: engine_id-per-DP, H11 wave-start (0x1016),
    # H1 skip-profile, H3 finished_sending guard, H10 dequeue clamp.
    if "--core-only" in flags:
        print("[patcher] --core-only: applying core/engine fixes only; "
              "leaving MoRIIO connector NATIVE")
        patch_core_py(root / "v1/engine/core.py")
        patch_skip_profile_run(root / "v1/worker/gpu_worker.py")
        patch_scheduler_kv_xfer_guard(root / "v1/core/sched/scheduler.py")
        patch_multiproc_executor_dequeue_timeout(
            root / "v1/executor/multiproc_executor.py"
        )
        patch_dp_wave_start(root / "v1/engine/core.py")
        return

    patch_core_py(root / "v1/engine/core.py")
    patch_moriio_common_py(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
    )
    patch_moriio_connector_py(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )
    patch_connector_remote_dp_rank(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )
    # 39276 notify-direction fix (decode notifies prefill, not itself).
    patch_writemode_notify_is_producer(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )
    patch_moriio_engine_py(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
    )
    _strip_runner_boot_extras(root / "v1/worker/gpu_model_runner.py")
    patch_skip_profile_run(root / "v1/worker/gpu_worker.py")
    patch_ep_warmup_barrier(root / "v1/worker/gpu_worker.py")

    # H3-H6: hang-proof patches
    patch_scheduler_kv_xfer_guard(root / "v1/core/sched/scheduler.py")
    patch_notify_listener_resilience(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
    )
    patch_send_notify_block_robustness(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )
    patch_scheduler_local_dp_rank(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )
    patch_deferred_expiry_dedup(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
    )
    patch_diag_notify_logging(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py",
    )

    # H10-H11: Shiksha Patel's r13+r14 ported from MAD-private-Shiksha
    #   H10 (r13): clamp negative dequeue_timeout (vllm PR #41213 backport)
    #   H11 (r14): DP coordinator wave-start broadcast fix - THE root cause fix
    # for cross-node EP all2all deadlock manifesting as HSA_STATUS_ERROR_EXCEPTION
    # 0x1016 at 2P/2D and beyond.  See LANEPE root-cause analysis at
    # /shared_inference/shikpate/DSV3_LANEPE_ROOT_CAUSE_ANALYSIS.md
    patch_multiproc_executor_dequeue_timeout(
        root / "v1/executor/multiproc_executor.py"
    )
    patch_dp_wave_start(root / "v1/engine/core.py")

    # 2P/2D LANEPE FIX: targets the v1.1.4_pr366_v2_v3 image's baked-but-broken
    # `_is_kv_master` global-rank check.  Must run AFTER patch_moriio_connector_py
    # so we don't fight earlier hunks.
    patch_is_kv_master_local_rank(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )

    # H12: tolerate unparseable request_ids in get_peer_zmq_from_request_id so
    # benchmark tools (`vllm bench serve`) or vLLM internal heartbeat requests
    # that lack the proxy-injected `___prefill_addr_X___decode_addr_Y_uid`
    # format don't crash the entire EngineCore.  Returns None and lets the
    # caller skip the request instead of raising.
    patch_get_peer_zmq_tolerant(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
    )
    patch_add_new_req_skip_unparseable(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
    )

    # H13: respect proxy-set remote_dp_rank.  The image's connector
    # OVERRIDES the proxy's chosen remote_dp_rank with blake2s(rid) % dp_size,
    # which disagrees with the proxy's round-robin choice -> notify lands on
    # wrong prefill DP -> defer expires.  Honour explicit remote_dp_rank key
    # presence as the signal "proxy already chose, don't second-guess".
    patch_respect_remote_dp_rank(
        root / "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
    )


if __name__ == "__main__":
    main()
