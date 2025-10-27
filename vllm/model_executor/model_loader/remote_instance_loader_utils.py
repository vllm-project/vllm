# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from  https://github.com/sgl-project/sglang/pull/8215

# Changes:
# - Add support for Pipeline parallel + Tensor parallel hybrid parallelism
# - Add basic model and tensor count validation

import logging
import threading
import time
from typing import Any

import requests
import torch
import torch.distributed
import torch.nn as nn

from vllm.distributed.kv_transfer.kv_connector.v1.weight_transfer_connector import (
    init_custom_process_group,
)

logger = logging.getLogger(__name__)

# Dictionary to store process groups
# Format: {group_name: process_group}
_weights_send_group: dict[str, Any] = {}


def cleanup_thread(group_name: str, delay: float = 60.0):
    # Create a thread to clean up the process group
    # if it's not used within delay seconds
    cleanup_thread = threading.Thread(
        target=_cleanup_stale_group, args=(group_name, delay)
    )
    cleanup_thread.daemon = (
        True  # Set as daemon thread so it doesn't prevent program exit
    )
    cleanup_thread.start()


def _cleanup_stale_group(group_name: str, delay: float = 60.0):
    """Clean up a stale process group that has timed out."""
    try:
        # Sleep for the specified delay
        time.sleep(delay)

        # Check if the group still exists and clean it up
        process_group = _weights_send_group.get(group_name)
        if process_group is not None:
            torch.distributed.distributed_c10d.destroy_process_group(process_group)
            del _weights_send_group[group_name]
            logger.info("Cleaned up stale process group: %s", group_name)

    except Exception as e:
        logger.warning("Failed to clean up stale process group %s: %s", group_name, e)


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_seed_instance_ip: str,
    remote_seed_instance_service_port: int,
    send_weights_group_ports: list[int],
    local_client_id: str,
):
    seed_instance_service_url = (
        f"http://{remote_seed_instance_ip}:{remote_seed_instance_service_port}"
    )

    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_seed_instance_ip,
                "ports": (",".join(str(p) for p in send_weights_group_ports)),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{local_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            "Failed to trigger init_weights_send_group_for_remote_instance_request to "
            "seed instance %s: %s.",
            seed_instance_service_url,
            e,
        )
        raise


def trigger_transferring_weights_request(
    remote_seed_instance_ip: str,
    remote_seed_instance_service_port: int,
    send_weights_group_ports: list[int],
    local_client_id: str,
    tensors_nums: int,
):
    seed_instance_service_url = (
        f"http://{remote_seed_instance_ip}:{remote_seed_instance_service_port}"
    )
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_seed_instance_ip,
                "ports": (",".join(str(p) for p in send_weights_group_ports)),
                "group_name": f"send_weights_{local_client_id}",
                "state_dict": f"{tensors_nums}",
            },
        )
    except Exception as e:
        logger.error("Failed to trigger send weights to remote instance request: %s", e)
        raise


def get_remote_instance_model(
    remote_seed_instance_ip: str,
    remote_seed_instance_service_port: int,
) -> str:
    # Get model information from the ready instance
    seed_instance_service_url = (
        f"http://{remote_seed_instance_ip}:{remote_seed_instance_service_port}"
    )
    response = requests.get(f"{seed_instance_service_url}/v1/models")
    models_info = response.json()

    # Verify if the model matches
    ready_model_id = models_info["data"][0]["id"]
    return ready_model_id


def init_weights_send_group_for_remote_instance(
    master_address: str,
    ports: str,
    group_rank: int,
    world_size: int,
    group_name: str,
    backend: str = "nccl",
):
    import time

    begin = time.perf_counter()

    assert torch.distributed.is_initialized(), (
        "Default torch process group must be initialized"
    )
    assert group_name != "", "Group name cannot be empty"
    from vllm.distributed.parallel_state import get_tp_group

    tp_rank = get_tp_group().rank_in_group
    tp_size = get_tp_group().world_size

    from vllm.distributed.parallel_state import get_pp_group

    pp_rank = get_pp_group().rank_in_group
    pp_size = get_pp_group().world_size
    global_rank = pp_rank * tp_size + tp_rank

    ports_list = ports.split(",")
    gpu_id = torch.cuda.current_device()

    assert len(ports_list) == tp_size * pp_size, (
        f"Expected {tp_size * pp_size} ports, but got {len(ports_list)} ports."
    )
    group_port = ports_list[global_rank]
    group_name = f"{group_name}_{global_rank}"

    logger.info(
        "init custom process group: pp_rank=%s, tp_rank=%s, "
        "gpu_id=%s, master_address=%s, master_port=%s, "
        "group_rank=%s, world_size=%s, group_name=%s, "
        "backend=%s",
        pp_rank,
        tp_rank,
        gpu_id,
        master_address,
        group_port,
        group_rank,
        world_size,
        group_name,
        backend,
    )

    try:
        from datetime import timedelta

        _weights_send_group[group_name] = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{group_port}",
            timeout=timedelta(seconds=60),
            world_size=world_size,
            rank=group_rank,
            group_name=group_name,
            device_id=torch.device("cuda", gpu_id),
        )
        cleanup_thread(group_name, 60)

        message = (
            f"Succeeded to init group through {master_address}:{group_port} group."
        )
        end = time.perf_counter()
        logger.info("init_weights_send_group_for_remote_instance using %s", end - begin)
        return {"success": True, "message": message}
    except Exception as e:
        message = f"Failed to init group: {e}."
        logger.error(message)
        return {"success": False, "message": message}


def send_weights_to_remote_instance(
    master_address: str,
    ports: str,
    group_name: str,
    remote_tensor_nums: int,
    model: nn.Module,
):
    assert torch.distributed.is_initialized(), (
        "Default torch process group must be initialized"
    )
    assert group_name != "", "Group name cannot be empty"

    from vllm.distributed.parallel_state import get_tp_group

    tp_rank = get_tp_group().rank_in_group
    tp_size = get_tp_group().world_size

    from vllm.distributed.parallel_state import get_pp_group

    pp_rank = get_pp_group().rank_in_group
    pp_size = get_pp_group().world_size
    global_rank = pp_rank * tp_size + tp_rank

    ports_list = ports.split(",")
    assert len(ports_list) == tp_size * pp_size, (
        f"Expected {tp_size * pp_size} ports, but got {len(ports_list)} ports."
    )
    group_port = ports_list[global_rank]
    group_name = f"{group_name}_{global_rank}"

    send_group = None
    success = False
    message = ""

    try:
        # Count non-empty tensors in the model's state_dict
        non_empty_count = sum(1 for v in model.state_dict().values() if v.numel() > 0)

        # Safety check: Only worker0 validates tensor count
        validation_passed = True
        if global_rank == 0 and remote_tensor_nums != non_empty_count:
            validation_passed = False
            logger.error(
                "[Worker0] Tensor count mismatch between local and remote instances. "
                "Local non-empty tensor count: %s, "
                "Remote tensor count: %s. "
                "Aborting weight broadcast for all workers.",
                non_empty_count,
                remote_tensor_nums,
            )

        # Broadcast validation result from worker0 to all workers
        from vllm.distributed.parallel_state import get_world_group

        world_group = get_world_group()
        validation_result = [validation_passed]
        world_group.broadcast_object_list(validation_result, src=0)
        validation_passed = validation_result[0]

        # If validation failed, all workers abort
        if not validation_passed:
            message = (
                f"[Worker{global_rank}] Aborting weight broadcast due to worker0 "
                f"validation failure."
            )
            logger.error(message)
            return {"success": False, "message": message}

        # Get the process group
        send_group = _weights_send_group.get(group_name)
        if send_group is None:
            message = (
                f"Group {group_name} not in _weights_send_group list. "
                f"Please call `init_weights_send_group_for_remote_instance` first."
            )
            logger.error(message)
            return {"success": False, "message": message}

        logger.info("Send weight in %s", send_group)
        torch.cuda.empty_cache()
        state_dict = model.state_dict()
        for key, tensor in state_dict.items():
            if tensor.numel():
                torch.distributed.broadcast(
                    tensor,
                    group_src=0,
                    group=send_group,
                )
        torch.cuda.empty_cache()
        success = True
        message = (
            f"Succeeded to send weights through {master_address}:{group_port} "
            f"{group_name}."
        )
    except Exception as e:
        message = f"Failed to send weights: {e}."
        logger.error(message)
        logger.error("Model state_dict keys: %s", list(model.state_dict().keys()))
        logger.error("Number of state_dict items: %s", len(model.state_dict()))
    finally:
        # destroy the process group after sending weights
        try:
            if group_name in _weights_send_group:
                del _weights_send_group[group_name]
            if send_group is not None:
                torch.distributed.distributed_c10d.destroy_process_group(send_group)
        except Exception as cleanup_error:
            logger.warning("Failed to clean up process group: %s", cleanup_error)

    return {"success": success, "message": message}
