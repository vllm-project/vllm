# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, Optional

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter()


class WorkerInfoResponse(BaseModel):
    instance_id: str
    worker_id: int
    ip: str
    port: int
    peer_init_url: Optional[str]
    registration_time: float
    last_heartbeat_time: float
    key_count: int


class WorkerListResponse(BaseModel):
    workers: list[WorkerInfoResponse]
    total_count: int


@router.get("/controller/workers")
async def get_workers(
    request: Request,
    instance_id: Annotated[Optional[str], Query()] = None,
    worker_id: Annotated[Optional[int], Query()] = None,
):
    """
    Get worker information with flexible query parameters.

    - No parameters: List all registered workers across all instances
    - instance_id only: List all workers for a specific instance
    - instance_id and worker_id: Get detailed info about a specific worker

    Args:
        instance_id: Optional instance ID to filter workers
        worker_id: Optional worker ID to get specific worker details
    """
    try:
        controller_manager = getattr(
            request.app.state, "lmcache_controller_manager", None
        )

        if controller_manager is None:
            raise HTTPException(
                status_code=503, detail="Controller manager not available"
            )

        reg_controller = controller_manager.reg_controller

        # Case 1: Get specific worker by instance_id and worker_id
        if instance_id is not None and worker_id is not None:
            worker_node = reg_controller.registry.get_worker(instance_id, worker_id)
            if worker_node is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Worker ({instance_id}, {worker_id}) not found",
                )

            worker_info = worker_node.to_worker_info(instance_id)
            key_count = worker_node.get_kv_count()
            return WorkerInfoResponse(
                instance_id=worker_info.instance_id,
                worker_id=worker_info.worker_id,
                ip=worker_info.ip,
                port=worker_info.port,
                peer_init_url=worker_info.peer_init_url,
                registration_time=worker_info.registration_time,
                last_heartbeat_time=worker_info.last_heartbeat_time,
                key_count=key_count,
            )

        # Case 2: Get all workers for a specific instance
        elif instance_id is not None:
            instance_node = reg_controller.registry.get_instance(instance_id)
            if instance_node is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No workers found for instance {instance_id}",
                )

            worker_infos = instance_node.get_all_worker_infos()
            workers = []
            for worker_info in worker_infos:
                worker_node = reg_controller.registry.get_worker(
                    instance_id, worker_info.worker_id
                )
                key_count = worker_node.get_kv_count() if worker_node else 0
                workers.append(
                    WorkerInfoResponse(
                        instance_id=worker_info.instance_id,
                        worker_id=worker_info.worker_id,
                        ip=worker_info.ip,
                        port=worker_info.port,
                        peer_init_url=worker_info.peer_init_url,
                        registration_time=worker_info.registration_time,
                        last_heartbeat_time=worker_info.last_heartbeat_time,
                        key_count=key_count,
                    )
                )

            return WorkerListResponse(workers=workers, total_count=len(workers))

        # Case 3: Get all workers across all instances
        else:
            worker_infos = reg_controller.registry.get_all_worker_infos_cached()
            workers = []
            for worker_info in worker_infos:
                worker_node = reg_controller.registry.get_worker(
                    worker_info.instance_id, worker_info.worker_id
                )
                key_count = worker_node.get_kv_count() if worker_node else 0
                workers.append(
                    WorkerInfoResponse(
                        instance_id=worker_info.instance_id,
                        worker_id=worker_info.worker_id,
                        ip=worker_info.ip,
                        port=worker_info.port,
                        peer_init_url=worker_info.peer_init_url,
                        registration_time=worker_info.registration_time,
                        last_heartbeat_time=worker_info.last_heartbeat_time,
                        key_count=key_count,
                    )
                )

            return WorkerListResponse(workers=workers, total_count=len(workers))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from None
