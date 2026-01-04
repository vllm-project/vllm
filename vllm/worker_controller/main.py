from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
import logging
import os

# Import WorkerController and config classes
from vllm.worker_controller.worker_controller import WorkerController
from vllm.config import (VllmConfig, ModelConfig, CacheConfig, ParallelConfig,
                         ObservabilityConfig, CompilationConfig, DeviceConfig)

logger = logging.getLogger(__name__)


def _build_vllm_config_from_dict(config_dict: Dict[str, Any]) -> VllmConfig:
    """
    Build VllmConfig from a nested dictionary structure.

    Expects format like:
    {
        "model_config": {"model": "...", "dtype": "...", ...},
        "cache_config": {"block_size": 16, ...},
        "parallel_config": {"world_size": 1, ...},
        ...
    }
    """
    # Extract or create each config component
    model_config_dict = config_dict.get("model_config", {})
    cache_config_dict = config_dict.get("cache_config", {})
    parallel_config_dict = config_dict.get("parallel_config", {})
    scheduler_config_dict = config_dict.get("scheduler_config", {})

    # Ensure worker_cls is set for parallel_config
    if "worker_cls" not in parallel_config_dict:
        parallel_config_dict["worker_cls"] = "vllm.worker_controller.worker.gpu_worker.Worker"

    # Build ModelConfig
    model_config = ModelConfig(**model_config_dict)

    # Build CacheConfig
    cache_config = CacheConfig(**cache_config_dict)

    # Build ParallelConfig
    parallel_config = ParallelConfig(**parallel_config_dict)

    # Build optional configs with defaults
    observability_config = ObservabilityConfig()
    compilation_config = CompilationConfig()
    device_config = DeviceConfig()

    # Build VllmConfig
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        observability_config=observability_config,
        compilation_config=compilation_config,
        device_config=device_config,
    )

    return vllm_config


class EngineCreateRequest(BaseModel):
    """Request to create a new engine with RemoteExecutor.

    Supports two formats:
    1. Simple format: Specify model, dtype, etc. directly
    2. Advanced format: Pass full vllm_config dict
    """
    engine_uuid: str

    # Simple format fields (optional if vllm_config provided)
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    dtype: str = "float16"
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4
    block_size: int = 16
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    seed: int = 0

    # Advanced format: full vllm_config dict (optional)
    vllm_config: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_config(self):
        """Ensure either model or vllm_config is provided."""
        if self.vllm_config is None and self.model is None:
            raise ValueError(
                "Either 'model' or 'vllm_config' must be provided")
        return self


class EngineDeleteRequest(BaseModel):
    """Request to delete an existing engine."""
    engine_uuid: str


class EngineStatusResponse(BaseModel):
    """Response with engine status information."""
    engine_uuid: str
    status: str
    api_url: Optional[str] = None
    port: Optional[int] = None
    assigned_ranks: Optional[list[int]] = None
    model: Optional[str] = None
    pid: Optional[int] = None


app = FastAPI(
    title="Worker Controller API",
    description="API for managing vLLM engines with RemoteExecutor and shared worker pool",
    version="1.0.0"
)

# Global WorkerController instance
worker_controller: Optional[WorkerController] = None


@app.on_event("startup")
async def startup_event():
    """Initialize WorkerController on startup."""
    global worker_controller
    logger.info("Initializing WorkerController...")
    worker_controller = WorkerController()
    logger.info(
        f"WorkerController initialized with {len(worker_controller.executor.workers)} workers")


@app.get("/")
def read_root():
    """Root endpoint - health check."""
    return {
        "message": "Worker Controller API",
        "status": "running",
        "num_workers": len(worker_controller.executor.workers) if worker_controller else 0
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if worker_controller is None:
        raise HTTPException(
            status_code=503, detail="WorkerController not initialized")
    return {
        "status": "healthy",
        "num_workers": len(worker_controller.executor.workers),
        "num_engines": len(worker_controller.executor.engines)
    }


@app.post("/engines", response_model=EngineStatusResponse)
def create_engine(request: EngineCreateRequest):
    """
    Create a new engine by assigning workers and spawning an API server with RemoteExecutor.

    The API server will be available at http://localhost:{port} where port is automatically assigned.

    Supports two request formats:
    1. Simple: {"engine_uuid": "test", "model": "facebook/opt-125m", ...}
    2. Advanced: {"engine_uuid": "test", "vllm_config": {...full config dict...}}
    """
    if worker_controller is None:
        raise HTTPException(
            status_code=503, detail="WorkerController not initialized")

    try:
        # Check if using advanced format with vllm_config dict
        if request.vllm_config is not None:
            logger.info(
                f"Creating engine {request.engine_uuid} with advanced vllm_config")

            # Build VllmConfig from nested dict structure
            vllm_config = _build_vllm_config_from_dict(request.vllm_config)
            model_name = vllm_config.model_config.model

        else:
            # Simple format - build from flat parameters
            if request.model is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'model' or 'vllm_config' must be provided"
                )

            logger.info(
                f"Creating engine {request.engine_uuid} with model {request.model}")

            model_config = ModelConfig(
                model=request.model,
                tokenizer=request.tokenizer or request.model,
                trust_remote_code=request.trust_remote_code,
                dtype=request.dtype,
                seed=request.seed,
                enforce_eager=True,
            )

            cache_config = CacheConfig(
                block_size=request.block_size,
                gpu_memory_utilization=request.gpu_memory_utilization,
                swap_space=request.swap_space,
            )

            # Calculate world_size from parallel config
            world_size = request.tensor_parallel_size * request.pipeline_parallel_size

            parallel_config = ParallelConfig(
                tensor_parallel_size=request.tensor_parallel_size,
                pipeline_parallel_size=request.pipeline_parallel_size,
                world_size=world_size,
                worker_cls='vllm.worker_controller.worker.gpu_worker.Worker'
            )

            observability_config = ObservabilityConfig()
            compilation_config = CompilationConfig()
            device_config = DeviceConfig()

            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                observability_config=observability_config,
                compilation_config=compilation_config,
                device_config=device_config,
            )
            model_name = request.model

        # Create engine: this assigns workers, loads model, calculates blocks,
        # and spawns API server with RemoteExecutor
        proc = worker_controller.create(vllm_config, request.engine_uuid)

        # Get assigned resources
        assigned_ranks = worker_controller.resource_allocator.get_ranks_by_uuid(
            request.engine_uuid)
        port = worker_controller.resource_allocator.get_port_by_uuid(
            request.engine_uuid)

        logger.info(
            f"Engine {request.engine_uuid} created successfully on port {port}")

        return EngineStatusResponse(
            engine_uuid=request.engine_uuid,
            status="created",
            api_url=f"http://localhost:{port}",
            port=port,
            assigned_ranks=assigned_ranks,
            model=request.model,
            pid=proc.pid if proc else None
        )

    except Exception as e:
        logger.error(f"Failed to create engine {request.engine_uuid}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/engines/{engine_uuid}")
def delete_engine(engine_uuid: str):
    """
    Delete an engine and release its resources.

    This will:
    1. Terminate the API server process
    2. Unload the model from workers
    3. Release assigned ranks back to the pool
    """
    if worker_controller is None:
        raise HTTPException(
            status_code=503, detail="WorkerController not initialized")

    logger.info(f"Deleting engine {engine_uuid}")

    try:
        # Check if engine exists
        if engine_uuid not in worker_controller.executor.engines:
            raise HTTPException(
                status_code=404, detail=f"Engine {engine_uuid} not found")

        # Get info before deletion
        assigned_ranks = worker_controller.resource_allocator.get_ranks_by_uuid(
            engine_uuid)
        port = worker_controller.resource_allocator.get_port_by_uuid(
            engine_uuid)

        # Delete the engine
        worker_controller.delete(engine_uuid)

        logger.info(
            f"Engine {engine_uuid} deleted, released ranks {assigned_ranks} and port {port}")

        return {
            "message": f"Engine {engine_uuid} deleted successfully",
            "released_ranks": assigned_ranks,
            "released_port": port
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete engine {engine_uuid}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines/{engine_uuid}", response_model=EngineStatusResponse)
def get_engine_status(engine_uuid: str):
    """Get status information for a specific engine."""
    if worker_controller is None:
        raise HTTPException(
            status_code=503, detail="WorkerController not initialized")

    if engine_uuid not in worker_controller.executor.engines:
        raise HTTPException(
            status_code=404, detail=f"Engine {engine_uuid} not found")

    engine_info = worker_controller.executor.engines[engine_uuid]
    assigned_ranks = worker_controller.resource_allocator.get_ranks_by_uuid(
        engine_uuid)
    port = worker_controller.resource_allocator.get_port_by_uuid(engine_uuid)

    proc = engine_info.get("proc")

    # Check if the process is alive with a timeout
    is_alive = False
    if proc:
        # The join() method with a timeout is a non-blocking way to check liveness
        proc.join(timeout=0.1)
        is_alive = proc.is_alive()

    return EngineStatusResponse(
        engine_uuid=engine_uuid,
        status="running" if is_alive else "stopped",
        api_url=f"http://localhost:{port}",
        port=port,
        assigned_ranks=assigned_ranks,
        model=engine_info.get(
            "vllm_config").model_config.model if "vllm_config" in engine_info else None,
        pid=proc.pid if proc else None
    )


@app.get("/engines")
def list_engines():
    """List all engines and their status."""
    if worker_controller is None:
        raise HTTPException(
            status_code=503, detail="WorkerController not initialized")

    engines = []
    for engine_uuid in worker_controller.executor.engines:
        try:
            status = get_engine_status(engine_uuid)
            engines.append(status.dict())
        except Exception as e:
            logger.error(f"Error getting status for {engine_uuid}: {e}")

    return {
        "num_engines": len(engines),
        "engines": engines
    }


@app.get("/workers")
def list_workers():
    """List all workers and their current assignments."""
    if worker_controller is None:
        raise HTTPException(
            status_code=503, detail="WorkerController not initialized")

    workers = []
    for rank, uuid in worker_controller.resource_allocator.resources.items():
        workers.append({
            "rank": rank,
            "status": "assigned" if uuid != 0 else "free",
            "assigned_to": uuid if uuid != 0 else None
        })

    return {
        "num_workers": len(workers),
        "workers": workers
    }


if __name__ == "__main__":
    # Use v0 as requested
    os.environ['VLLM_USE_V1'] = '0'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger.info("Starting Worker Controller API on port 8000")
    logger.info("API docs will be available at http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
