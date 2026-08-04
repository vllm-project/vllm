# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple
import argparse
import asyncio
import json
import os
import sys
import uuid

# Add project root to Python path for local development
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

# Third Party
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.config import (
    load_controller_config_with_overrides,
)
from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import (  # noqa: E501
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    ErrorMsg,
    HealthMsg,
    HealthRetMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PinMsg,
    PinRetMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    WorkerInfo,
)
from lmcache.v1.config_base import parse_command_line_extra_params
from lmcache.v1.internal_api_server.api_registry import APIRegistry

logger = init_logger(__name__)


def parse_extra_params(extra_args: list) -> Dict[str, Any]:
    """Parse extra parameters in key=value format"""
    params = {}
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            try:
                if value.lower() in ("true", "false"):
                    params[key] = value.lower() == "true"
                elif value.isdigit():
                    params[key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    params[key] = float(value)
                else:
                    params[key] = value
            except ValueError:
                params[key] = value
            logger.info(f"Extra parameter: {key} = {params[key]}")
    return params


def create_app(
    controller_urls: dict[str, str],
    health_check_interval: int,
    lmcache_worker_timeout: int,
) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    lmcache_controller_manager = LMCacheControllerManager(
        controller_urls, health_check_interval, lmcache_worker_timeout
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start background task here
        lmcache_cluster_monitor_task = asyncio.create_task(
            lmcache_controller_manager.start_all()
        )
        yield
        # Optionally cancel the task on shutdown
        lmcache_cluster_monitor_task.cancel()
        try:
            await lmcache_cluster_monitor_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(lifespan=lifespan)
    app.state.lmcache_controller_manager = lmcache_controller_manager

    # Register internal APIs (only common APIs, not vllm-specific ones)
    registry = APIRegistry(app)
    registry.register_all_apis(categories=["common", "controller"])

    # Add static files for frontend
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    static_dir = os.path.join(
        project_root,
        "lmcache",
        "v1",
        "cache_controller",
        "frontend",
        "static",
    )
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("Controller frontend static files mounted at /static")
    else:
        logger.warning("Controller frontend static directory not found: %s", static_dir)

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the Controller frontend HTML page."""
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(
                content="<h1>Controller Frontend not found</h1>"
                "<p>Please build the frontend first.</p>",
                status_code=404,
            )

    class QueryInstRequest(BaseModel):
        event_id: str
        ip: str

    class QueryInstResponse(BaseModel):
        event_id: str
        res: str  # the instance id

    @app.post("/query_instance")
    async def query_instance(req: QueryInstRequest):
        try:
            event_id = "QueryInst" + str(uuid.uuid4())
            msg = QueryInstMsg(
                event_id=event_id,
                ip=req.ip,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, QueryInstRetMsg)
            return QueryInstResponse(
                event_id=ret_msg.event_id,
                res=ret_msg.instance_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class LookupRequest(BaseModel):
        tokens: List[int]

    class LookupResponse(BaseModel):
        event_id: str
        # a list of (instance_id, location, token_count)
        layout_info: Dict[str, Tuple[str, int]]

    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):
        try:
            event_id = "Lookup" + str(uuid.uuid4())
            msg = LookupMsg(
                event_id=event_id,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, LookupRetMsg)
            return LookupResponse(
                event_id=ret_msg.event_id, layout_info=ret_msg.layout_info
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearRequest(BaseModel):
        instance_id: str
        location: str

    class ClearResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):
        try:
            event_id = "Clear" + str(uuid.uuid4())
            msg = ClearMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, ClearRetMsg)
            return ClearResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PinRequest(BaseModel):
        instance_id: str
        location: str
        tokens: list[int]

    class PinResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):
        try:
            event_id = "Pin" + str(uuid.uuid4())
            msg = PinMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, PinRetMsg)
            return PinResponse(event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class CompressResponse(BaseModel):
        event_id: str
        num_tokens: int

    class DecompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class DecompressResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):
        try:
            event_id = "Compress" + str(uuid.uuid4())
            msg = CompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CompressRetMsg)
            return CompressResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/decompress", response_model=DecompressResponse)
    async def decompress(req: DecompressRequest):
        try:
            event_id = "Decompress" + str(uuid.uuid4())
            msg = DecompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, DecompressRetMsg)
            return DecompressResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class MoveRequest(BaseModel):
        # (instance_id, location)
        old_position: Tuple[str, str]
        new_position: Tuple[str, str]
        tokens: Optional[List[int]] = []
        should_copy: Optional[bool] = False

    class MoveResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):
        try:
            event_id = "Move" + str(uuid.uuid4())
            msg = MoveMsg(
                event_id=event_id,
                old_position=req.old_position,
                new_position=req.new_position,
                tokens=req.tokens,
                copy=req.should_copy,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, MoveRetMsg)
            return MoveResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class HealthRequest(BaseModel):
        instance_id: str

    class HealthResponse(BaseModel):
        event_id: str
        # worker_id -> error_code
        error_codes: dict[int, int]

    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):
        try:
            event_id = "health" + str(uuid.uuid4())
            msg = HealthMsg(
                event_id=event_id,
                instance_id=req.instance_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, HealthRetMsg)
            return HealthResponse(
                event_id=ret_msg.event_id, error_codes=ret_msg.error_codes
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CheckFinishRequest(BaseModel):
        event_id: str

    class CheckFinishResponse(BaseModel):
        status: str

    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):
        try:
            msg = CheckFinishMsg(
                event_id=req.event_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CheckFinishRetMsg)
            return CheckFinishResponse(status=ret_msg.status)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class QueryWorkerInfoRequest(BaseModel):
        instance_id: str
        worker_ids: Optional[list[int]] = None

    class QueryWorkerInfoResponse(BaseModel):
        event_id: str
        worker_infos: list[WorkerInfo]

    @app.post("/query_worker_info", response_model=QueryWorkerInfoResponse)
    async def query_worker_info(req: QueryWorkerInfoRequest):
        try:
            event_id = "QueryWorkerInfo" + str(uuid.uuid4())
            msg = QueryWorkerInfoMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                worker_ids=req.worker_ids,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, QueryWorkerInfoRetMsg)
            return QueryWorkerInfoResponse(
                event_id=ret_msg.event_id, worker_infos=ret_msg.worker_infos
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to controller configuration file"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--monitor-ports",
        type=json.loads,
        default=None,
        help='JSON string of monitor ports, e.g. \'{"pull": 8300, "reply": 8400}\'',
    )
    parser.add_argument(
        "--monitor-port",
        type=int,
        default=9001,
        help="The controller pull port to maintain backward compatibility.",
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=-1,
        help="Health check interval in secs, default is -1, which means disabled.",
    )
    parser.add_argument(
        "--lmcache-worker-timeout",
        type=int,
        default=300,
        help="The lmcache worker timeout in seconds.",
    )

    # Parse known args first, then handle extra parameters
    args, extra = parser.parse_known_args()
    extra_params = parse_command_line_extra_params(extra)

    try:
        # Build overrides dictionary from command-line arguments
        override_dict = {}

        # Map command-line arguments to config keys
        arg_mappings = {
            "host": "controller_host",
            "port": "controller_port",
            "monitor_ports": "controller_monitor_ports",
            "health_check_interval": "health_check_interval",
            "lmcache_worker_timeout": "lmcache_worker_timeout",
        }

        for arg_name, config_key in arg_mappings.items():
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                override_dict[config_key] = arg_value

        # Add extra parameters
        if extra_params:
            override_dict.update(extra_params)

        # Load configuration using the generic utility function
        # This replaces the previous manual config loading code
        config = load_controller_config_with_overrides(
            config_file_path=args.config,
            overrides=override_dict,
        )

        # Build controller URLs from config or arguments
        if config.controller_monitor_ports is not None:
            controller_urls = {
                "pull": (
                    f"{config.controller_host}:"
                    f"{config.controller_monitor_ports['pull']}"
                ),
                "reply": (
                    f"{config.controller_host}:"
                    f"{config.controller_monitor_ports['reply']}"
                ),
                "heartbeat": (
                    f"{config.controller_host}:"
                    f"{config.controller_monitor_ports['heartbeat']}"
                    if config.controller_monitor_ports.get("heartbeat")
                    else None
                ),
            }
        else:
            if args.monitor_port != 9001:  # Only warn if explicitly set
                logger.warning(
                    "Argument --monitor-port will be deprecated soon. "
                    "Please use --monitor-ports instead."
                )
            controller_urls = {
                "pull": f"{config.controller_host}:{args.monitor_port}",
                "reply": None,
                "heartbeat": None,
            }

        # Use config values for health check and timeout
        health_check_interval = config.health_check_interval
        lmcache_worker_timeout = config.lmcache_worker_timeout

        app = create_app(controller_urls, health_check_interval, lmcache_worker_timeout)

        logger.info(
            f"Starting LMCache controller at "
            f"{config.controller_host}:{config.controller_port}"
        )
        ports_message = f"Monitoring lmcache workers at ports {controller_urls}"
        logger.info(ports_message)
        logger.info(f"Health check interval: {health_check_interval}s")
        logger.info(f"Worker timeout: {lmcache_worker_timeout}s")

        uvicorn.run(app, host=config.controller_host, port=config.controller_port)
    except TimeoutError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"Failed to start controller: {e}", exc_info=True)
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
