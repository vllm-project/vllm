import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from asyncio import Task
from typing import Optional

import aiohttp
import ray
from ray import actor
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from dllm.config import DPConfig, EPConfig, VllmInstanceConfig
from dllm.constants import (BALANCER_ACTOR_NAME, DLLM_NAMESPACE,
                            INSTANCE_HEALTHCHECK_INTERVAL_SEC)
from dllm.entities import Role, VllmInstanceInfo, VllmInstanceStatus
from dllm.utils import (find_free_port, find_interface_by_ip,
                        find_ip_by_interface, find_node_ip)

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


def select_distributed_torch_interface():
    for env in ["GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"]:
        if env in os.environ:
            return os.environ[env]


class VllmInstance:
    """
    VllmInstance is a vllm engine wrapped by a ray actor, responsibilities:
    1. start vllm api server (and pass some args)
        ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve
    2. do the health check job (report to Controller if any failure)
    """

    _vllm_instance_config: VllmInstanceConfig
    _vllm_instance_info: VllmInstanceInfo
    #: the actor handle of balancer
    _balancer_handle: Optional[actor.ActorHandle]
    _vllm_api_server_process: Optional[subprocess.Popen]
    _vllm_api_server_health_monitor_task: Optional[Task[None]]

    def __init__(self, name: str, vllm_config: VllmInstanceConfig):
        """
        Args:
            env: the environment variables pass to subprocess
            exec_cmd: the vllm api server startup command, e.g. ["vllm", "serve", "--a=1", "--b=2"]
        """
        assert vllm_config.pd_config is not None, "vllm instance PD config is None, abort"
        self._vllm_instance_config = vllm_config
        self._vllm_instance_info = VllmInstanceInfo(
            id=name, uri="", role=vllm_config.pd_config.role)
        self._balancer_handle = None
        self._vllm_api_server_process = None
        self._vllm_api_server_health_monitor_task = None
        self._env = dict(os.environ)
        self._env["HCCL_IF_BASE_PORT"] = os.environ.get(
            'HCCL_IF_BASE_PORT', "50000")

        self.__has_process_started = False

    async def init_dp_master_ip_port(self):
        """
        if dp config is None, init dp master
        """
        intf = select_distributed_torch_interface()
        if intf:
            ip = find_ip_by_interface(intf)
        else:
            ip = find_node_ip()
            intf = find_interface_by_ip(ip)
        assert intf is not None and ip is not None, "failed to find an available network interface for DP group communication, set env GLOO_SOCKET_IFNAME or NCCL_SOCKET_IFNAME manually and try again"
        self._env["GLOO_SOCKET_IFNAME"] = intf
        self._env["NCCL_SOCKET_IFNAME"] = intf
        master_port = find_free_port(ip)
        return ip, master_port

    async def initialize(self) -> None:
        """launch subprocess"""
        logger.info(
            f"initialize with ASCEND_RT_VISIBLE_DEVICES: {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}"
        )
        # normalize and set some env vars
        self._resort_ascend_rt_visible_devices_env()
        self._env["VLLM_USE_V1"] = "1"

        # init all None configs
        if self._vllm_instance_config.dp_config is None:
            self._vllm_instance_config.dp_config = DPConfig()
        if self._vllm_instance_config.ep_config is None:
            self._vllm_instance_config.ep_config = EPConfig()

        # api server options
        # dp slaves have no http api server
        if self._vllm_instance_config.dp_config.dp_size == 0 or self._vllm_instance_config.dp_config.dp_rank == 0:
            protocol = "http"
            ip = find_node_ip()
            port = find_free_port()
            self._vllm_instance_info.uri = f"{protocol}://{ip}:{port}"
            self._vllm_instance_config.exec_cmd.extend(
                ["--host", ip, "--port", str(port)])

        # tp, pd, and dp options
        self._vllm_instance_config.exec_cmd.extend(
            ["--tensor-parallel-size",
             str(self._vllm_instance_config.tp)])
        self._add_dp_command_options()
        self._add_ep_command_options()
        self._add_env()

        logger.info(
            f"initialize with command: {self._vllm_instance_config.exec_cmd}, env:{self._env}"
        )
        self._vllm_api_server_process = subprocess.Popen(
            self._vllm_instance_config.exec_cmd,
            stdout=sys.stdout,
            stdin=sys.stdin,
            stderr=sys.stderr,
            text=True,
            preexec_fn=os.setpgrp,
            env=self._env,
        )

        # use a thread to check and report health status
        # thread safety issue: https://github.com/ray-project/ray/issues/2385
        self._vllm_api_server_health_monitor_task = asyncio.create_task(
            self._monitor_health())

    def _resort_ascend_rt_visible_devices_env(self):
        if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ.keys():
            return
        try:
            device_ids = [
                int(id.strip())
                for id in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            ]
        except ValueError:
            return
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
            map(str, sorted(device_ids)))
        self._env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
            map(str, sorted(device_ids)))

    def _add_dp_command_options(self):
        assert self._vllm_instance_config.dp_config is not None, "vllm instance DP config is None, abort"
        if not self._vllm_instance_config.dp_config.is_dp_enabled():
            return

        self._vllm_instance_config.exec_cmd.extend([
            "--data-parallel-size",
            str(self._vllm_instance_config.dp_config.dp_size),
            "--data-parallel-size-local",
            str(self._vllm_instance_config.dp_config.dp_local_size),
            "--data-parallel-start-rank",
            str(self._vllm_instance_config.dp_config.dp_rank),
            "--data-parallel-address",
            str(self._vllm_instance_config.dp_config.dp_master_ip),
            "--data-parallel-rpc-port",
            str(self._vllm_instance_config.dp_config.dp_master_port),
        ])
        dp_config = self._vllm_instance_config.dp_config
        if dp_config and dp_config.dp_rank > 0:
            self._vllm_instance_config.exec_cmd.extend(["--headless"])

    def _add_ep_command_options(self):
        assert self._vllm_instance_config.ep_config is not None, "vllm instance EP config is None, abort"
        if not self._vllm_instance_config.ep_config.is_ep_enabled():
            return

        self._vllm_instance_config.exec_cmd.extend([
            "--enable-expert-parallel",
        ])

    def _add_env(self):
        if self._vllm_instance_config.env is None:
            return

        env_dict = dict(
            item.split('=') for item in self._vllm_instance_config.env.split())
        for env_key, env_value in env_dict.items():
            self._env[env_key] = env_value

    async def _monitor_health(self):
        """Asynchronously monitor subprocess health and report to controller"""
        while not self._balancer_handle:
            try:
                self._balancer_handle = ray.get_actor(name=BALANCER_ACTOR_NAME,
                                                      namespace=DLLM_NAMESPACE)
            except Exception:
                logger.warning(
                    'Instance get _balancer_handle failed, wait for 1 second and retry.'
                )
                await asyncio.sleep(1)

        async with aiohttp.ClientSession() as session:
            last_report_time = asyncio.get_event_loop().time()
            last_status = self._vllm_instance_info.status
            while True:
                self._vllm_instance_info.status = VllmInstanceStatus.RUNNING
                assert self._vllm_api_server_process is not None, "vllm api server process is not started"
                if self._vllm_api_server_process.poll() is not None:
                    self._vllm_instance_info.status = VllmInstanceStatus.SUBPROCESS_EXITED
                elif self._vllm_instance_info.uri is not None:  # only check DP master's healthy
                    try:
                        async with session.get(
                                f"{self._vllm_instance_info.uri}/health",
                                timeout=aiohttp.ClientTimeout(
                                    total=2)) as response:
                            self._vllm_instance_info.status = (
                                VllmInstanceStatus.HEALTHCHECK_FAILED
                                if response.status != 200 else
                                VllmInstanceStatus.RUNNING)
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        self._vllm_instance_info.status = VllmInstanceStatus.HEALTHCHECK_FAILED
                if (
                        # not healthy
                        self._vllm_instance_info.status !=
                        VllmInstanceStatus.RUNNING
                        # or changed
                        or self._vllm_instance_info.status != last_status
                        # or past quite long time, we should let controller know that we are still alive
                        or asyncio.get_event_loop().time() - last_report_time >
                        INSTANCE_HEALTHCHECK_INTERVAL_SEC):
                    await self._balancer_handle.update_vllm_instance_health.remote(
                        [self._vllm_instance_info
                         ])  # type: ignore # ray remote call
                    last_report_time = asyncio.get_event_loop().time()
                    last_status = self._vllm_instance_info.status

                if self._vllm_instance_info.status == VllmInstanceStatus.SUBPROCESS_EXITED:
                    # terminate self
                    logger.info(
                        "vllm subprocess exited unexpectedly, VllmInstance exit with vllm together"
                    )
                await asyncio.sleep(5)

    async def terminate(self, timeout_s=5):
        if self._vllm_api_server_process is None:
            return

        try:
            pgid = os.getpgid(self._vllm_api_server_process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            logger.info("process already exited")
            return

        # Another way is "self._vllm_api_server_process.terminate()"
        try:
            self._vllm_api_server_process.wait(timeout_s)
        except (TimeoutError, subprocess.TimeoutExpired):
            pass
        finally:
            if self._vllm_api_server_process.poll() is None:
                # Another way is "self._vllm_api_server_process.kill()"
                os.killpg(pgid, signal.SIGKILL)


def start_vllm_instance(
        vllm_instance_config: VllmInstanceConfig,
        pg: Optional[PlacementGroup] = None) -> actor.ActorHandle:
    assert vllm_instance_config.pd_config is not None, "vllm instance PD config is None, abort"
    name = f"vllm-instance-{vllm_instance_config.pd_config.role.name}-{vllm_instance_config.pd_config.pd_rank}"
    assert vllm_instance_config.dp_config is not None, "vllm instance DP config is None, abort"
    if vllm_instance_config.dp_config.dp_size > 1:
        # DP env should be set by `init_dp_config` method
        name = (
            f"{name}-DP-{vllm_instance_config.dp_config.dp_rank}-"
            f"{vllm_instance_config.dp_config.dp_rank+vllm_instance_config.dp_config.dp_local_size}"
        )

    actor_options = {
        "resources": {
            current_platform.device_name:
            vllm_instance_config.dp_config.dp_local_size *
            vllm_instance_config.tp
        },
        "name": name,
        "num_cpus": 0,
    }
    if pg:
        actor_options[
            "scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=pg, )

    vllm_instance_actor = ray.remote(VllmInstance).options(
        **actor_options).remote(name, vllm_instance_config)
    return vllm_instance_actor  # type: ignore # ray actor handle
