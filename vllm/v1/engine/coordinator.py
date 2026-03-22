# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""数据并行（DP）协调器模块。

本模块实现了 vLLM V1 引擎的数据并行协调器，用于多 DP 引擎部署场景：
- 收集每个 DP 引擎的统计信息（等待和运行队列长度）
- 发布统计信息到前端 API 服务器用于负载均衡决策
- 跟踪当前 DP"请求波次"编号和引擎运行状态
- 广播 START_DP_WAVE 消息以协调引擎从暂停状态进入运行状态
- 支持弹性 EP 扩缩容场景
"""
import copy
import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_tcp_uri, make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """用于数据并行部署（DP>1）的协调器进程。

    在多个 DP 引擎进程和一个或多个前端 API 服务器进程之间进行中介。

    主要功能：
    - 收集每个 DP 引擎的统计信息（当前仅为等待和运行队列长度），
      并发布到所有前端用于负载均衡决策
    - 跟踪当前 DP"请求波次"编号和引擎运行状态，由 DP rank 0 引擎
      接收并发布到前端进程
    - 广播 START_DP_WAVE 消息使引擎从暂停状态进入运行状态

    引擎在以下情况下进入运行状态：
    1. 前端在引擎暂停时发送新请求，并发通知协调器
    2. 引擎在暂停状态下接收到过时波次的请求时通知协调器

    注意：在外部 LB 模式下部署时，引擎不会发布统计信息，
    因此仅在请求波次/运行状态变化时发送更新到前端。
    """

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        """等待 ZMQ 地址报告。

        Args:
            zmq_addr_pipe: ZMQ 地址管道

        Returns:
            (front_publish_address, back_output_address, back_publish_address) 元组

        Raises:
            RuntimeError: 超时或进程失败时抛出
        """
        try:
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=30
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    "during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
    ):
        """初始化 DP 协调器。

        Args:
            parallel_config: 并行配置
            enable_wave_coordination: 是否启用波次协调
        """
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # 当不在外部或混合 DP LB 模式时，假设协调器与前端进程共存
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # NOTE(yongji): 处理从节点内到节点间的扩展
        if parallel_config.enable_elastic_ep:
            local_only_eng = False

        def bind_address(local_only: bool) -> str:
            return (
                get_engine_client_zmq_addr(local_only=True, host=host)
                if local_only
                else get_tcp_uri(host, 0)
            )

        # 绑定三个 ZMQ 地址
        front_publish_address = bind_address(local_only)  # 前端发布
        back_output_address = bind_address(local_only_eng)  # 后端输出
        back_publish_address = bind_address(local_only_eng)  # 后端发布

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        """返回统计信息发布地址。"""
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """返回引擎套接字地址元组。

        Returns:
            (ZMQ 输入地址，ZMQ 输出地址) 元组
        """
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """关闭协调器进程。

        Args:
            timeout: 关闭超时时间（秒）
        """
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class EngineState:
    """引擎状态类，用于跟踪请求计数。

    Attributes:
        request_counts: 请求计数列表 [waiting, running]
    """

    def __init__(self):
        self.request_counts = [0, 0]  # [waiting, running]


class DPCoordinatorProc:
    """DP 协调器进程实现类。

    负责处理 ZMQ 套接字通信、统计信息收集和波次协调。

    Attributes:
        engines: 引擎状态列表
        stats_update_interval_ms: 统计信息更新间隔（毫秒）
        enable_wave_coordination: 是否启用波次协调
    """

    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        """初始化 DP 协调器进程。

        Args:
            engine_count: 引擎数量
            min_stats_update_interval_ms: 最小统计信息更新间隔（毫秒）
            enable_wave_coordination: 是否启用波次协调
        """
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        """运行协调器的主入口函数。

        Args:
            engine_count: 引擎数量
            front_publish_address: 前端发布地址
            back_output_address: 后端输出地址
            back_publish_address: 后端发布地址
            zmq_addr_pipe: ZMQ 地址管道
            min_stats_update_interval_ms: 最小统计信息更新间隔
            enable_wave_coordination: 是否启用波次协调
        """
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        """处理输入套接字的主循环。

        负责：
        - 监听前端和后端的 ZMQ 消息
        - 收集引擎统计信息并发布到前端
        - 处理波次协调逻辑
        - 处理弹性 EP 扩缩容通知

        Args:
            front_publish_address: 前端发布地址
            back_output_address: 后端输出地址
            back_publish_address: 后端发布地址
            zmq_addr_pipe: ZMQ 地址管道
        """
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # 用于跟踪请求波次进度
        current_wave = 0
        engines_running = False

        # 用于跟踪内部负载均衡的请求计数
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # 等待所有引擎订阅
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # 发送就绪消息到引擎
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # 如果统计信息已变化，则按 stats_update_interval_ms 间隔发送
                # 否则每 5 秒发送一次
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # 等待至少 50ms 以确保已收到当前步的所有统计信息
                min_timeout = 50 if last_step_counts is None else 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # Poller 超时 - 发布当前统计信息到前端
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_back in events:
                    buffer = publish_back.recv()
                    if buffer == b"\x01":
                        # NOTE(yongji): 新启动的引擎已订阅
                        # 我们需要在此处发送 READY 消息，而不是从引擎核心客户端
                        # 接收 SCALE_ELASTIC_EP 通知，因为 SCALE_ELASTIC_EP
                        # 仅在新引擎完成初始化后才会发送
                        # 订阅消息则由每个引擎在初始化期间发送
                        publish_back.send(b"READY")
                    elif buffer != b"\x00":
                        logger.error(
                            "DP Coordinator received unexpected message from engines"
                        )

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # 忽略订阅消息
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        # 处理扩展通知
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            # NOTE(yongji): 处理新引擎 current_wave = 0 的情况
                            # 如果现有引擎刚刚完成波次且 engine_running 尚未在
                            # CoordinatorProc 更新，路由到新引擎的请求可能无法
                            # 唤醒现有引擎，因为 0 < request.wave < 现有引擎的
                            # current_wave
                            # 注意 0 是新引擎的波次编号
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue  # 跳过正常的引擎通知处理

                    # 波次协调：处理来自前端的新请求消息
                    # 仅在启用波次协调时处理
                    if self.enable_wave_coordination:
                        # 在前端 XPUB 套接字上收到消息
                        # 来自 API 服务器在引擎暂停时发送新请求
                        # 以便唤醒其他引擎
                        engine_to_exclude, wave = decoded
                        if not engines_running:
                            if wave < current_wave:
                                # 如果波次编号过时，确保消息被所有引擎处理
                                engine_to_exclude = None

                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                if output_back in events:
                    # 从引擎收到消息

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        # 1. 更新的请求负载统计 - 更新本地状态
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    # 波次协调：处理波次完成和启动通知
                    # 仅在启用波次协调时处理
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            # 2. 来自 rank 0 引擎的通知，表示已进入全局暂停状态
                            # (engines_running==False)
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            # 3. 引擎接收到非当前波次的请求
                            # 必须确保其他引擎进入下一波次（竞态条件处理）
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        """向所有引擎广播 START_DP_WAVE 消息。

        消息包含当前波次编号和已收到该波次请求的引擎索引
        （该引擎不需要额外的通知）。

        Args:
            socket: ZMQ 套接字
            wave: 波次编号
            exclude_engine_index: 要排除的引擎索引
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self, do_copy=False) -> list[list[int]]:
        """返回每个引擎的 [waiting, running] 计数列表。

        Args:
            do_copy: 是否返回副本

        Returns:
            每个引擎的请求计数列表
        """
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
