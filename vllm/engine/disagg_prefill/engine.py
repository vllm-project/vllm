import pickle
import signal
from typing import Optional, Union
import threading
import time

import cloudpickle
from vllm import AsyncEngineArgs
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, RPCAbortRequest,
                                         RPCError, RPCProcessRequest,
                                         RPCUProfileRequest, REQUEST_OUTPUTS_T)
# yapf: enable
from vllm.logger import init_logger
from vllm import AsyncEngineArgs, LLMEngine
from vllm.usage.usage_lib import UsageContext
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.rabbitmq.rabbitmq import RabbitMQ
from vllm.envs import VLLM_DISTRIBUTED_KV_ROLE
from vllm.engine.multiprocessing.engine import POLLING_TIMEOUT_MS
from vllm.engine.disagg_prefill.client import (decode_req_id,
                                               GLOBAL_DECODE_REQ_QUEUE_NAME)

CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]

logger = init_logger(__name__)


class PDMQLLMEngine(MQLLMEngine):
    IS_KV_PRODUCER: bool = (VLLM_DISTRIBUTED_KV_ROLE in ["producer"])
    IS_KV_CONSUMER: bool = (VLLM_DISTRIBUTED_KV_ROLE in ["consumer"])

    def __init__(self,
                 ipc_path: str,
                 use_async_sockets: bool,
                 *args,
                 mq_addr: str,
                 mq_port: int,
                 log_requests: bool = True,
                 **kwargs) -> None:
        super().__init__(ipc_path, use_async_sockets, *args, **kwargs)

        self.mq_addr = mq_addr
        self.mq_port = mq_port

        if self.IS_KV_CONSUMER:
            self.global_decode_req_mq: RabbitMQ = RabbitMQ(mq_addr, mq_port, GLOBAL_DECODE_REQ_QUEUE_NAME)
            self.res_mq_map = {}
            self.consumer_engine_loop = threading.Thread(target=self.consumer_run_engine_loop)

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str, mq_addr: str, mq_port: int):
        engine_config = engine_args.create_engine_config()

        executor_class = LLMEngine._get_executor_cls(engine_config)

        return cls(
            ipc_path=ipc_path,
            use_async_sockets=engine_config.model_config.use_async_output_proc,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            mq_addr=mq_addr,
            mq_port=mq_port)

    def _consumer_handle_process_request(self, request: RPCProcessRequest):
        request_id = request.request_id

        if self._errored_with is not None:
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=True,
                               exception=ENGINE_DEAD_ERROR(self._errored_with))
            self._consumer_send_outputs(rpc_err)

        try:
            self.engine.add_request(
                request_id=request_id,
                prompt=request.prompt,
                params=request.params,
                lora_request=request.lora_request,
                trace_headers=request.trace_headers,
                prompt_adapter_request=request.prompt_adapter_request)

            if self.log_requests:
                logger.info("Added request %s.", request.request_id)

        except Exception as e:
            is_errored = self._errored_with is not None
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=is_errored,
                               exception=e)
            self._consumer_send_outputs(rpc_err)

            self.engine.abort_request(request_id)

    def consumer_handle_new_input(self) -> bool:
        frame, body = self.global_decode_req_mq.pull(auto_ack=True)
        if frame is None:
            return False

        req_info: list[str] = pickle.loads(body)
        assert len(req_info) > 0, "Empty request"
        request_str = req_info[0]
        request = pickle.loads(request_str)
    
        if isinstance(request, RPCProcessRequest):
            if len(req_info) > 1:
                lprocs = cloudpickle.loads(req_info[1])
                request.params.logits_processors = lprocs
            self._consumer_handle_process_request(request)
        elif isinstance(request, RPCAbortRequest):
            self._handle_abort_request(request)
        elif isinstance(request, RPCUProfileRequest):
            if request == RPCUProfileRequest.START_PROFILE:
                self.start_profile()
            else:
                self.stop_profile()
        else:
            raise ValueError("Unknown RPCRequest Type: {request}")

        return True

    def consumer_run_engine_loop(self):
        while True:
            if not self.engine.has_unfinished_requests():
                have_req = self.consumer_handle_new_input()
                if not have_req:
                    time.sleep(0.1)

            request_outputs = self.engine_step()
            if not self.use_async_sockets:
                self._consumer_send_outputs(request_outputs)

    def consumer_cleanup(self):
        for mq in self.res_mq_map.values():
            mq.close()
    
    def cleanup(self):
        super().cleanup()

        if self.IS_KV_CONSUMER:
            self.consumer_cleanup()
    
    def run_engine_loop(self):
        self._alive()
        if not self.engine.has_unfinished_requests():
            # Poll until there is work to do.
            while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                self._alive()
                self.engine.do_log_stats()
                logger.debug("Waiting for new requests in engine loop.")

        self.handle_new_input()

    def start(self):
        try:
            try:
                logger.debug("Starting Startup Loop.")
                self.run_startup_loop()
                logger.debug("Starting heartbeat thread")
                self.heartbeat_thread.start()
                logger.debug("Starting Engine Loop.")
                if self.IS_KV_CONSUMER:
                    self.consumer_engine_loop.start()
                    self.run_engine_loop()
                else:
                    super().run_engine_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug("Shutting down MQLLMEngine.")
        finally:
            logger.debug("MQLLMEngine is shut down.")
            self.cleanup()

    def _consumer_send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            logger.debug(f"Sending request to MQLLMEngine producer.")
            send_mq_name = decode_req_id(outputs[0].request_id)[1]
            output_bytes = pickle.dumps(outputs)
            logger.debug(f"send to MQ {send_mq_name}")
            if send_mq_name not in self.res_mq_map:
                self.res_mq_map[send_mq_name] = RabbitMQ(self.mq_addr, self.mq_port, send_mq_name)

            send_mq = self.res_mq_map[send_mq_name]
            send_mq.push(output_bytes)
            

    def _async_socket_engine_callback(self,
                                      request_outputs: REQUEST_OUTPUTS_T):
        if self.IS_KV_CONSUMER:
            self._consumer_send_outputs(request_outputs)
            self.consumer_handle_new_input()
        else:
            super()._async_socket_engine_callback(request_outputs)

def run_pd_mp_engine(engine_args: AsyncEngineArgs, usage_context: UsageContext,
                  ipc_path: str, addr: str, port: int):

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm
        raise KeyboardInterrupt("MQLLMEngine terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine = PDMQLLMEngine.from_engine_args(engine_args=engine_args,
                                                usage_context=usage_context,
                                                ipc_path=ipc_path,
                                                mq_addr=addr,
                                                mq_port=port)
    engine.start()
