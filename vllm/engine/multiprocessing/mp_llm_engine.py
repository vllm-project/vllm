import ray
import zmq
import cloudpickle
import pickle
from typing import Any, Type, Union, Iterator
from contextlib import contextmanager

import vllm.envs as envs
from vllm import AsyncEngineArgs, LLMEngine, AsyncLLMEngine
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.engine.multiprocessing import (VLLM_RPC_SUCCESS_STR,
                             RPCUtilityRequest)
from vllm.utils import print_warning_once
from vllm.usage.usage_lib import UsageContext

CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]

logger = init_logger(__name__)

class MPLLMEngine:
    """A multiprocessing wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to enable use
    in asynchronous manner. It runs a background loop and uses zeromq to 
    recieve new requests and stream outputs incrementally to another process.
    
    The :class:`LLMEngine` is kicked off when a new RPCGenerateRequest 
    is recieved by the input_socket.
    
    The self.engine_loop checks the input_socket for new requests,
    adds them to the LLMEngine if there are any, calls the internal
    :class:`LLMEngine.step()` and sends the RequestOutputs back over
    the output_socket. 

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        async_engine_args: AsyncLLMEngine args
        log_requests: Whether to log the requests.
    """

    _engine_class: Type[LLMEngine] = LLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 ipc_path: str,
                 log_requests: bool = True,
                 **kwargs) -> None:

        if engine_use_ray:
            raise NotImplementedError("Not yet supported.")
        
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.engine = self._init_engine(*args, **kwargs)

        if self.engine_use_ray:
            print_warning_once(
                "DEPRECATED. `--engine-use-ray` is deprecated and will "
                "be removed in a future update. "
                "See https://github.com/vllm-project/vllm/issues/7045.")

            if envs.VLLM_ALLOW_ENGINE_USE_RAY:
                print_warning_once(
                    "VLLM_ALLOW_ENGINE_USE_RAY is set, force engine use Ray")
            else:
                raise ValueError("`--engine-use-ray` is deprecated. "
                                 "Set `VLLM_ALLOW_ENGINE_USE_RAY=1` to "
                                 "force use it")

        self.ctx = zmq.Context()

        # Recieve RPCGenerateRequest from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}_input_socket")

        # Send streams of RequestOutput back to Client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}_output_socket")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}_data_socket"

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs, 
                         usage_context: UsageContext, ipc_path: str):
        """Creates an RPCLLM engine from the engine arguments."""
        
        engine_config = engine_args.create_engine_config()

        if engine_args.engine_use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        # TODO: better abstraction?
        executor_class = AsyncLLMEngine._get_executor_cls(engine_config)

        return cls(
            executor_class.uses_ray,
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            ipc_path=ipc_path,
        )

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        self.input_socket.close()
        self.output_socket.close()
        self.ctx.destroy(linger=0)
        del self.engine

    def _init_engine(self, *args, **kwargs) -> Union[LLMEngine, "ray.ObjectRef"]:
        """Initialize the LLMEngine"""

        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = kwargs["cache_config"]
            parallel_config = kwargs["parallel_config"]
            if (parallel_config.tensor_parallel_size == 1
                    and parallel_config.pipeline_parallel_size == 1):
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)
    
    def run_background_loop(self):
        """Entrypoint that kicks off the background processing loop."""
        
        # Allow RPCClient to query data in startup phase. 
        self.run_startup_loop()

        # Kick off core processing loop.
        self.run_engine_loop()
    
    @contextmanager
    def make_data_socket(self) -> Iterator[zmq.Socket]:
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Loop over startup RPCRequests from RPCClient."""
        
        with self.make_data_socket() as socket:

            # Loop until the RPCClient has all the data it needs.
            client_is_ready = False
            while not client_is_ready:
                try:
                    identity, message = socket.recv_multipart(copy=False)
                    request: RPCUtilityRequest = cloudpickle.loads(message.buffer)

                    # Handle the query from the Client.
                    if request == RPCUtilityRequest.GET_MODEL_CONFIG:
                        response = self.engine.get_model_config()
                    elif request == RPCUtilityRequest.GET_DECODING_CONFIG:
                        response = self.engine.get_decoding_config()
                    elif request == RPCUtilityRequest.GET_LORA_CONFIG:
                        response = self.engine.get_lora_config()
                    elif request == RPCUtilityRequest.GET_SCHEDULER_CONFIG:
                        response = self.engine.get_scheduler_config()
                    elif request == RPCUtilityRequest.GET_PARALLEL_CONFIG:
                        response = self.engine.get_parallel_config()
                    elif request == RPCUtilityRequest.IS_SERVER_READY:
                        response = VLLM_RPC_SUCCESS_STR
                    elif request == RPCUtilityRequest.IS_TRACING_ENABLED:
                        response = self.engine.is_tracing_enabled()
                    elif request == RPCUtilityRequest.CLIENT_IS_READY:
                        response = VLLM_RPC_SUCCESS_STR
                        # Once client ready, breakout of loop.
                        client_is_ready = True
                    else:
                        raise ValueError(f"Unknown RPCRequest: {request}")
                
                    socket.send_multipart(
                        (identity, pickle.dumps(response)), copy=False)

                except Exception as e:
                    socket.send_multipart((identity, pickle.dumps(e)), copy=False)

    def run_engine_loop(self) -> None:
        # TODO: handle PP

        while True:
            # Block until there is a new request.
            if not self.engine.has_unfinished_requests():
                self.wait_for_new_requests()

            # Add new work from input socket.
            self.maybe_add_new_requests()
            
            # Engine step.
            request_outputs = self.engine.step()
            
            # Stream results to output socket.
            self.stream_outputs(request_outputs)        


    def wait_for_new_requests(self):
        while self.input_socket.poll(timeout=10000) == 0:
            logger.debug("Waiting for new request.")

    def stream_outputs(self, request_outputs):
        self.output_socket.send_multipart(
            (pickle.dumps(request_outputs),), copy=False)

    def maybe_add_new_requests(self):
        while self.input_socket.poll(timeout=0) != 0:
            message = self.input_socket.recv(copy=False)
            generate_rpc_request = pickle.loads(message.buffer)
            self.engine.add_request(
                request_id=generate_rpc_request.request_id,
                inputs=generate_rpc_request.inputs,
                params=generate_rpc_request.sampling_params,
                lora_request=generate_rpc_request.lora_request,
                trace_headers=generate_rpc_request.trace_headers,
                prompt_adapter_request=generate_rpc_request.prompt_adapter_request,
            )


def run_mp_engine(engine_args: AsyncEngineArgs, 
                  usage_context: UsageContext, 
                  ipc_path: str):
    engine = MPLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=usage_context,
        ipc_path=ipc_path)

    engine.run_background_loop()
