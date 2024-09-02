import zmq
import cloudpickle, pickle
from vllm.logger import init_logger
from vllm import EngineArgs, LLMEngine
from vllm.entrypoints.openai.rpc import (VLLM_RPC_SUCCESS_STR,
                                         RPCUtilityRequest)

logger = init_logger(__name__)

class MPLLMEngine:
    def __init__(self, engine_args) -> None:
        self.engine = LLMEngine.from_engine_args(engine_args)

        self.ctx = zmq.Context()

        self.new_req_socket = self.ctx.socket(zmq.constants.PULL)
        self.new_req_socket.bind("ipc:///tmp/new_req_socket")

        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind("ipc:///tmp/output_socket")

        self.data_socket = self.ctx.socket(zmq.constants.ROUTER)
        self.data_socket.bind("ipc:///tmp/data_socket")

    def run(self):
        logger.info("Running Startup Loop.")
        self.startup_loop()
        logger.info("Running Engine Loop.")
        self.engine_loop()

    def startup_loop(self):
        client_is_ready = False
        while not client_is_ready:
            identity, message = self.data_socket.recv_multipart(copy=False)
            request = cloudpickle.loads(message.buffer)
            if request in [
                RPCUtilityRequest.GET_MODEL_CONFIG,
                RPCUtilityRequest.GET_PARALLEL_CONFIG,
                RPCUtilityRequest.GET_DECODING_CONFIG,
                RPCUtilityRequest.GET_SCHEDULER_CONFIG,
                RPCUtilityRequest.GET_LORA_CONFIG
            ]:
                config = self.get_config(request)
                self.data_socket.send_multipart((identity, pickle.dumps(config)), copy=False)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                self.data_socket.send_multipart((identity, pickle.dumps(VLLM_RPC_SUCCESS_STR)), copy=False)
            elif request == RPCUtilityRequest.IS_TRACING_ENABLED:
                self.data_socket.send_multipart((identity, pickle.dumps(self.engine.is_tracing_enabled())), copy=False)
            elif request == RPCUtilityRequest.CLIENT_IS_READY:
                self.data_socket.send_multipart((identity, pickle.dumps(VLLM_RPC_SUCCESS_STR)), copy=False)
                client_is_ready = True
                self.data_socket.close()
                del self.data_socket

    def engine_loop(self):
        while True:
            if not self.engine.has_unfinished_requests():
                self.wait_for_new_requests()
            
            self.add_new_requests()
            request_outputs = self.engine.step()
            self.send_request_outputs(request_outputs)

    def send_request_outputs(self, request_outputs):
        self.output_socket.send_multipart(
            (pickle.dumps(request_outputs),), copy=False)

    def add_new_requests(self):
        while self.new_req_socket.poll(timeout=0) != 0:
            message = self.new_req_socket.recv(copy=False)
            generate_rpc_request = pickle.loads(message.buffer)
            self.engine.add_request(
                request_id=generate_rpc_request.request_id,
                inputs=generate_rpc_request.inputs,
                params=generate_rpc_request.sampling_params,
                lora_request=generate_rpc_request.lora_request,
                trace_headers=generate_rpc_request.trace_headers,
                prompt_adapter_request=generate_rpc_request.prompt_adapter_request,
            )
    
    def wait_for_new_requests(self):
        while self.new_req_socket.poll(timeout=1000) == 0:
            logger.info("Waiting for new requests...")
        logger.info("Found new request!")
    
    def get_config(self, request):
        if request == RPCUtilityRequest.GET_MODEL_CONFIG:
            model_config = self.engine.get_model_config()
            return model_config
        elif request == RPCUtilityRequest.GET_DECODING_CONFIG:
            return self.engine.get_decoding_config()
        elif request == RPCUtilityRequest.GET_LORA_CONFIG:
            return self.engine.get_lora_config()
        elif request == RPCUtilityRequest.GET_SCHEDULER_CONFIG:
            return self.engine.get_scheduler_config()
        elif request == RPCUtilityRequest.GET_PARALLEL_CONFIG:
            return self.engine.get_parallel_config()
        else:
            raise ValueError("Unknown Config Request: %s", request)

def run_rpc_server(engine_args: EngineArgs):
    engine = MPLLMEngine(engine_args)
    engine.run()
