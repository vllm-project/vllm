import os
import time
from typing import List
import onnxruntime
from onnxruntime import InferenceSession, SessionOptions, RunOptions
import torch
from metrics import TtsMetrics

import numpy as np
import soundfile as sf
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils import *
from model_setting import ModelSetting


class AudioGenerator:
    def __init__(self, model_setting: ModelSetting):
        self.onnx_session: InferenceSession = None
        self.trt_engine = None
        self.model_setting = model_setting
        self.speaker_embedding = torch.zeros((1, 192, 1), dtype=self.model_setting.dtype).to('cuda')
        self.model_path: str = None

        self.post_init()

    def post_init(self):
        extension = 'onnx' if self.model_setting.runtime == 'onnx' else 'trt'
        self.model_path = os.path.join(self.model_setting.model_dir, f'generator.{self.model_setting.dtype_str}.{extension}')
        logger.info(f'Loading generator model from {self.model_path}')
        if self.model_setting.runtime == 'onnx':
            if self.model_setting.use_onnx_graph:
                providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1'})]
                sess_options = onnxruntime.SessionOptions()
                self.onnx_session = onnxruntime.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
                self.onnx_input_buffer = torch.zeros(1, self.model_setting.chunk_size, self.model_setting.hidden_size, dtype=self.model_setting.dtype).to('cuda')
                self.onnx_output_buffer = torch.zeros(1, 1, self.model_setting.chunk_size * self.model_setting.frame_shift, dtype=self.model_setting.dtype).to('cuda')
            else:
                providers = ["CUDAExecutionProvider"]
                sess_options = onnxruntime.SessionOptions()
                self.onnx_session = onnxruntime.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        else:
            trt_logger = trt.Logger(trt.Logger.ERROR)
            trt_runtime = trt.Runtime(trt_logger)
            with open(self.model_path, 'rb') as f:
                self.trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
        logger.info('Generator model loaded')
        self.warm_up()

    def warm_up(self):
        logger.info('warmup generator...')
        warmup_input = torch.zeros(1, self.model_setting.chunk_size, self.model_setting.hidden_size).to('cuda').to(self.model_setting.dtype)
        metrics = TtsMetrics()
        self.generate_audio(warmup_input, metrics)
        logger.info('warmup generator done')
    
    def generate_audio_onnx(self, latent: torch.Tensor) -> np.ndarray:
        if self.model_setting.use_onnx_graph:
            io_binding = self.onnx_session.io_binding()
            latent_len = latent.size(1)
            ro = onnxruntime.RunOptions()
            ro.add_run_config_entry("gpu_graph_id", "1")
            # copy latent to input buffer
            self.onnx_input_buffer.copy_(latent)
            io_binding.bind_input('input',
                                device_type='cuda',
                                device_id=0,
                                element_type=np.float16,
                                shape=tuple(self.onnx_input_buffer.shape),
                                buffer_ptr=self.onnx_input_buffer.data_ptr())
            io_binding.bind_input('speaker_embedding',
                                device_type='cuda',
                                device_id=0,
                                element_type=np.float16,
                                shape=tuple(self.speaker_embedding.shape),
                                buffer_ptr=self.speaker_embedding.data_ptr())
            io_binding.bind_output('output',
                                    device_type='cuda',
                                    device_id=0,
                                    element_type=np.float16,
                                    shape=tuple(self.onnx_output_buffer.shape),
                                    buffer_ptr=self.onnx_output_buffer.data_ptr())
            self.onnx_session.run_with_iobinding(io_binding, ro)

            # copy output tensor to host
            onnxruntime_outputs = self.onnx_output_buffer.cpu().numpy()[0][0]
        else:
            onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(self.onnx_session.get_inputs(), (latent, self.speaker_embedding))}
            onnxruntime_outputs = self.onnx_session.run(None, onnxruntime_input)
            onnxruntime_outputs = onnxruntime_outputs[0][0][0]
        return onnxruntime_outputs
    
    def generate_audio_trt(self, latent: torch.Tensor) -> np.ndarray:
        with self.trt_engine.create_execution_context() as context:
            stream = cuda.Stream()

            # set input shape as it supports dynamic shape
            latent_len = latent.size(1)
            context.set_input_shape('input', (1, latent_len, self.model_setting.hidden_size))
            context.set_input_shape('speaker_embedding', (1, 192, 1))

            # set input data
            bindings = []
            input_buffer_1 = latent.to('cuda').to(torch.float32)
            bindings.append(input_buffer_1.data_ptr())
            input_buffer_2 = self.speaker_embedding.to('cuda').to(torch.float32)
            bindings.append(input_buffer_2.data_ptr())

            # get output size based on input
            # and set output buffer
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype("output"))
            size = trt.volume(context.get_tensor_shape('output'))
            logger.debug(f'output dtype: {dtype}, size: {size}')
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))
            
            # set input and output buffer to context
            for i in range(len(bindings)):
                context.set_tensor_address(self.trt_engine.get_tensor_name(i), bindings[i])
            
            # execute inference
            logger.debug('execute trt engine')
            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
            logger.debug('execute trt engine done')

            # copy output buffer to host
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)

            return output_buffer

    def generate_chunk_audio(self, latent: torch.Tensor, metric: TtsMetrics, padding: bool, trim_begin: bool, trim_end: bool) -> np.ndarray:
        latent_len = latent.size(1)

        if latent_len % self.model_setting.chunk_size != 0 and padding:
            # pad to chunk size with last frame
            pad_len = self.model_setting.chunk_size - latent_len % self.model_setting.chunk_size
            latent = torch.cat([latent, latent[:, -1:, :].repeat(1, pad_len, 1)], dim=1)

        if self.model_setting.runtime == 'trt':
            audio_outputs = self.generate_audio_trt(latent)
        else:
            audio_outputs = self.generate_audio_onnx(latent)

        metric.audio_chunk_times.append(time.perf_counter())

        if padding:
            audio_outputs = audio_outputs[:latent_len * self.model_setting.frame_shift]

        if self.model_setting.overlap_window > 0 and trim_begin:
            audio_outputs = audio_outputs[self.model_setting.overlap_window * self.model_setting.frame_shift:]

        if self.model_setting.overlap_window > 0 and trim_end:
            audio_outputs = audio_outputs[:-self.model_setting.overlap_window * self.model_setting.frame_shift]

        # convert to fp32
        audio_outputs = audio_outputs.astype(np.float32)
        return audio_outputs
            
    def generate_audio(self, latent: torch.Tensor, metric: TtsMetrics) -> List[np.ndarray]:
        logger.debug(f'latent shape: {latent.shape}')
        latent_len = latent.size(1)
        total_audio: List[np.ndarray] = []
        padding = self.model_setting.chunk_padding
        overlap_window = self.model_setting.overlap_window
        chunk_size = self.model_setting.chunk_size

        if latent.shape[1] > chunk_size:  
            total_len = latent.shape[1]
            start_idx = 0
            audio_seg = self.generate_chunk_audio(latent[:, :chunk_size, :], metric, False, False, True)
            total_audio.append(audio_seg)
            start_idx = chunk_size - overlap_window * 2
            while start_idx < total_len:
                if total_len <= start_idx + chunk_size:
                    audio_seg = self.generate_chunk_audio(latent[:, start_idx:, :], metric, padding, True, False)
                else:  
                    audio_seg = self.generate_chunk_audio(latent[:, start_idx:start_idx + chunk_size, :], metric, False, True, True)  
                total_audio.append(audio_seg)
                start_idx += chunk_size - overlap_window * 2
        else:
            audio_outputs = self.generate_chunk_audio(latent, metric, padding, False, False)
            total_audio.append(audio_outputs)

        return total_audio
