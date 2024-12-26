import torch


class ModelSetting:
    def __init__(self,
                 model_dir: str = None,
                 runtime: str = 'onnx',
                 dtype: str = 'float32',
                 chunk_size: int = 20,
                 overlap_window: int = 0,
                 hidden_size: int = 1536,
                 frame_shift: int = 1200,
                 streaming: bool = False,
                 first_chunk_size: int = 10,
                 chunk_padding: bool = True,
                 cut_tail: int = 150,
                 support_lora: bool = False,
                 scale_rate: float = 2.7,
                 profile_run: bool = False):
        self.model_dir = model_dir
        self.runtime = runtime
        self.chunk_size = chunk_size
        self.overlap_window = overlap_window
        self.hidden_size = hidden_size
        self.frame_shift = frame_shift
        self.streaming = streaming
        self.first_chunk_size = first_chunk_size
        self.chunk_padding = chunk_padding
        self.dtype_str = dtype
        if dtype == 'float32':
            self.dtype = torch.float32
        elif dtype == 'float16':
            self.dtype = torch.float16
        self.cut_tail = cut_tail
        self.support_lora = support_lora
        self.use_onnx_graph = False

        self.gpu_memory_utilization = 0.3
        self.scale_rate = scale_rate
        self.profile_run = profile_run