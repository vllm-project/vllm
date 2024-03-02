from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a 
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: Optional[str],
        load_format: str,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture

        if os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true":
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import snapshot_download  # pylint: disable=C
            if not os.path.exists(model):
                model_path = snapshot_download(model_id=model,
                                               cache_dir=download_dir,
                                               revision=revision)
            else:
                model_path = model
            self.model = model_path
            self.download_dir = model_path
            self.tokenizer = model_path

        self.hf_config = get_config(self.model, trust_remote_code, revision,
                                    code_revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config,
                                                     max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        supported_load_format = [
            "auto", "pt", "safetensors", "npcache", "dummy"
        ]
        rocm_not_supported_load_format = []
        if load_format not in supported_load_format:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in supported_load_format
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format \'{load_format}\' is not supported in ROCm. "
                f"Supported load format are "
                f"{rocm_supported_load_format}")

        # TODO: Remove this check once HF updates the pt weights of Mixtral.
        architectures = getattr(self.hf_config, "architectures", [])
        if "MixtralForCausalLM" in architectures and load_format == "pt":
            raise ValueError(
                "Currently, the 'pt' format is not supported for Mixtral. "
                "Please use the 'safetensors' format instead. ")
        self.load_format = load_format

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'.")
        self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        supported_quantization = ["awq", "gptq", "squeezellm", "marlin"]
        rocm_not_supported_quantization = ["awq", "marlin"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        if hf_quant_config is not None:

            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
            # If the GPTQ model is serialized in marlin format, use marlin.
            if (hf_quant_method == "gptq"
                    and "is_marlin_format" in hf_quant_config
                    and hf_quant_config["is_marlin_format"]):
                hf_quant_method = "marlin"
            if self.quantization is None:
                self.quantization = hf_quant_method
            elif self.quantization != hf_quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({hf_quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            if is_hip(
            ) and self.quantization in rocm_not_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not supported "
                    f"in ROCm.")
            if self.quantization != "marlin":
                logger.warning(
                    f"{self.quantization} quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.")

    def _verify_cuda_graph(self) -> None:
        if self.max_context_len_to_capture is None:
            self.max_context_len_to_capture = self.max_model_len
        self.max_context_len_to_capture = min(self.max_context_len_to_capture,
                                              self.max_model_len)

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size}).")

    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        if hasattr(self.hf_config, "head_dim"):
            return self.hf_config.head_dim
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        cache_dtype: str,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.cache_dtype = cache_dtype
        self.sliding_window = sliding_window
        self._verify_args()
        self._verify_cache_dtype()

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def metrics_info(self):
        # convert cache_config to dict(key: str, value:str) for prometheus metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype == "fp8_e5m2":
            nvcc_cuda_version = get_nvcc_cuda_version()
            if nvcc_cuda_version and nvcc_cuda_version < Version("11.8"):
                raise ValueError(
                    "FP8 is not supported when cuda version is lower than 11.8."
                )
            device_name = torch.cuda.get_device_name()
            if "AMD" in device_name:
                raise NotImplementedError(
                    "FP8_E5M2 KV Cache on AMD GPU has not been supported yet.")
            logger.info(
                "Using fp8_e5m2 data type to store kv cache. It reduces "
                "the GPU memory footprint and boosts the performance. "
                "But it may cause slight accuracy drop. "
                "Currently we only support fp8 without scaling factors and "
                "make e5m2 as a default format.")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / _GB:.2f} GiB out of "
               f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
               "allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. " + msg)


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
        max_parallel_loading_workers: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        if is_neuron():
            # For Neuron device support, here we assign TP=1 to avoid sharding within vLLM directly.
            # Transformer-neuronx would take neuron_tp_degree attribute, and distribute the workload
            # to multiple NeuronCores.
            self.tensor_parallel_size = 1
            self.neuron_tp_degree = tensor_parallel_size
        else:
            self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce

        self.world_size = pipeline_parallel_size * self.tensor_parallel_size
        # Ray worker is not supported for Neuron backend.
        if self.world_size > 1 and not is_neuron():
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")
        if not self.disable_custom_all_reduce and self.world_size > 1:
            if is_hip():
                self.disable_custom_all_reduce = True
                logger.info(
                    "Disabled the custom all-reduce kernel because it is not "
                    "supported on AMD GPUs.")
            elif self.pipeline_parallel_size > 1:
                self.disable_custom_all_reduce = True
                logger.info(
                    "Disabled the custom all-reduce kernel because it is not "
                    "supported with pipeline parallelism.")

        # FIXME(woosuk): Fix the stability issues and re-enable the custom
        # all-reduce kernel.
        if not self.disable_custom_all_reduce and self.world_size > 1:
            self.disable_custom_all_reduce = True
            logger.info(
                "Custom all-reduce kernels are temporarily disabled due to "
                "stability issues. We will re-enable them once the issues are "
                "resolved.")


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        max_paddings: Maximum number of paddings to be added to a batch.
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
        max_paddings: int,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")


class DeviceConfig:

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            if torch.cuda.is_available():
                self.device_type = "cuda"
            elif is_neuron():
                self.device_type = "neuron"
            else:
                raise RuntimeError("No supported device detected.")
        else:
            # Device type is assigned explicitly
            self.device_type = device

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron"]:
            self.device = torch.device("cpu")
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)

    @property
    def is_neuron(self):
        return self.device_type == "neuron"


@dataclass
class LoRAConfig:
    max_lora_rank: int
    max_loras: int
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[torch.dtype] = None
    lora_extra_vocab_size: int = 256
    # This is a constant.
    lora_vocab_padding_size: ClassVar[int] = 256

    def __post_init__(self):
        # Keep this in sync with csrc/punica/bgmv/bgmv_config.h
        possible_max_ranks = (8, 16, 32, 64)
        possible_lora_extra_vocab_size = (0, 256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.quantization is not None:
            raise ValueError(
                "LoRA is not supported with quantized models yet.")

    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        if scheduler_config.max_num_batched_tokens > 65528:
            raise ValueError(
                "Due to limitations of the custom LoRA CUDA kernel, "
                "max_num_batched_tokens must be <= 65528 when "
                "LoRA is enabled.")


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_ROCM_NOT_SUPPORTED_DTYPE = ["float", "float32"]


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # Following the common practice, we use float16 for float32
                # models.
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if is_hip() and torch_dtype == torch.float32:
        rocm_supported_dtypes = [
            k for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items()
            if (k not in _ROCM_NOT_SUPPORTED_DTYPE)
        ]
        raise ValueError(f"dtype \'{dtype}\' is not supported in ROCm. "
                         f"Supported dtypes are {rocm_supported_dtypes}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            f"{possible_keys}. Assuming the model's maximum length is "
            f"{default_max_len}.")
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        assert "factor" in rope_scaling
        scaling_factor = rope_scaling["factor"]
        if rope_scaling["type"] == "yarn":
            derived_max_model_len = rope_scaling[
                "original_max_position_embeddings"]
        derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        raise ValueError(
            f"User-specified max_model_len ({max_model_len}) is greater than "
            f"the derived max_model_len ({max_len_key}={derived_max_model_len}"
            " in model's config.json). This may lead to incorrect model "
            "outputs or CUDA errors. Make sure the value is correct and "
            "within the model context size.")
    return int(max_model_len)
