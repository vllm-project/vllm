from typing import Optional, Union
from dataclasses import dataclass
import copy
import os

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory

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
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        flash_style: Enable flash style page attention. This is only supported
            by llama models.
        max_chunked_prefill_len: The maximum length of tokens for prefill
            requests. Longer requests will be chunked into multiple chunks.
            -1 means no chunking (disabled). This features is only supported
            for flash style attention.
    """

    def __init__(self,
                 model: str,
                 tokenizer: str,
                 tokenizer_mode: str,
                 trust_remote_code: bool,
                 download_dir: Optional[str],
                 load_format: str,
                 dtype: Union[str, torch.dtype],
                 seed: int,
                 revision: Optional[str],
                 tokenizer_revision: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 quantization: Optional[str] = None,
                 enable_cuda_graph: bool = False,
                 cuda_graph_max_context_len: int = 5000,
                 cuda_graph_cache_size: int = 10,
                 flash_style: bool = False,
                 max_chunked_prefill_len: int = -1,
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization

        if os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true":
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import snapshot_download  # pylint: disable=C
            model_path = snapshot_download(model_id=model,
                                           cache_dir=download_dir,
                                           revision=revision)
            self.model = model_path
            self.download_dir = model_path
            self.tokenizer = model_path

        self.hf_config = get_config(self.model, trust_remote_code, revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config,
                                                     max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()
        self._verify_quantization()

        self.enable_cuda_graph = enable_cuda_graph
        self.cuda_graph_max_context_len = cuda_graph_max_context_len
        self.cuda_graph_cache_size = cuda_graph_cache_size
        self.flash_style = flash_style
        self.max_chunked_prefill_len = max_chunked_prefill_len

        self._verify_chunk_prefill()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in [
                "auto", "pt", "safetensors", "npcache", "dummy"
        ]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")

        self.load_format = load_format

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'.")

        self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        supported_quantization = ["awq", "squeezellm"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        if hf_quant_config is not None:
            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
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
            logger.warning(f"{self.quantization} quantization is not fully "
                           "optimized yet. The speed can be slower than "
                           "non-quantized models.")

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

    def _verify_chunk_prefill(self) -> None:
        if self.max_chunked_prefill_len == 0:
            raise ValueError("max_chunked_prefill_len can't be 0")
        if self.max_chunked_prefill_len > 0 and not self.flash_style:
            raise ValueError(
                "chunked prefill is only supported for flash style")

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
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
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        sliding_window: Optional[int] = None,
        flash_style: bool = False,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.sliding_window = sliding_window
        self.flash_style = flash_style
        self._verify_args()

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

        if self.flash_style and self.block_size < 32:
            raise ValueError(
                "Flash style attention only supports block size >= 32. Got"
                f"{self.block_size}.")
        if not self.flash_style and self.block_size > 32:
            raise ValueError(
                "vLLM Page attention only supports block size <= 32. Got"
                f"{self.block_size}.")

        if self.flash_style:
            logger.info("Flash attention enabled.")

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
        disable_shared_memory: Whether to not use shared memory for
            engine<->worker communication. Not used if Ray isn't used.
        ray_workers_use_nsight: Whether to profile Ray workers with nvidia
            nsight (See https://github.com/ray-project/ray/pull/39998 ).
    """

    def __init__(self,
                 pipeline_parallel_size: int,
                 tensor_parallel_size: int,
                 worker_use_ray: bool,
                 disable_shared_memory: bool = False,
                 num_tokenizer_actors: int = 0,
                 tokenizer_actor_options: Optional[dict] = None,
                 ray_workers_use_nsight: bool = False) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.num_tokenizer_actors = num_tokenizer_actors
        self.tokenizer_actor_options = tokenizer_actor_options
        self.ray_workers_use_nsight = ray_workers_use_nsight

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

        self.disable_shared_memory = (not self.worker_use_ray
                                      or disable_shared_memory)

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")

        if self.ray_workers_use_nsight and not self.worker_use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        num_preallocated_slots_per_step: The number of slots the scheduler will
            allocate per model step, in addition to the slots allocated for
            every logical token. Defaults to 0.
        use_deltas: Whether scheduler output is emitted as a "delta" or update.
            Deltas are smaller and incur less overhead over IPC.
        max_chunked_prefill_len: The maximum length of tokens for prefill
            requests. Longer requests will be chunked into multiple chunks.
            -1 means no chunking (disabled). This features is only supported
            for flash style attention.
        max_num_prompt_seqs: The maximum number of prompt sequences to be
            processed in a single iteration.
        flash_style: Whether to use flash style attention. Only support
            LLaMA models.
        input_padding_size: The padding size for input tokens. This is used
            to better support CUDAGRAPH and ultize TENSOR CORES. Has to be
            a multiple of 8.
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
        num_preallocated_slots_per_step: int = 0,
        use_deltas: bool = False,
        max_chunked_prefill_len: int = -1,
        max_num_prompt_seqs: int = 1024,
        flash_style: bool = False,
        input_padding_size: int = 8,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)

        self.max_num_seqs = max_num_seqs
        self.max_num_decoding_tokens = max_num_seqs
        self.max_model_len = max_model_len
        self.num_preallocated_slots_per_step = num_preallocated_slots_per_step
        self.use_deltas = use_deltas
        # We pad the prompt and generation tokens with padding size 8
        # to better support CUDAGRAPH and ultize TENSOR CORES
        self.input_padding_size = input_padding_size
        self.max_chunked_prefill_len = max_chunked_prefill_len
        self.max_num_prompt_seqs = max_num_prompt_seqs
        self.flash_style = flash_style
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len and \
                self.max_chunked_prefill_len == -1:
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

        if self.max_num_batched_tokens < self.max_model_len:
            logger.warning(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This means that the user will not be able to use the full "
                "model context length.")

        if self.num_preallocated_slots_per_step < 0:
            raise ValueError(
                f"num_preallocated_slots_per_step"
                f"({self.num_preallocated_slots_per_step}) must be greater than"
                " or equal to 1.")

        if self.max_chunked_prefill_len >= 0 and not self.flash_style:
            raise ValueError(
                "chunked prefill is only supported for flash style")

        if self.input_padding_size % 8 != 0 or self.input_padding_size == 0:
            raise ValueError(
                f"input_padding_size ({self.input_padding_size}) must be a "
                "multiple of 8.")


class SpeculativeConfig:
    """Configuration for speculative decoding.

    Args:
        draft_model_config: ModelConfig for the draft model.
        draft_parallel_config: ParallelConfig for the draft model.
        num_speculative_tokens: The number of tokens to sample from the draft
            model before scoring with the target model.
    """

    @staticmethod
    def maybe_create_spec_config(
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
        dtype: str,
        speculative_model: Optional[str],
        num_speculative_tokens: Optional[int],
        speculative_model_uses_tp_1: bool,
        target_model_input_padding_size: Optional[int] = None,
        draft_model_input_padding_size: Optional[int] = None,
    ) -> Optional["SpeculativeConfig"]:
        """Create a SpeculativeConfig if all fields required are not None.
        """

        if (speculative_model is None and num_speculative_tokens is None
                and not speculative_model_uses_tp_1):
            return None

        if (speculative_model is None and num_speculative_tokens
                is not None) or (speculative_model is not None
                                 and num_speculative_tokens is None):
            raise ValueError(
                "Expected both speculative_model and "
                "num_speculative_tokens to be provided, but found "
                f"{speculative_model=} and {num_speculative_tokens=}.")

        # TODO these should be provided as a top-level draft model config.
        revision = None
        quantization = None
        max_model_len = None
        draft_model_config = ModelConfig(
            speculative_model, target_model_config.tokenizer,
            target_model_config.tokenizer_mode,
            target_model_config.trust_remote_code,
            target_model_config.download_dir, target_model_config.load_format,
            dtype, target_model_config.seed, revision,
            target_model_config.tokenizer_revision, max_model_len,
            quantization, target_model_config.enable_cuda_graph,
            target_model_config.cuda_graph_max_context_len,
            target_model_config.cuda_graph_cache_size)

        draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
            target_parallel_config, speculative_model_uses_tp_1)

        return SpeculativeConfig(
            draft_model_config,
            draft_parallel_config,
            num_speculative_tokens,
            target_model_input_padding_size,
            draft_model_input_padding_size,
        )

    @staticmethod
    def create_draft_parallel_config(
            target_parallel_config: ParallelConfig,
            speculative_model_uses_tp_1: bool) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.
        """
        tp_size = target_parallel_config.tensor_parallel_size

        if speculative_model_uses_tp_1:
            tp_size = 1

        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.
            pipeline_parallel_size,
            tensor_parallel_size=tp_size,
            worker_use_ray=target_parallel_config.worker_use_ray,
            disable_shared_memory=target_parallel_config.disable_shared_memory,
            num_tokenizer_actors=target_parallel_config.num_tokenizer_actors,
            tokenizer_actor_options=target_parallel_config.
            tokenizer_actor_options,
            ray_workers_use_nsight=target_parallel_config.
            ray_workers_use_nsight,
        )

        return draft_parallel_config

    def __init__(
        self,
        draft_model_config: ModelConfig,
        draft_parallel_config: ParallelConfig,
        num_speculative_tokens: int,
        target_model_input_padding_size: Optional[int],
        draft_model_input_padding_size: Optional[int],
    ):
        self.draft_model_config = draft_model_config
        self.draft_parallel_config = draft_parallel_config
        self.num_speculative_tokens = num_speculative_tokens
        self.target_model_input_padding_size = target_model_input_padding_size
        self.draft_model_input_padding_size = draft_model_input_padding_size

        self._verify_args()

    def _verify_args(self) -> None:
        if self.num_speculative_tokens < 0:
            raise ValueError("Expected num_speculative_tokens to be greater "
                             f"than zero ({self.num_speculative_tokens}).")

        self.draft_model_config.verify_with_parallel_config(
            self.draft_parallel_config)

    def create_target_scheduler_config(
        self,
        scheduler_config: SchedulerConfig,
    ) -> SchedulerConfig:
        """Create a SchedulerConfig for the target model.
        """
        config = copy.deepcopy(scheduler_config)
        # in the worst case, the target model has
        # batch_size * (num_speculative_tokens + 1) * 2 number of
        # tokens. we should increase max_num_decoding_token.
        config.max_num_decoding_tokens = config.max_num_decoding_tokens * 2 * (
            self.num_speculative_tokens + 1)
        if self.target_model_input_padding_size is not None:
            config.input_padding_size = self.target_model_input_padding_size
        return config

    def create_draft_scheduler_config(
        self,
        scheduler_config: SchedulerConfig,
    ) -> SchedulerConfig:
        """Create a SchedulerConfig for the draft model.
        """
        config = copy.deepcopy(scheduler_config)
        # in the worst case, the draft model has
        # batch_size * (num_speculative_tokens + 1) number of
        # tokens. we should increase max_num_decoding_token.
        config.max_num_decoding_tokens = config.max_num_decoding_tokens * (
            self.num_speculative_tokens + 1)
        if self.draft_model_input_padding_size is not None:
            config.input_padding_size = self.draft_model_input_padding_size
        return config

    @property
    def num_preallocated_slots_per_step(self) -> int:
        """The number of slots the scheduler should allocate per step, in
        addition to the slots allocated for each logical token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored.
        """
        return self.num_speculative_tokens


@dataclass
class LoRAConfig:
    max_lora_rank: int
    max_loras: int
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[torch.dtype] = None
    lora_extra_vocab_size: int = 256

    def __post_init__(self):
        # Keep this in sync with csrc/punica/bgmv/bgmv_config.h
        possible_max_ranks = (8, 16, 32, 64)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_num_seqs ({self.max_loras})")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.max_chunked_prefill_len > 0:
            raise ValueError("chunked prefill is not supported for lora")

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


@dataclass
class LoadConfig:
    s3_bucket: str
    s3_prefix: str
    region: str
    target_throughput_gbps: float = 100.0
    part_size = 4 * 1024 * 1024
    upload_if_not_exist: bool = True
