# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 的主入口模块 - LLM 类

本模块提供了 vLLM 的高级 API，用于离线批处理推理。
LLM 类封装了：
- Tokenizer（分词器）
- 语言模型（可能分布在多个 GPU 上）
- KV Cache（用于加速注意力计算的中间状态缓存）
- 智能批处理机制和高效的内存管理

主要功能：
1. 文本生成 (generate, chat)
2. Embedding 向量生成 (embed)
3. 分类任务 (classify)
4. 相似度打分 (score)
5. Reward 模型 (reward)
6. Beam Search 搜索
"""

import itertools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import TypeVar, overload

# Beam Search 相关导入
from vllm.beam_search import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)
# 配置相关导入
from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    PoolerConfig,
    ProfilerConfig,
    StructuredOutputsConfig,
    is_init_field,
)
from vllm.config.compilation import CompilationMode
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
# 分布式权重传输（用于 RL 训练）
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import EngineArgs
# Chat 工具
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateConfig,
    ChatTemplateContentFormatOption,
    load_chat_template,
)
# Pooling 任务相关（embedding、分类等）
from vllm.entrypoints.pooling.io_processor_factories import init_pooling_io_processors
from vllm.entrypoints.pooling.score.utils import (
    ScoreData,
    ScoreMultiModalParam,
    _cosine_similarity,
    compress_token_type_ids,
    compute_maxsim_score,
    get_score_prompt,
    score_data_to_prompts,
    validate_score_input,
)
from vllm.entrypoints.utils import log_non_default_args
# 输入数据类型
from vllm.inputs.data import (
    DataPrompt,
    ProcessorInputs,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.logger import init_logger
# LoRA 低秩适配
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
# 输出类型
from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
    RequestOutput,
    ScoringRequestOutput,
)
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    conversation_to_seq,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.tasks import PoolingTask
from vllm.tokenizers import TokenizerLike
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.tqdm_utils import maybe_tqdm
from vllm.v1.engine import PauseMode
# 核心引擎
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from vllm.v1.metrics.reader import Metric

logger = init_logger(__name__)

# 类型变量定义，用于泛型编程
# _O: 输出类型，可以是 RequestOutput 或 PoolingRequestOutput
_O = TypeVar(
    "_O",
    bound=RequestOutput | PoolingRequestOutput,
    default=RequestOutput | PoolingRequestOutput,
)
# _P: 参数类型，SamplingParams 或 PoolingParams
_P = TypeVar("_P", bound=SamplingParams | PoolingParams | None)
# _R: 通用返回类型
_R = TypeVar("_R", default=Any)


class LLM:
    """
    vLLM 的核心类，用于从给定的提示和采样参数生成文本。

    该类包含：
    - 一个 tokenizer（分词器）
    - 一个语言模型（可能分布在多个 GPU 上）
    - 为中间状态分配的 GPU 内存空间（即 KV cache）

    给定一批 prompts 和采样参数，该类使用智能批处理机制和高效的内存管理
    来生成文本。

    【使用示例】
    ```python
    from vllm import LLM, SamplingParams

    # 初始化 LLM
    llm = LLM(model="facebook/opt-125m")

    # 定义采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 生成文本
    prompts = ["Hello, my name is", "The capital of France is"]
    outputs = llm.generate(prompts, sampling_params)
    ```

    参数:
        model: HuggingFace Transformers 模型的名称或路径
        tokenizer: 分词器的名称或路径
        tokenizer_mode: 分词器模式，"auto" 使用快速分词器（如果可用），"slow" 总是使用慢速分词器
        skip_tokenizer_init: 如果为 True，跳过分词器和 detokenizer 的初始化
        trust_remote_code: 是否信任远程代码（来自 HuggingFace）
        tensor_parallel_size: 用于张量并行的 GPU 数量
        dtype: 模型权重和激活的数据类型，支持 float32、float16、bfloat16
        quantization: 量化方法，支持 "awq"、"gptq"、"fp8" 等
        gpu_memory_utilization: GPU 内存保留比例 (0-1)，用于模型权重、激活和 KV cache
        kv_cache_memory_bytes: 每个 GPU 的 KV cache 大小（字节），比 gpu_memory_utilization 更精细的控制
        cpu_offload_gb: 用于卸载模型权重的 CPU 内存大小 (GiB)
        seed: 随机数生成器的种子
        enforce_eager: 是否强制使用 eager 模式（禁用 CUDA graph）

    注意:
        该类设计用于离线推理。对于在线服务，请使用 AsyncLLMEngine 类。
    """

    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        LLM 构造函数。

        初始化流程：
        1. 处理废弃参数和特殊参数
        2. 将字典参数转换为配置对象
        3. 创建 EngineArgs 配置
        4. 初始化 LLMEngine（核心推理引擎）
        5. 初始化各种处理器和配置
        """

        # ========== 1. 处理废弃参数 ==========
        if "swap_space" in kwargs:
            kwargs.pop("swap_space")
            import warnings
            warnings.warn(
                "The 'swap_space' parameter is deprecated and ignored. "
                "It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        # 默认禁用日志统计
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        # ========== 2. 处理 worker 类（用于自定义 worker） ==========
        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # 如果 worker_cls 不是字符串形式的类名，使用 cloudpickle 序列化
            # 这样可以避免 pickle 问题
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        # ========== 3. 处理 KV 传输配置（用于分布式推理） ==========
        if "kv_transfer_config" in kwargs and isinstance(
            kwargs["kv_transfer_config"], dict
        ):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        # ========== 4. 辅助函数：将 dict/None/实例 转换为配置对象 ==========
        def _make_config(value: Any, cls: type[_R]) -> _R:
            """将 dict/None/实例 转换为配置类实例"""
            if value is None:
                return cls()
            if isinstance(value, dict):
                return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})
            return value

        # ========== 5. 处理各个配置对象 ==========
        if isinstance(compilation_config, int):
            # 如果是整数，作为编译模式
            compilation_config_instance = CompilationConfig(
                mode=CompilationMode(compilation_config)
            )
        else:
            compilation_config_instance = _make_config(
                compilation_config, CompilationConfig
            )

        structured_outputs_instance = _make_config(
            structured_outputs_config, StructuredOutputsConfig
        )
        profiler_config_instance = _make_config(profiler_config, ProfilerConfig)
        attention_config_instance = _make_config(attention_config, AttentionConfig)

        # ========== 6. 数据并行警告 ==========
        # 检查是否使用了单进程数据并行（不推荐）
        _dp_size = int(kwargs.get("data_parallel_size", 1))
        _distributed_executor_backend = kwargs.get("distributed_executor_backend")
        if (
            _dp_size > 1
            and not _distributed_executor_backend == "external_launcher"
            and not current_platform.is_tpu()
        ):
            raise ValueError(
                f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
                "process usage and may hang. Please use "
                "the explicit multi-process data-parallel example at "
                "'examples/offline_inference/data_parallel.py'."
            )

        # ========== 7. 创建 EngineArgs ==========
        # EngineArgs 包含了引擎运行所需的所有配置参数
        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            cpu_offload_gb=cpu_offload_gb,
            offload_group_size=offload_group_size,
            offload_num_in_group=offload_num_in_group,
            offload_prefetch_step=offload_prefetch_step,
            offload_params=offload_params or set(),
            enforce_eager=enforce_eager,
            enable_return_routed_experts=enable_return_routed_experts,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            structured_outputs_config=structured_outputs_instance,
            profiler_config=profiler_config_instance,
            attention_config=attention_config_instance,
            compilation_config=compilation_config_instance,
            logits_processors=logits_processors,
            **kwargs,
        )

        # 记录非默认参数（用于调试和日志）
        log_non_default_args(engine_args)

        # ========== 8. 初始化核心引擎 LLMEngine ==========
        # 这是整个 LLM 类的核心，所有的推理工作都由 LLMEngine 完成
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.engine_class = type(self.llm_engine)

        # ========== 9. 初始化请求计数器 ==========
        # 用于生成唯一的请求 ID
        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        # ========== 10. 获取支持的任务类型 ==========
        # 例如："generate"（文本生成）、"embed"（嵌入）、"classify"（分类）等
        supported_tasks = self.llm_engine.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)
        self.supported_tasks = supported_tasks

        # ========== 11. 缓存常用配置和处理器 ==========
        self.model_config = self.llm_engine.model_config
        self.renderer = self.llm_engine.renderer
        self.chat_template = load_chat_template(chat_template)
        self.io_processor = self.llm_engine.io_processor
        self.input_processor = self.llm_engine.input_processor
        self.chat_template_config = ChatTemplateConfig(chat_template=self.chat_template)

        # 初始化 pooling 任务的 IO 处理器（用于 embedding、分类等任务）
        self.pooling_io_processors = init_pooling_io_processors(
            supported_tasks=supported_tasks,
            model_config=self.model_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )

        # 缓存 __repr__ 结果，避免重复调用 collective_rpc
        self._cached_repr: str | None = None

    def get_tokenizer(self) -> TokenizerLike:
        """获取分词器"""
        return self.llm_engine.get_tokenizer()

    def get_world_size(self, include_dp: bool = True) -> int:
        """
        获取并行世界大小（GPU 总数）。

        参数:
            include_dp: 如果为 True（默认），返回包括数据并行的世界大小 (TP * PP * DP)
                       如果为 False，返回不包括数据并行的世界大小 (TP * PP)

        返回:
            世界大小（GPU 总数）
        """
        parallel_config = self.llm_engine.vllm_config.parallel_config
        if include_dp:
            return parallel_config.world_size_across_dp
        return parallel_config.world_size

    def reset_mm_cache(self) -> None:
        """重置多模态缓存"""
        self.renderer.clear_mm_cache()
        self.llm_engine.reset_mm_cache()

    def get_default_sampling_params(self) -> SamplingParams:
        """
        获取模型默认的采样参数。

        某些模型在 generation_config.json 中定义了默认的采样参数，
        该方法会返回这些默认参数。
        """
        if self.default_sampling_params is None:
            self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:
        """
        为输入提示生成补全文本。

        这是最常用的离线推理方法。该类会自动批处理给定的提示，
        考虑内存限制。为了最佳性能，建议将所有提示放入一个列表中传递。

        【使用示例】
        ```python
        from vllm import LLM, SamplingParams

        llm = LLM(model="facebook/opt-125m")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        prompts = ["Hello, my name is", "The capital of France is"]
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
        ```

        参数:
            prompts: 提示列表，可以是字符串或 token IDs。
                    支持批量推理，详见 [PromptType][vllm.inputs.PromptType]
            sampling_params: 采样参数。如果为 None，使用默认采样参数。
                           如果是单个值，应用于所有提示；如果是列表，必须与 prompts 长度相同
            use_tqdm: 如果为 True，显示 tqdm 进度条
            lora_request: LoRA 请求（如果使用 LoRA 微调）
            priority: 请求优先级列表（仅在启用优先级调度时使用）
            tokenization_kwargs: tokenizer.encode 的覆盖参数

        返回:
            RequestOutput 对象列表，包含生成的文本，顺序与输入提示相同
        """
        # 检查模型类型是否为生成式模型
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        # 如果未提供采样参数，使用默认参数
        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        # 调用内部完成推理方法
        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

    def enqueue(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        将提示加入生成队列，但不等待完成。

        该方法将请求添加到引擎队列，但不会立即处理它们。
        需要调用 wait_for_completion() 来获取结果。

        适用于需要异步处理的场景。

        参数:
            prompts: 提示列表，见 generate()
            sampling_params: 采样参数
            lora_request: LoRA 请求
            priority: 请求优先级
            use_tqdm: 是否显示进度条
            tokenization_kwargs: tokenizer.encode 的覆盖参数

        返回:
            已加入请求的 ID 列表
        """
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError("LLM.enqueue() is only supported for generative models.")

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._add_completion_requests(
            prompts=prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        )

    @overload
    def wait_for_completion(
        self,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[RequestOutput | PoolingRequestOutput]: ...

    @overload
    def wait_for_completion(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]: ...

    def wait_for_completion(
        self,
        output_type: type[Any] | tuple[type[Any], ...] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[Any]:
        """
        等待所有已加入队列的请求完成并返回结果。

        该方法处理引擎队列中的所有请求并返回它们的输出。
        在 enqueue() 后调用此方法获取结果。

        参数:
            output_type: 期望的输出类型，默认为 RequestOutput
            use_tqdm: 是否显示进度条

        返回:
            所有已完成请求的输出对象列表
        """
        if output_type is None:
            output_type = (RequestOutput, PoolingRequestOutput)

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def _resolve_mm_lora(
        self,
        prompt: ProcessorInputs,
        lora_request: LoRARequest | None,
    ) -> LoRARequest | None:
        if prompt["type"] != "multimodal":
            return lora_request

        lora_config = self.llm_engine.vllm_config.lora_config
        default_mm_loras = None if lora_config is None else lora_config.default_mm_loras
        if not default_mm_loras:
            return lora_request

        prompt_modalities = prompt["mm_placeholders"].keys()
        intersection = set(prompt_modalities).intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request

        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be "
                "used by a single prompt consuming several modalities; "
                "currently we only support one lora per request; as such, "
                "lora(s) registered with modalities: %s will be skipped",
                intersection,
            )
            return lora_request

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        modality_name = intersection.pop()
        modality_lora_path = default_mm_loras[modality_name]
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        if lora_request:
            if lora_request.lora_int_id != modality_lora_id:
                logger.warning(
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """

        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.

        !!! warning
            To reduce the overhead of data transfer, avoid returning large
            arrays or tensors from this method. If you must return them,
            make sure you move them to CPU first to avoid taking up additional
            VRAM!
        """
        return self.llm_engine.apply_model(func)

    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],
        params: BeamSearchParams,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        tokenizer = self.renderer.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        engine_prompts = self._preprocess_cmpl(prompts)
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_prompts))

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar."
            )
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(engine_prompts)

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        sampling_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
            skip_clone=True,  # Internal beam search, safe to skip clone
        )
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, engine_prompts):
            if prompt["type"] == "embeds":
                raise NotImplementedError(
                    "Embedding prompt not supported for beam search"
                )

            instances.append(
                BeamSearchInstance(
                    prompt,
                    lora_request=lora_req,
                    logprobs=None,
                ),
            )

        for prompt_start in range(0, len(instances), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]

            token_iter = range(max_tokens)
            if use_tqdm:
                token_iter = tqdm(
                    token_iter, desc="Beam search", unit="token", unit_scale=False
                )
                logger.warning(
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress."
                )
            for _ in token_iter:
                all_beams: list[BeamSearchSequence] = list(
                    sum((instance.beams for instance in instances_batch), [])
                )
                pos = [0] + list(
                    itertools.accumulate(
                        len(instance.beams) for instance in instances_batch
                    )
                )
                instance_start_and_end: list[tuple[int, int]] = list(
                    zip(pos[:-1], pos[1:])
                )

                if len(all_beams) == 0:
                    break

                # only runs for one step
                # we don't need to use tqdm here
                output = self._render_and_run_requests(
                    prompts=(beam.get_prompt() for beam in all_beams),
                    params=self._params_to_seq(sampling_params, len(all_beams)),
                    output_type=RequestOutput,
                    lora_requests=[beam.lora_request for beam in all_beams],
                    use_tqdm=False,
                )

                for (start, end), instance in zip(
                    instance_start_and_end, instances_batch
                ):
                    instance_new_beams = []
                    for i in range(start, end):
                        current_beam = all_beams[i]
                        result = output[i]

                        if result.outputs[0].logprobs is not None:
                            # if `result.outputs[0].logprobs` is None, it means
                            # the sequence is completed because of the
                            # max-model-len or abortion. we don't need to add
                            # it to the new beams.
                            logprobs = result.outputs[0].logprobs[0]
                            for token_id, logprob_obj in logprobs.items():
                                new_beam = BeamSearchSequence(
                                    current_beam.orig_prompt,
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs + [logprobs],
                                    lora_request=current_beam.lora_request,
                                    cum_logprob=current_beam.cum_logprob
                                    + logprob_obj.logprob,
                                )

                                if token_id == eos_token_id and not ignore_eos:
                                    instance.completed.append(new_beam)
                                else:
                                    instance_new_beams.append(new_beam)
                    sorted_beams = sorted(
                        instance_new_beams, key=sort_beams_key, reverse=True
                    )
                    instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(
                instance.completed, key=sort_beams_key, reverse=True
            )
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)

            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def _preprocess_cmpl(
        self,
        prompts: Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[ProcessorInputs]:
        """
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `ProcessorInputs` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            parse_model_prompt(model_config, prompt) for prompt in prompts
        ]
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        return renderer.render_cmpl(parsed_prompts, tok_params)

    def _preprocess_cmpl_one(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> ProcessorInputs:
        (engine_prompt,) = self._preprocess_cmpl([prompt], tokenization_kwargs)
        return engine_prompt

    def _preprocess_chat(
        self,
        conversations: Sequence[list[ChatCompletionMessageParam]],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[ProcessorInputs]:
        """
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `ProcessorInputs` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer

        chat_params = ChatParams(
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                chat_template_kwargs,
                dict(
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenize=is_mistral_tokenizer(renderer.tokenizer),
                ),
            ),
        )
        tok_params = renderer.default_chat_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        _, engine_prompts = renderer.render_chat(
            conversations,
            chat_params,
            tok_params,
            prompt_extras={"mm_processor_kwargs": mm_processor_kwargs},
        )

        return engine_prompts

    def _preprocess_chat_one(
        self,
        conversation: list[ChatCompletionMessageParam],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> ProcessorInputs:
        (engine_prompt,) = self._preprocess_chat(
            [conversation],
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return engine_prompt

    def chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A sequence of conversations or a single conversation.

                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                  Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                  similar to OpenAI schema.
                  Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_chat(
            messages=messages,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def encode(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        pooling_task: PoolingTask | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_task: Override the pooling task to use.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """

        if pooling_task is None:
            raise ValueError(
                "pooling_task required for `LLM.encode`\n"
                "Please use one of the more specific methods or set the "
                "pooling_task when using `LLM.encode`:\n"
                "  - For embeddings, use `LLM.embed(...)` "
                'or `pooling_task="embed"`.\n'
                "  - For classification logits, use `LLM.classify(...)` "
                'or `pooling_task="classify"`.\n'
                "  - For similarity scores, use `LLM.score(...)`.\n"
                "  - For rewards, use `LLM.reward(...)` "
                'or `pooling_task="token_classify"`\n'
                "  - For token classification, "
                'use `pooling_task="token_classify"`\n'
                '  - For multi-vector retrieval, use `pooling_task="token_embed"`'
            )

        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "pooling":
            raise ValueError(
                "LLM.encode() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        if isinstance(prompts, dict) and "data" in prompts:
            if self.io_processor is None:
                raise ValueError(
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details."
                )

            # Validate the request data is valid for the loaded plugin
            prompt_data = prompts.get("data")
            if prompt_data is None:
                raise ValueError(
                    "The 'data' field of the prompt is expected to contain "
                    "the prompt data and it cannot be None. "
                    "Refer to the documentation of the IOProcessor "
                    "in use for more details."
                )
            validated_prompt = self.io_processor.parse_data(prompt_data)

            # obtain the actual model prompts from the pre-processor
            prompts = self.io_processor.pre_process(prompt=validated_prompt)
            prompts_seq = prompt_to_seq(prompts)

            params_seq: Sequence[PoolingParams] = [
                self.io_processor.merge_pooling_params(param)
                for param in self._params_to_seq(
                    pooling_params,
                    len(prompts_seq),
                )
            ]
            for p in params_seq:
                if p.task is None:
                    p.task = "plugin"

            outputs = self._run_completion(
                prompts=prompts_seq,
                params=params_seq,
                output_type=PoolingRequestOutput,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
            )

            # get the post-processed model outputs
            assert self.io_processor is not None
            processed_outputs = self.io_processor.post_process(outputs)

            return [
                PoolingRequestOutput[Any](
                    request_id="",
                    outputs=processed_outputs,
                    num_cached_tokens=getattr(
                        processed_outputs, "num_cached_tokens", 0
                    ),
                    prompt_token_ids=[],
                    finished=True,
                )
            ]
        else:
            if pooling_params is None:
                # Use default pooling params.
                pooling_params = PoolingParams()

            prompts_seq = prompt_to_seq(prompts)
            params_seq = self._params_to_seq(pooling_params, len(prompts_seq))

            for param in params_seq:
                if param.task is None:
                    param.task = pooling_task
                elif param.task != pooling_task:
                    msg = (
                        f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                    )
                    raise ValueError(msg)

            if pooling_task in self.pooling_io_processors:
                io_processor = self.pooling_io_processors[pooling_task]
                processor_inputs = io_processor.pre_process_offline(
                    prompts_seq, tokenization_kwargs
                )
                seq_lora_requests = self._lora_request_to_seq(
                    lora_request, len(prompts_seq)
                )
                seq_priority = self._priority_to_seq(None, len(prompts))

                self._render_and_add_requests(
                    prompts=processor_inputs,
                    params=params_seq,
                    lora_requests=seq_lora_requests,
                    priorities=seq_priority,
                )

                outputs = self._run_engine(
                    use_tqdm=use_tqdm, output_type=PoolingRequestOutput
                )
                outputs = io_processor.post_process_offline(outputs)
            else:
                outputs = self._run_completion(
                    prompts=prompts_seq,
                    params=params_seq,
                    output_type=PoolingRequestOutput,
                    use_tqdm=use_tqdm,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                )
        return outputs

    def embed(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "embed" not in self.supported_tasks:
            raise ValueError(
                "Embedding API is not supported by this model. "
                "Try converting the model using `--convert embed`."
            )

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "classify" not in self.supported_tasks:
            raise ValueError(
                "Classification API is not supported by this model. "
                "Try converting the model using `--convert classify`."
            )

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="classify",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def reward(
        self,
        prompts: PromptType | Sequence[PromptType],
        /,
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        """
        Generate rewards for each prompt.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """
        return self.encode(
            prompts,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="token_classify",
            tokenization_kwargs=tokenization_kwargs,
        )

    def _embedding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        *,
        use_tqdm: bool | Callable[..., tqdm],
        pooling_params: PoolingParams | None,
        lora_request: list[LoRARequest] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any],
    ) -> list[ScoringRequestOutput]:
        tokenizer = self.get_tokenizer()

        input_texts: list[str] = []
        for text in data_1 + data_2:
            if not isinstance(text, str):
                raise NotImplementedError(
                    "Embedding scores currently do not support multimodal input."
                )
            input_texts.append(text)

        encoded_output = self.encode(
            input_texts,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        encoded_output_1 = encoded_output[0 : len(data_1)]
        encoded_output_2 = encoded_output[len(data_1) :]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        scores = _cosine_similarity(
            tokenizer=tokenizer,
            embed_1=encoded_output_1,
            embed_2=encoded_output_2,
        )

        return [ScoringRequestOutput.from_base(item) for item in scores]

    def _late_interaction_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        *,
        use_tqdm: bool | Callable[..., tqdm],
        pooling_params: PoolingParams | None,
        lora_request: list[LoRARequest] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any],
    ) -> list[ScoringRequestOutput]:
        """
        Late interaction scoring (ColBERT MaxSim).

        Encodes queries and documents into per-token embeddings, then computes
        MaxSim: sum over query tokens of max similarity to any document token.
        """
        from vllm.outputs import PoolingOutput

        tokenizer = self.get_tokenizer()

        # Convert ScoreData to PromptType (handles both text and multimodal)
        model_config = self.model_config
        prompts_1 = score_data_to_prompts(data_1, "query", model_config)
        prompts_2 = score_data_to_prompts(data_2, "document", model_config)

        encoded_output: list[PoolingRequestOutput] = self.encode(
            prompts_1 + prompts_2,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="token_embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        encoded_output_1: list[PoolingRequestOutput] = encoded_output[: len(prompts_1)]
        encoded_output_2: list[PoolingRequestOutput] = encoded_output[len(prompts_1) :]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        # Compute MaxSim scores
        scores: list[PoolingRequestOutput] = []
        padding: list[int] = []
        if (pad_token_id := tokenizer.pad_token_id) is not None:
            padding = [pad_token_id]

        for emb_1, emb_2 in zip(encoded_output_1, encoded_output_2):
            # emb_1.outputs.data: [query_len, dim]
            # emb_2.outputs.data: [doc_len, dim]
            q_emb = emb_1.outputs.data
            d_emb = emb_2.outputs.data

            maxsim_score = compute_maxsim_score(q_emb, d_emb)

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

            scores.append(
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                    outputs=PoolingOutput(data=maxsim_score),
                    prompt_token_ids=tokens,
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                    finished=True,
                )
            )

        return [ScoringRequestOutput.from_base(item) for item in scores]

    def _cross_encoding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        *,
        use_tqdm: bool | Callable[..., tqdm],
        pooling_params: PoolingParams | None,
        lora_request: list[LoRARequest] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any],
        score_template: str | None,
    ) -> list[ScoringRequestOutput]:
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        if is_mistral_tokenizer(tokenizer):
            raise ValueError("Score API is not supported for Mistral tokenizer")

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        if pooling_params is None:
            pooling_params = PoolingParams(task="classify")
        elif pooling_params.task is None:
            pooling_params.task = "classify"

        pooling_params_list = list[PoolingParams]()

        prompts = list[PromptType]()

        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]

        for q, d in input_pairs:
            _, engine_prompt = get_score_prompt(
                model_config=model_config,
                data_1=q,
                data_2=d,
                tokenizer=tokenizer,
                tokenization_kwargs=tokenization_kwargs,
                score_template=score_template,
            )

            if token_type_ids := engine_prompt.pop("token_type_ids", None):
                params = pooling_params.clone()
                compressed = compress_token_type_ids(token_type_ids)
                params.extra_kwargs = {"compressed_token_type_ids": compressed}
                pooling_params_list.append(params)
            else:
                pooling_params_list.append(pooling_params)

            prompts.append(engine_prompt)

        outputs = self._run_completion(
            prompts=prompts,
            params=pooling_params_list,
            output_type=PoolingRequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

        return [ScoringRequestOutput.from_base(item) for item in outputs]

    def score(
        self,
        data_1: SingletonPrompt
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        data_2: SingletonPrompt
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        /,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        chat_template: str | None = None,
    ) -> list[ScoringRequestOutput]:
        """Generate similarity scores for all pairs `<text,text_pair>` or
          `<multi-modal data, multi-modal data pair>`.

        The inputs can be `1 -> 1`, `1 -> N` or `N -> N`.
        In the `1 - N` case the `data_1` input will be replicated `N`
        times to pair with the `data_2` inputs.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your inputs into a single list and pass it to this method.

        Supports both text and multi-modal data (images, etc.) when used with
        appropriate multi-modal models. For multi-modal inputs, ensure the
        prompt structure matches the model's expected input format.

        Args:
            data_1: Can be a single prompt, a list of prompts or
                `ScoreMultiModalParam`, which can contain either text or
                multi-modal data. When a list, it must have the same length as
                the `data_2` list.
            data_2: The data to pair with the query to form the input to
                the LLM. Can be text or multi-modal data. See [PromptType]
                [vllm.inputs.PromptType] for more details about the format of
                each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The chat template to use for the scoring. If None, we
                use the model's default chat template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        """
        model_config = self.model_config

        runner_type = model_config.runner_type
        if runner_type != "pooling":
            raise ValueError(
                "LLM.score() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        supported_tasks = self.supported_tasks
        score_type = self.model_config.score_type
        is_late_interaction = score_type == "late-interaction"
        is_cross_encoder = score_type == "cross-encoder"

        # Late interaction models (e.g., ColBERT) use token_embed for scoring
        if not is_late_interaction and all(
            t not in supported_tasks for t in ("embed", "classify")
        ):
            raise ValueError(
                "Score API is not supported by this model. "
                "Try converting the model using "
                "`--convert embed` or `--convert classify`."
            )

        if is_cross_encoder and getattr(model_config.hf_config, "num_labels", 0) != 1:
            raise ValueError("Score API is only enabled for num_labels == 1.")

        if not is_cross_encoder and chat_template is not None:
            raise ValueError(
                "chat_template is only supported for cross-encoder models."
            )

        is_multimodal_model = model_config.is_multimodal_model
        architecture = model_config.architecture

        score_data_1, score_data_2 = validate_score_input(
            data_1,  # type: ignore[arg-type]
            data_2,  # type: ignore[arg-type]
            is_multimodal_model=is_multimodal_model,
            architecture=architecture,
        )

        renderer = self.renderer
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )
        encode_kwargs = tok_params.get_encode_kwargs()

        if is_cross_encoder:
            return self._cross_encoding_score(
                score_data_1,
                score_data_2,
                use_tqdm=use_tqdm,
                pooling_params=pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=encode_kwargs,
                score_template=chat_template,
            )
        elif is_late_interaction:
            return self._late_interaction_score(
                score_data_1,
                score_data_2,
                use_tqdm=use_tqdm,
                pooling_params=pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=encode_kwargs,
            )
        else:
            return self._embedding_score(
                score_data_1,
                score_data_2,
                use_tqdm=use_tqdm,
                pooling_params=pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=encode_kwargs,
            )

    def start_profile(self, profile_prefix: str | None = None) -> None:
        """Start profiling with optional custom trace prefix.

        Args:
            profile_prefix: Optional prefix for the trace file names. If provided,
                           trace files will be named as "<prefix>_dp<X>_pp<Y>_tp<Z>".
                           If not provided, default naming will be used.
        """
        self.llm_engine.start_profile(profile_prefix)

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.llm_engine.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level.
                - Level 0: Pause scheduling but continue accepting requests.
                           Requests are queued but not processed.
                - Level 1: Offload model weights to CPU, discard KV cache.
                           The content of kv cache is forgotten. Good for
                           sleeping and waking up the engine to run the same
                           model again. Please make sure there's enough CPU
                           memory to store the model weights.
                - Level 2: Discard all GPU memory (weights + KV cache).
                           Good for sleeping and waking up the engine to run
                           a different model or update the model, where
                           previous model weights are not needed. It reduces
                           CPU memory pressure.
            mode: How to handle any existing requests, can be "abort", "wait",
                or "keep".
        """
        self.llm_engine.sleep(level=level, mode=mode)

    def wake_up(self, tags: list[str] | None = None):
        """
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache", "scheduling")`. If None, all memory
                is reallocated. wake_up should be called with all tags
                (or None) before the engine is used again.
                Use tags=["scheduling"] to resume from level 0 sleep.
        """
        self.llm_engine.wake_up(tags)

    def get_metrics(self) -> list["Metric"]:
        """Return a snapshot of aggregated metrics from Prometheus.

        Returns:
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        return self.llm_engine.get_metrics()

    # ========== 辅助方法：参数序列化处理 ==========

    def _params_to_seq(
        self,
        params: _P | Sequence[_P],
        num_requests: int,
    ) -> Sequence[_P]:
        """
        将参数转换为序列格式。

        如果参数是单个值，复制 num_requests 份；
        如果参数已是序列，验证长度是否匹配。

        参数:
            params: 单个参数或参数序列
            num_requests: 请求数量

        返回:
            参数序列
        """
        if isinstance(params, Sequence):
            if len(params) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({params}) "
                    f"and params ({len(params)}) must be the same."
                )
            return params
        # 单个参数广播到所有请求
        return [params] * num_requests

    def _lora_request_to_seq(
        self,
        lora_request: LoRARequest | None | Sequence[LoRARequest | None],
        num_requests: int,
    ) -> Sequence[LoRARequest | None]:
        """
        将 LoRA 请求转换为序列格式。

        参数:
            lora_request: 单个 LoRA 请求或请求序列
            num_requests: 请求数量

        返回:
            LoRA 请求序列
        """
        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )
            return lora_request
        return [lora_request] * num_requests

    def _priority_to_seq(
        self,
        priority: list[int] | None,
        num_requests: int,
    ) -> Sequence[int]:
        """
        将优先级转换为序列格式。

        参数:
            priority: 优先级列表或 None
            num_requests: 请求数量

        返回:
            优先级序列，默认为全 0
        """
        if priority is not None:
            if len(priority) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )
            return priority
        # 默认优先级为 0
        return [0] * num_requests

    # ========== 内部方法：完成请求处理流程 ==========

    def _add_completion_requests(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        添加完成请求到引擎队列。

        流程：
        1. 将 prompts 转换为序列格式
        2. 将参数、LoRA 请求、优先级转换为序列格式
        3. 预处理每个 prompt（分词、应用模板等）
        4. 调用 _render_and_add_requests 添加到引擎

        参数:
            prompts: 原始提示
            params: 采样参数或 pooling 参数
            use_tqdm: 是否显示进度条
            lora_request: LoRA 请求
            priority: 优先级
            tokenization_kwargs: tokenizer.encode 的覆盖参数

        返回:
            已添加请求的 ID 列表
        """
        # 转换为序列格式
        seq_prompts = prompt_to_seq(prompts)
        seq_params = self._params_to_seq(params, len(seq_prompts))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))
        seq_priority = self._priority_to_seq(priority, len(prompts))

        # 渲染并添加请求
        return self._render_and_add_requests(
            prompts=(
                # 预处理每个 prompt（分词等）
                self._preprocess_cmpl_one(prompt, tokenization_kwargs)
                for prompt in maybe_tqdm(
                    seq_prompts,
                    use_tqdm=use_tqdm,
                    desc="Rendering prompts",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

    def _run_completion(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ):
        """
        运行完整的完成推理流程。

        这是 generate() 方法调用的核心内部方法。

        流程：
        1. 调用 _add_completion_requests 将请求加入队列
        2. 调用 _run_engine 执行推理直到完成

        参数:
            prompts: 原始提示
            params: 采样参数或 pooling 参数
            output_type: 输出类型
            use_tqdm: 是否显示进度条
            lora_request: LoRA 请求
            priority: 优先级
            tokenization_kwargs: tokenizer.encode 的覆盖参数

        返回:
            推理输出列表
        """
        # 添加请求到队列
        self._add_completion_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        )
        # 运行引擎直到所有请求完成
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)

    def _run_chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        seq_convs = conversation_to_seq(messages)
        seq_params = self._params_to_seq(params, len(seq_convs))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_convs))

        return self._render_and_run_requests(
            prompts=(
                self._preprocess_chat_one(
                    conversation,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format,
                    chat_template_kwargs=chat_template_kwargs,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenization_kwargs=tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for conversation in maybe_tqdm(
                    seq_convs,
                    use_tqdm=use_tqdm,
                    desc="Rendering conversations",
                )
            ),
            params=seq_params,
            output_type=output_type,
            lora_requests=seq_lora_requests,
            use_tqdm=use_tqdm,
        )

    # ========== 内部方法：渲染和添加请求 ==========

    def _render_and_run_requests(
        self,
        prompts: Iterable[ProcessorInputs],
        params: Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ):
        """
        渲染提示并运行请求。

        流程：
        1. 渲染提示（应用模板、分词等）
        2. 调用 _render_and_add_requests 添加到引擎
        3. 调用 _run_engine 执行推理

        参数:
            prompts: 已预处理的提示迭代器
            params: 参数序列
            output_type: 输出类型
            lora_requests: LoRA 请求序列
            priorities: 优先级序列
            use_tqdm: 是否显示进度条

        返回:
            推理输出列表
        """
        # 如果 prompts 是列表或元组，发出警告
        # 因为这样会降低效率（无法边渲染边执行）
        if isinstance(prompts, (list, tuple)):
            logger.warning_once(
                "Rendering all prompts before adding them to the engine "
                "is less efficient than performing both on the same prompt "
                "before processing the next prompt. You should instead pass "
                "a generator that renders one prompt per iteration, as that allows "
                "engine execution to begin for the first prompt while processing "
                "the next prompt."
            )

        # 添加请求到引擎
        self._render_and_add_requests(
            prompts=prompts,
            params=params,
            lora_requests=lora_requests,
            priorities=priorities,
        )

        # 运行引擎直到完成
        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def _render_and_add_requests(
        self,
        prompts: Iterable[ProcessorInputs],
        params: Sequence[SamplingParams | PoolingParams],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
    ) -> list[str]:
        """
        渲染提示并添加到引擎队列。

        该方法遍历所有提示，逐个添加到引擎队列。
        如果中途出错，会中止已添加的所有请求。

        参数:
            prompts: 已预处理的提示迭代器
            params: 参数序列
            lora_requests: LoRA 请求序列
            priorities: 优先级序列

        返回:
            已添加请求的 ID 列表
        """
        added_request_ids: list[str] = []

        try:
            # 逐个添加请求
            for i, prompt in enumerate(prompts):
                request_id = self._add_request(
                    prompt,
                    params[i],
                    lora_request=self._resolve_mm_lora(
                        prompt,
                        None if lora_requests is None else lora_requests[i],
                    ),
                    priority=0 if priorities is None else priorities[i],
                )
                added_request_ids.append(request_id)
        except Exception as e:
            # 如果出错，中止已添加的所有请求
            if added_request_ids:
                self.llm_engine.abort_request(added_request_ids, internal=True)
            raise e

        return added_request_ids

    def _add_request(
        self,
        prompt: ProcessorInputs,
        params: SamplingParams | PoolingParams,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
    ) -> str:
        """
        添加单个请求到引擎。

        这是将请求提交给 LLMEngine 的核心方法。

        流程：
        1. 如果是采样参数，设置输出类型为 FINAL_ONLY（只关心最终输出）
        2. 生成唯一的请求 ID
        3. 调用 LLMEngine.add_request 将请求加入引擎队列

        参数:
            prompt: 处理后的提示（ProcessorInputs 格式）
            params: 采样参数或 pooling 参数
            lora_request: LoRA 请求
            priority: 请求优先级

        返回:
            请求 ID 字符串
        """
        if isinstance(params, SamplingParams):
            # 我们只关心最终输出，不需要中间结果
            # 这样可以减少内存占用和通信开销
            params.output_kind = RequestOutputKind.FINAL_ONLY

        # 生成唯一的请求 ID
        request_id = str(next(self.request_counter))

        # 将请求添加到引擎队列
        # LLMEngine 负责调度、批处理和执行
        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            priority=priority,
        )

    def _run_engine(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]:
        """
        运行引擎直到所有请求完成。

        这是推理执行的核心循环。它不断调用 engine.step() 来处理请求，
        直到所有请求都完成。

        流程：
        1. 初始化 tqdm 进度条（如果启用）
        2. 循环调用 engine.step() 执行推理
        3. 收集已完成的请求输出
        4. 更新进度条和速度统计
        5. 按请求 ID 排序输出（因为有些请求可能提前完成）

        参数:
            output_type: 期望的输出类型
            use_tqdm: 是否显示进度条

        返回:
            按请求 ID 排序的输出列表
        """
        # ========== 1. 初始化 tqdm 进度条 ==========
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # ========== 2. 运行引擎主循环 ==========
        outputs: list[_O] = []
        total_in_toks = 0  # 累计输入 token 数
        total_out_toks = 0  # 累计输出 token 数

        # 主循环：直到所有请求都完成
        while self.llm_engine.has_unfinished_requests():
            # step() 执行一次调度迭代
            # 它会根据可用资源调度请求，执行模型前向传播，返回输出
            step_outputs = self.llm_engine.step()

            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    # 请求已完成，添加到输出列表
                    outputs.append(output)  # type: ignore[arg-type]

                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # 只计算 RequestOutput 的 token 数
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            # 累计输入 token（prompt token * 输出数量）
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            # 累计输出 token
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            # 更新速度显示
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        # ========== 3. 清理和排序 ==========
        if use_tqdm:
            pbar.close()

        # 按请求 ID 排序输出
        # 因为有些请求可能比前面的请求先完成，需要恢复原始顺序
        return sorted(outputs, key=lambda x: int(x.request_id))

    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """
        Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info
        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    def __repr__(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        # Cache the result to avoid repeated collective_rpc calls
        if self._cached_repr is None:
            results = self.llm_engine.collective_rpc("get_model_inspection")
            # In distributed settings, we get results from all workers
            # Just return the first one (they should all be the same)
            if results:
                self._cached_repr = results[0]
            else:
                self._cached_repr = f"LLM(model={self.model_config.model!r})"
        return self._cached_repr
