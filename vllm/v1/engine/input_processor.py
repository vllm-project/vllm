# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""输入处理器模块。

本模块实现了 vLLM V1 引擎的输入处理功能，负责：
- 将原始输入（提示词、ProcessorInputs）转换为 EngineCoreRequest
- 验证采样参数和池化参数
- 处理 LoRA 请求验证
- 处理多模态输入
- 分配唯一的请求 ID

主要类：
- InputProcessor: 输入处理器类
"""

import time
from collections.abc import Mapping
from typing import Any, Literal

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.inputs.data import (
    ProcessorInputs,
    PromptType,
    SingletonInputs,
)
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
)
from vllm.multimodal.utils import argsort_mm_positions
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.sampling_params import SamplingParams
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


class InputProcessor:
    """输入处理器类。

    负责将原始输入转换为 EngineCoreRequest，包括：
    - 输入验证
    - 参数验证
    - 多模态数据处理
    - 请求 ID 分配
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        """初始化输入处理器。

        Args:
            vllm_config: 全局配置
            renderer: 渲染器
            mm_registry: 多模态注册表
        """
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        self.observability_config = vllm_config.observability_config

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer or renderer_from_config(vllm_config)

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False
        if self.supports_mm_inputs:
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # 不再使用

        self.input_preprocessor = InputPreprocessor(
            vllm_config,
            renderer=renderer,
            mm_registry=mm_registry,
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        """返回分词器。"""
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        """获取分词器。

        Returns:
            分词器实例
        """
        return self.renderer.get_tokenizer()

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> None:
        """验证采样参数或池化参数的有效性。

        Args:
            params: 采样参数或池化参数
            supported_tasks: 支持的任务类型列表

        Raises:
            ValueError: 参数不支持或任务类型不匹配时抛出
            TypeError: 参数类型不正确时抛出
        """
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise ValueError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise ValueError("This model does not support pooling")

            if params.task is None:
                if "token_embed" in supported_pooling_tasks:
                    params.task = "token_embed"
                elif "token_classify" in supported_pooling_tasks:
                    params.task = "token_classify"
                elif "plugin" in supported_pooling_tasks:
                    params.task = "plugin"

            if params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        """验证 LoRA 请求的有效性。

        Args:
            lora_request: LoRA 请求

        Raises:
            ValueError: 当提供 LoRA 请求但 LoRA 未启用时抛出
        """
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    def _get_mm_identifier(
        self,
        mm_hash: str,
        lora_request: LoRARequest | None,
    ) -> str:
        """获取多模态标识符。

        当 enable_tower_connector_lora 为 True 时，多模态嵌入
        会随 LoRA 请求而变化。因此，mm_hash 必须基于
        LoRA 请求生成，以防止错误的缓存命中。

        Args:
            mm_hash: 多模态哈希值
            lora_request: LoRA 请求

        Returns:
            多模态标识符字符串
        """
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        return f"{lora_request.lora_name}:{mm_hash}"

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """分配请求 ID。

        将外部提供的请求 ID 替换为内部请求 ID，
        添加 8 个随机字符以确保唯一性。

        Args:
            request: 引擎核心请求

        Raises:
            ValueError: 当 external_req_id 字段已被设置时抛出
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | ProcessorInputs,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        """处理输入并生成 EngineCoreRequest。

        负责将原始输入转换为引擎核心请求，包括：
        - 参数验证
        - 多模态数据处理
        - 停止字符串处理
        - 请求 ID 分配

        Args:
            request_id: 请求 ID
            prompt: 提示词或处理器输入
            params: 采样参数或池化参数
            supported_tasks: 支持的任务类型列表
            arrival_time: 到达时间，默认为当前时间
            lora_request: LoRA 请求
            tokenization_kwargs: 分词参数字典（已废弃）
            trace_headers: 分布式追踪头信息
            priority: 请求优先级
            data_parallel_rank: 数据并行 rank
            resumable: 是否可恢复

        Returns:
            EngineCoreRequest: 引擎核心请求

        Raises:
            ValueError: 参数无效或数据并行 rank 超出范围时抛出
        """
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise ValueError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            if tokenization_kwargs:
                logger.warning_once(
                    "Passing tokenization_kwargs to InputProcessor is deprecated "
                    "and will be removed in v0.18. You should instead pass "
                    "them to Renderer.render_cmpl() or Renderer.render_chat()."
                )

            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            processed_inputs: ProcessorInputs = prompt  # type: ignore[assignment]
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            processed_inputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
            )

        current_platform.validate_request(processed_inputs, params)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_token_ids = None
            prompt_embeds = decoder_inputs["prompt_embeds"]
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
        )

    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        """验证提示词长度。

        Args:
            prompt_len: 提示词长度
            prompt_type: 提示词类型（"encoder" 或 "decoder"）

        Raises:
            ValueError: 提示词为空或超出最大长度时抛出
        """
        if self.skip_prompt_length_check:
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise ValueError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            if self.supports_mm_inputs:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        """验证模型输入的有效性。

        Args:
            prompt_inputs: 单例输入
            prompt_type: 提示词类型（"encoder" 或 "decoder"）

        Raises:
            ValueError: 提示词为空、超出最大长度、多模态嵌入超出缓存大小
                      或 token ID 超出词汇表范围时抛出
        """
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_ids = (
            None
            if prompt_inputs["type"] == "embeds"
            else prompt_inputs["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_inputs["prompt_embeds"]
            if prompt_inputs["type"] == "embeds"
            else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_inputs["type"] == "multimodal":
            decoder_mm_positions = prompt_inputs["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    embed_length = mm_position.get_num_embeds()
                    if embed_length > self.mm_encoder_cache_size:
                        raise ValueError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with length {embed_length}, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        if prompt_ids and tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)

            # NOTE: tokenizer.max_token_id is the tokenizer’s vocab size while
            # self.model_config.get_vocab_size() is the model’s vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            model_vocab_size = model_config.get_vocab_size()
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")

    def _validate_model_inputs(
        self,
        encoder_inputs: SingletonInputs | None,
        decoder_inputs: SingletonInputs,
    ):
        """验证模型输入的有效性。

        分别验证 encoder 和 decoder 输入。

        Args:
            encoder_inputs: Encoder 输入（可选）
            decoder_inputs: Decoder 输入
        """
        if encoder_inputs is not None:
            self._validate_model_input(encoder_inputs, prompt_type="encoder")

        self._validate_model_input(decoder_inputs, prompt_type="decoder")
