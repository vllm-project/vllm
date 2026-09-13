# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CN] 文件总览：默认权重加载器 DefaultModelLoader
# [CN] 职责：把磁盘/HF 上的一堆权重文件，变成一个统一的 (name, tensor) 迭代器，
# [CN]       并负责「哪些文件该读、用哪种 reader 读」的全部判断。
# [CN] 链路：load_weights() -> get_all_weights() -> _get_weights_iterator()
# [CN]       -> _prepare_weights() 选文件 -> 具体 weights_iterator 逐个产出
# [CN]       -> model.load_weights(iterator) 由模型自己按名字装进各层
# [CN] 支持的文件：safetensors(.safetensors) / bin(.bin) / pt(.pt) / npcache / mistral
# [CN]              以及三种加速变体：fastsafetensors、instanttensor、多线程 pt/safetensors
# [CN] 设计要点：
# [CN]   1. 迭代器而非一次性读入 —— 大模型权重远超内存，必须流式；
# [CN]      而且模型侧按名字拿.tensor，边读边装可以把峰值内存压到最低。
# [CN]   2. 加载器不认识任何模型结构 —— 名字到模块的映射完全由 model.load_weights 负责，
# [CN]      所以新增模型不需要动这里。
# [CN] 易错点：加载完成后是否做严格检查（有没有参数没被初始化）取决于
# [CN]         是否量化 + 是否支持 loaded_weights 追踪 —— 量化模型默认不严格检查。

import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    instanttensor_weights_iterator,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


# [CN] BaseModelLoader 的一个实现，负责 load_format != 特殊格式时的默认路径。

class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    # [CN] 并行读取的线程数。超过 8 通常受磁盘 IOPS 限制不再线性加速。

    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    # [CN] 权重的来源描述。允许多个 Source（见 secondary_weights），
    # [CN] 用于把 tokenizer 之外的辅助权重（如 draft 模型）从另一处一边流式读出。

    class Source:
        """A source for weights."""

        # [CN] 既可以是本地目录也可以是 HF repo id，两者在本文件里全程走同一套逻辑。

        model_or_path: str
        """The model ID or path."""

        # [CN] None 表示用主分支最新commit —— 生产环境务必固定，否则权重可能悄悄变化。

        revision: str | None
        """The optional model revision."""

        subfolder: str | None = None
        """The subfolder inside the model repo."""

        # [CN] 给所有权重名加前缀。多模型共用一组权重文件时用它做命名空间隔离。

        prefix: str = ""
        """A prefix to prepend to all weights."""

        # [CN] 找不到 safetensors/bin 时是否退回去找 .pt。显式指定了 safetensors 格式时
        # [CN] 这条路会被关掉（见 _prepare_weights），避免把 .pt 当 safetensors 打开。

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        # [CN] 给了这个值之后，本类自己按 load_format 推 patterns 的逻辑就被完全跳过。

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    # [CN] 两个时间戳是类属性而非实例属性：多个 loader 实例也要能对得上加载耗时。

    counter_before_loading_weights: float = 0.0
    # [CN] 与起始时间配对，两者相减得到纯粹的加载耗时。

    counter_after_loading_weights: float = 0.0

    # [CN] 这里把 model_loader_extra_config 这种「dict 形式的扩展参数」提前校验掉 ——
    # [CN] 失败早于真正开始读盘，避免下载了几百 GB 才告诉我参数写错。

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"model_loader_extra_config must be a dict for load format "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )
        allowed_keys = {
            "enable_multithread_load",
            "num_threads",
            "enable_weights_track",
        }
        # [CN] 白名单式校验：拼错的 key（比如多线程拼错）不会被静默忽略。

        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        enable_multithread_load = extra_config.get("enable_multithread_load", False)
        if not isinstance(enable_multithread_load, bool):
            raise ValueError(
                f"enable_multithread_load must be a bool, got "
                f"{type(enable_multithread_load).__name__}"
            )
        num_threads = extra_config.get("num_threads")
        if num_threads is not None and not (
            isinstance(num_threads, int) and num_threads > 0
        ):
            raise ValueError(
                f"num_threads must be a positive integer, got {num_threads!r}"
            )

        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

        # [CN] 显式拒绝不支持的组合，而不是悄悄丢掉用户指定的策略 ——
        # [CN] 静默降级会让用户以为配置生效了，实际行为完全不同。

        # The multi-thread loader ignores safetensors_load_strategy, so reject
        # the combination instead of silently dropping the requested strategy.
        if extra_config.get("enable_multithread_load") and (
            load_config.safetensors_load_strategy not in (None, "lazy")
        ):
            raise ValueError(
                "enable_multithread_load does not support "
                "safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; the multi-thread "
                "loader only implements the default lazy strategy."
            )

    # [CN] 这一步只做「决定读哪些文件」，不读内容。返回 (目录, 文件列表, 是否 safetensors)。

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        # [CN] auto 会优先探测 mistral 官方格式（consolidated*.safetensors）——
        # [CN] 因为这两类文件的组织方式不同，必须二选一，不能混。

        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        # [CN] mistral 官方格式用 consolidated*.safetensors，索引文件名也与 HF 的不同。

        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        # [CN] npcache 走 numpy 内存映射缓存，只认 .bin。

        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        # Don't fall back to .pt for explicit safetensors formats; otherwise a
        # .pt file is matched and later opened as safetensors.
        # [CN] 只在没走显式 safetensors 格式时才追加 .pt：否则下一轮会把 .pt 当 safetensors 解析而炸掉。

        if fall_back_to_pt and not use_safetensors:
            allow_patterns += ["*.pt"]

        # [CN] 模型类可以通过 allow_patterns_overrides 完全接管文件选择（多模态/多子模型场景常用）。

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        # [CN] 按通配符顺序找，一旦某类命中就停止 —— 顺序隐含优先级（safetensors 优先于 bin）。

        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern.endswith(".safetensors"):
                    use_safetensors = True
                break

        # [CN] 去重这一步很关键：Mistral-7B-Instruct-v0.3 同时存在分片文件和 consolidated 文件，
        # [CN] 两组一起读会出现重复权重（且不保证一致），因此按 index 过滤掉多余的。

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local and len(hf_weights_files) > 1:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        # [CN] 一个文件都没找到就明确报错 —— 常见原因其实是 revision/subfolder 写错。

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    # [CN] 按 load_format 分派到具体的 reader。所有分支最终都产出统一的 (name, tensor) 流。

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        # [CN] npcache 只支持 .bin checkpoint —— 它依赖 pickle 加载后的 numpy 缓存。

        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            # [CN] fastsafetensors / instanttensor 是两个第三方加速后端，按包名分派。

            elif self.load_config.load_format == "instanttensor":
                weights_iterator = instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get(
                            "num_threads", self.DEFAULT_NUM_THREADS
                        ),
                    )
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads
                        ),
                        safetensors_prefetch_block_size=(
                            self.load_config.safetensors_prefetch_block_size
                        ),
                    )
        else:
            # [CN] 多线程读取牺牲了 lazy/strategy 语义换并发度，所以只实现了默认策略。

            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        # [CN] 只在首次记录起始时间：多次迭代同一份权重时不希望把间隔计入耗时。

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # [CN] 返回生成器而非列表：加上 prefix 这一步本身也必须是惰性的，否则会一次性拉出全部权重。

        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    # [CN] 先给主权重，再依次给 secondary_weights —— 让模型能用一份迭代器装完所有子模型。

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        # [CN] secondary_weights 由模型类自己声明（如 spec decode 的 draft 模型），加载器不做假设。

        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    # [CN] 只下载不加载：通常用于提前预热 HF 缓存，避免真正启动时才拉权重。

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    # [CN] EP 下的专家权重过滤：每个 rank 只需要自己那部分专家，
    # [CN] 在**读盘之前**跳过不相关的 tensor 能显著省掉 IO 与内存。

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering.

        When expert parallelism is active, each rank only needs a subset of
        expert weights.  By computing the set upfront we can skip non-local
        expert tensors *before* reading them from disk.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        # [CN] 三个条件同时满足才启用：是 MoE + 开了 EP + 允许过滤。

        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return

        # When EPLB is enabled, redundant physical expert slots may map to
        # logical experts that belong to other ranks in the default partition.
        # The weight loader needs to see ALL logical expert weights so it can
        # populate these redundant slots.  Skip the filter entirely.
        # [CN] EPLB 下冗余的物理专家槽位可能映射到别的 rank 的逻辑专家，
        # [CN] 因此必须让 loader 看到全部逻辑专家权重 —— 这里直接放弃过滤而非「部分过滤」。

        if parallel_config.enable_eplb:
            return

        num_experts = model_config.get_num_experts()
        # [CN] 拿不到专家数就放弃过滤（保守但需要多读盘），总比算错 EP 切分好。

        if num_experts <= 0:
            return

        # [CN] EP 坐标是把 dp/pcp/tp 三个维度展平算出来的，必须与 FusedMoE 侧的算法一致，
        # [CN] 否则会加载到别人那一份专家。

        # EP size/rank computation mirrors FusedMoEParallelConfig.make():
        #   ep_size = dp_size * pcp_size * tp_size (flattened)
        #   ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    @instrument(span_name="Load weights")
    # [CN] 主入口。真正转移所有权的一步是 model.load_weights(...)，加载器只负责提供流。

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # [CN] torchao 序列化的 checkpoint 需要专门的 safetensors 读取策略，并且有版本下限要求。

        if model_config.quantization == "torchao":
            quant_config = get_quant_config(model_config, self.load_config)
            if (
                hasattr(quant_config, "is_checkpoint_torchao_serialized")
                and quant_config.is_checkpoint_torchao_serialized
                and torchao_version_at_least("0.15.0")
            ):
                self.load_config.safetensors_load_strategy = "torchao"

        self._init_ep_weight_filter(model_config)

        # [CN] model.load_weights 返回它**实际消费过的**权重名集合，返回值可能 None
        # [CN] （表示这个模型不支持追踪），下面据此决定是否做严格检查。

        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # [CN] 严格检查只对未量化且支持追踪的模型默认开启：量化模型的很多参数是
        # [CN] process_weights_after_loading 现场生成的，不存在于 checkpoint 里。

        # We only enable strict check for non-quantized models
        # that have loaded weights tracking by default.
        default_enable_weights_track = (
            model_config.quantization is None and loaded_weights is not None
        )
        enable_weights_track = (
            self.enable_weights_track
            if self.enable_weights_track is not None
            else default_enable_weights_track
        )
        if enable_weights_track:
            self.track_weights_loading(model, loaded_weights)

    # [CN] 严格性检查：找出「存在但从未从 checkpoint 初始化过」的参数并报错。

    def track_weights_loading(
        self, model: nn.Module, loaded_weights: set[str] | None
    ) -> None:
        # [CN] 以 named_parameters 为准而不是 named_modules：buffer 本来就可以不来自 checkpoint。

        weights_to_load = {name for name, _ in model.named_parameters()}
        # [CN] 先把那些「允许缺失」的补进 loaded_weights：在线量化 scale、加载后处理的 scale、
        # [CN] kv_cache scale 都不在 checkpoint 里，属于正常缺席。

        if loaded_weights is not None:
            # ignore online quantization scales
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                has_online_quant = getattr(quant_method, "uses_meta_device", False)
                has_postprocess_quant = getattr(
                    quant_method, "process_weights_after_loading", None
                )
                # ignore kv_cache scale and online quant scale,
                # which can be missing in checkpoints
                if has_online_quant or has_postprocess_quant:
                    for param_name, _ in module.named_parameters():
                        full_name = f"{name}.{param_name}" if name else param_name
                        loaded_weights.add(full_name)
            # [CN] 集合差。非空意味着有参数留着随机初始值 —— 通常源自权重名不匹配（改名/版本差异）。

            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )
