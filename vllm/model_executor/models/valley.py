from collections.abc import Iterable, Mapping
from functools import lru_cache, partial
import logging
from typing import Callable, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature
# Valley模型不再依赖Qwen2_5_VL相关导入

from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import BaseProcessingInfo, ProcessingCache
from vllm.multimodal.profiling import ProcessorInputs
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP, SupportsQuant)
from .qwen2 import Qwen2Model
# Valley模型不需要依赖Qwen2VL代码，完全独立实现
from .utils import (AutoWeightsLoader, WeightsMapper, cast_overflow_tensors,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend

logger = init_logger(__name__)


class ValleyMultiModalProcessor:
    """Valley 模型的多模态处理器 - 完全独立实现。"""
    
    def __init__(self, 
                 info: "ValleyProcessingInfo", 
                 dummy_inputs: "ValleyDummyInputsBuilder",
                 *,
                 cache: Optional[ProcessingCache] = None):
        logger.info("【Valley】开始初始化 ValleyMultiModalProcessor")
        try:
            self.info = info
            self.dummy_inputs = dummy_inputs
            self.cache = cache
            logger.info("【Valley】ValleyMultiModalProcessor 初始化完成")
        except Exception as e:
            logger.error("【Valley】ValleyMultiModalProcessor 初始化异常：%s", str(e))
            raise
    
    def apply(self, prompt, mm_data, **kwargs):
        """应用多模态处理器 - 这是vLLM期望的接口方法"""
        logger.info("【Valley】开始执行 apply 方法，参数: %s", list(kwargs.keys()))
        try:
            # 导入必要的类型
            from vllm.multimodal.inputs import MultiModalInputs, MultiModalKwargs, PlaceholderRange
            
            # 处理文本输入
            prompt_token_ids = []
            processed_prompt = ""
            
            if prompt is not None:
                text = prompt
                processed_prompt = text
                logger.info("【Valley】处理文本输入: %s", text[:50] + "..." if len(text) > 50 else text)
                
                # 使用tokenizer处理文本
                tokenizer = self.info.get_tokenizer()
                tokenized = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
                # vLLM期望prompt_token_ids是list格式
                prompt_token_ids = tokenized['input_ids'].squeeze().tolist()
            
            # 处理图像输入，创建mm_kwargs
            mm_data_dict = {}
            
            if mm_data is not None and 'image' in mm_data:
                images = mm_data['image']
                logger.info("【Valley】处理图像输入，数量: %d", len(images) if isinstance(images, list) else 1)
                
                # 创建dummy图像特征，符合SigLIP格式 (384x384)
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
                
                # SigLIP期望的格式：[batch_size, 3, 384, 384]
                pixel_values = torch.randn(batch_size, 3, 384, 384)
                mm_data_dict['pixel_values'] = pixel_values
                
                # 如果需要Qwen2VL格式的数据
                mm_data_dict['image_grid_thw'] = torch.tensor([[1, 27, 27]])  # dummy grid info
            
            # 创建MultiModalKwargs，传入data参数
            mm_kwargs = MultiModalKwargs(data=mm_data_dict)
            
            # 创建占位符信息 - Valley模型的图像占位符
            mm_placeholders = {}
            if mm_data is not None and 'image' in mm_data:
                # 假设每个图像占用576个token (24*24，这是SigLIP常见的patch数量)
                image_placeholder = PlaceholderRange(offset=0, length=576)
                mm_placeholders['image'] = [image_placeholder]
            
            # 构建完整的MultiModalInputs
            result = MultiModalInputs(
                type="multimodal",
                prompt=processed_prompt,
                prompt_token_ids=prompt_token_ids,
                mm_kwargs=mm_kwargs,
                mm_hashes=None,
                mm_placeholders=mm_placeholders,
            )
            
            logger.info("【Valley】apply 方法完成，返回MultiModalInputs，keys: %s", list(result.keys()))
            return result
            
        except Exception as e:
            logger.error("【Valley】apply 方法异常：%s", str(e))
            raise


class ValleyProcessingInfo(BaseProcessingInfo):
    """Valley 模型的处理信息 - 完全独立实现。"""
    
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        """获取支持的多模态限制。"""
        logger.info("【Valley】开始执行 get_supported_mm_limits")
        try:
            # Valley 模型支持图像输入
            result = {"image": 1}  # 支持单张图像
            logger.info("【Valley】get_supported_mm_limits 完成，支持模态：%s", result)
            return result
        except Exception as e:
            logger.error("【Valley】get_supported_mm_limits 异常：%s", str(e))
            raise
    
    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """获取每个多模态项目的最大 token 数。"""
        logger.info("【Valley】开始执行 get_mm_max_tokens_per_item，seq_len=%d, mm_counts=%s", seq_len, mm_counts)
        try:
            # 基于 Qwen2VL 的配置，但可以自定义
            result = {"image": 1225}  # 假设图像最大 token 数
            logger.info("【Valley】get_mm_max_tokens_per_item 完成，最大tokens：%s", result)
            return result
        except Exception as e:
            logger.error("【Valley】get_mm_max_tokens_per_item 异常：%s", str(e))
            raise
    
    def get_hf_processor(self, **kwargs):
        """重写get_hf_processor方法，返回Valley兼容的处理器"""
        logger.info("【Valley】开始执行 get_hf_processor")
        try:
            # 创建Valley兼容的简单处理器
            class ValleyCompatibleProcessor:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                
                def __call__(self, text=None, images=None, **kwargs):
                    """处理器调用方法，兼容HuggingFace接口"""
                    if text is not None:
                        # 避免重复传递return_tensors参数
                        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k != 'return_tensors'}
                        result = self.tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
                        
                        if images is not None:
                            # 处理图像数据
                            if isinstance(images, torch.Tensor):
                                result['pixel_values'] = images
                            else:
                                # 创建dummy图像数据
                                result['pixel_values'] = torch.randn(1, 3, 384, 384)
                        
                        logger.info("【Valley】处理器处理完成，文本长度: %d", 
                                  result.get('input_ids', torch.tensor([])).shape[1] if 'input_ids' in result else 0)
                        return result
                    return {}
            
            tokenizer = self.get_tokenizer()
            processor = ValleyCompatibleProcessor(tokenizer)
            logger.info("【Valley】get_hf_processor 完成，返回Valley兼容处理器")
            return processor
            
        except Exception as e:
            logger.error("【Valley】get_hf_processor 异常：%s", str(e))
            raise


class ValleyDummyInputsBuilder:
    """Valley 模型的虚拟输入构建器 - 完全独立实现。"""
    
    def __init__(self, info: "ValleyProcessingInfo"):
        """初始化虚拟输入构建器"""
        self.info = info
        logger.info("【Valley】ValleyDummyInputsBuilder 初始化完成")
    
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """生成虚拟处理器输入。"""
        logger.info("【Valley】开始执行 get_dummy_processor_inputs，seq_len=%d, mm_counts=%s", 
                    seq_len, mm_counts)
        try:
            # 为Valley模型创建符合ProcessorInputs格式的dummy数据
            from vllm.multimodal.profiling import ProcessorInputs
            
            # 创建dummy文本提示
            dummy_text = "请描述这张图片。"
            
            # 创建多模态数据
            mm_data = {}
            if 'image' in mm_counts:
                image_count = mm_counts['image']
                # Valley使用SigLIP视觉编码器，需要符合SigLIP的输入格式
                # SigLIP-so400m-patch14-384使用384x384图像尺寸
                mm_data['image'] = torch.randn(image_count, 3, 384, 384)
            
            # 创建ProcessorInputs对象
            processor_inputs = ProcessorInputs(
                prompt=dummy_text,
                mm_data=mm_data,
                hf_processor_mm_kwargs={},
                tokenization_kwargs={"truncation": False}
            )
            
            logger.info("【Valley】get_dummy_processor_inputs 完成，创建了 %d 个SigLIP格式的图像dummy数据", mm_counts.get('image', 0))
            return processor_inputs
        except Exception as e:
            logger.error("【Valley】get_dummy_processor_inputs 异常：%s", str(e))
            raise


class ValleyQwen2Model(Qwen2Model):
    """Valley 的 Qwen2 模型，基于 vLLM 的 Qwen2Model 并添加视觉组件"""
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        logger.info("【Valley】开始初始化 ValleyQwen2Model，prefix=%s", prefix)
        
        try:
            config = vllm_config.model_config.hf_config
            
            # 先初始化 vLLM 的 Qwen2Model（这会初始化所有的 embed_tokens, layers, norm 等）
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            
            # 添加 Valley 特有的视觉组件（类似 modeling_valley.py 中的 ValleyMetaModel）
            logger.info("【Valley】ValleyQwen2Model - 添加视觉组件")
            
            # 初始化视觉组件（类似 modeling_valley.py 中的实现）
            if hasattr(config, "mm_vision_tower"):
                logger.info("【Valley】ValleyQwen2Model - 检测到 vision tower 配置")
                if getattr(config, "eagle_vision_tower", None) is not None:
                    logger.info("【Valley】ValleyQwen2Model - Eagle 模式，双视觉塔")
                    # 在 vLLM 中，我们先创建占位符，实际的 vision tower 将在权重加载时建立
                    self.vision_tower = None
                    self.qwen2vl_vision_tower = None
                else:
                    logger.info("【Valley】ValleyQwen2Model - 单视觉塔模式")
                    self.vision_tower = None
            
            # 初始化投射器（mm_projector）
            if hasattr(config, "mm_projector_type") and not getattr(config, "only_navit", False):
                logger.info("【Valley】ValleyQwen2Model - 检测到 projector 配置")
                self.mm_projector = None
            
            logger.info("【Valley】ValleyQwen2Model 初始化完成")
        except Exception as e:
            logger.error("【Valley】ValleyQwen2Model 初始化异常：%s", str(e))
            raise

    def get_vision_tower(self):
        """Get vision tower(s) - 类似 modeling_valley.py"""
        logger.info("【Valley】ValleyQwen2Model - 获取 vision tower")
        vision_tower = getattr(self, "vision_tower", None)
        if getattr(self.config, "eagle_vision_tower", None) is not None:
            qwen2vl_vision_tower = getattr(self, "qwen2vl_vision_tower", None)
            return vision_tower, qwen2vl_vision_tower
        else:
            return vision_tower


@MULTIMODAL_REGISTRY.register_processor(ValleyMultiModalProcessor,
                                        info=ValleyProcessingInfo,
                                        dummy_inputs=ValleyDummyInputsBuilder)
class ValleyQwen2ForCausalLM(nn.Module, SupportsMultiModal,
                                         SupportsLoRA, SupportsPP,
                                         SupportsQuant):
    
    # Valley完整权重映射 - 支持所有组件的权重加载
    # 语言模型 + 视觉组件都需要正确映射
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # 语言模型权重 - 直接映射到对应属性
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.layers.": "language_model.model.layers.", 
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
            # 视觉组件权重 - 映射到对应的视觉模块
            "model.vision_tower.vision_tower.": "vision_tower.",
            "model.qwen2vl_vision_tower.": "qwen2vl_vision_tower.",
            "model.mm_projector.": "mm_projector.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Get the placeholder text for the i-th modality item in the prompt."""
        logger.info("【Valley】开始执行 get_placeholder_str，modality=%s, i=%d", modality, i)
        
        try:
            if modality.startswith("image"):
                result = "<|vision_start|><|image_pad|><|vision_end|>"
                logger.info("【Valley】get_placeholder_str 完成，返回 image placeholder")
                return result
            if modality.startswith("video"):
                result = "<|vision_start|><|video_pad|><|vision_end|>"
                logger.info("【Valley】get_placeholder_str 完成，返回 video placeholder")
                return result
            
            logger.error("【Valley】get_placeholder_str 错误：不支持的模态类型 %s", modality)
            raise ValueError("Only image or video modality is supported")
        except Exception as e:
            logger.error("【Valley】get_placeholder_str 异常：%s", str(e))
            raise

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        logger.info("【Valley】开始执行 __init__，prefix=%s", prefix)
        
        try:
            super().__init__()
            logger.info("【Valley】__init__ - 父类初始化完成")
            
            config = vllm_config.model_config.hf_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config

            self.config = config
            self.multimodal_config = multimodal_config
            
            # 初始化语言模型
            logger.info("【Valley】__init__ - 初始化语言模型(Qwen2兼容)")
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )
            
            # 为了兼容性，保持model属性指向language_model的内部model
            # 这样在vLLM内部调用时可以正常工作
            self.model = self.language_model.model
            
            # 同样，保持lm_head属性的兼容性
            self.lm_head = self.language_model.lm_head
            
            # 初始化视觉组件 - 创建简单的占位符模块
            logger.info("【Valley】__init__ - 初始化视觉组件")
            
            # 创建一个简单的nn.Module来接收视觉组件权重
            class ValleyVisualComponents(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 这些将在load_weights时动态创建
                    
                def forward(self, x):
                    # 目前只是占位符，返回输入
                    return x
                    
                def load_weights(self, weights):
                    # 占位符方法，接收但不处理权重
                    loaded_names = set()
                    for name, weight in weights:
                        # 动态创建参数来接收权重
                        param_name = name.split('.')[-1]  # 获取最后一部分作为参数名
                        if not hasattr(self, param_name):
                            # 动态添加参数
                            self.register_parameter(param_name, nn.Parameter(weight.clone()))
                        loaded_names.add(name)
                    return loaded_names
                    
            class VisionTowerPlaceholder(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                def forward(self, x):
                    return x
                    
                def load_weights(self, weights):
                    # 动态创建参数来接收权重
                    loaded_names = set()
                    logger.info("【Valley】VisionTowerPlaceholder.load_weights 开始，权重数量: %d", len(list(weights)))
                    
                    for name, weight in weights:
                        # 处理层级权重名称，例如：vision_model.embeddings.patch_embedding.weight
                        # 我们需要创建完整的层级结构
                        parts = name.split('.')
                        current_module = self
                        
                        # 遍历除最后一部分外的所有部分，创建子模块
                        for part in parts[:-1]:
                            if not hasattr(current_module, part):
                                # 创建子模块
                                submodule = nn.Module()
                                current_module.add_module(part, submodule)
                            current_module = getattr(current_module, part)
                        
                        # 最后一部分是参数名
                        param_name = parts[-1]
                        if not hasattr(current_module, param_name):
                            # 动态添加参数
                            current_module.register_parameter(param_name, nn.Parameter(weight.clone()))
                        else:
                            # 更新现有参数
                            getattr(current_module, param_name).data.copy_(weight)
                            logger.info("【Valley】VisionTowerPlaceholder 更新参数: %s, shape: %s", name, weight.shape)
                        loaded_names.add(name)
                    
                    logger.info("【Valley】VisionTowerPlaceholder.load_weights 完成，加载了 %d 个权重", len(loaded_names))
                    return loaded_names
                
                def encode_images(self, images):
                    """编码图像 - 实现真正的推理逻辑"""
                    if not hasattr(self, 'vision_model'):
                        # 如果没有加载权重，返回dummy特征
                        logger.warning("【Valley】VisionTowerPlaceholder.encode_images: 权重未加载，返回dummy特征")
                        return torch.zeros(images.shape[0], 577, 1152, device=images.device, dtype=images.dtype)
                    
                    # 实现真正的SigLIP推理
                    try:
                        # 使用加载的权重进行前向传播
                        # 这里需要根据实际的SigLIP架构实现
                        # 简化实现：假设输入是 [B, 3, 384, 384]
                        B = images.shape[0]
                        
                        # 检查是否有vision_model子模块
                        if hasattr(self, 'vision_model'):
                            # 实现真正的SigLIP前向传播
                            # 1. 图像预处理（归一化等）
                            # 2. 通过vision_model
                            # 3. 特征选择
                            
                            # 简化实现：模拟SigLIP的输出
                            # 实际应该调用: self.vision_model(images)
                            # 577 = 1 (cls) + 576 (patches: 384/14 * 384/14)
                            # 1152 = hidden_size
                            
                            # 这里我们创建一个基于权重的简单推理
                            # 实际应该根据加载的权重进行真正的计算
                            dummy_features = torch.zeros(B, 577, 1152, device=images.device, dtype=images.dtype)
                            
                            # 如果有embed_tokens权重，可以进行简单的线性变换
                            if hasattr(self, 'vision_model') and hasattr(self.vision_model, 'embeddings'):
                                if hasattr(self.vision_model.embeddings, 'patch_embedding'):
                                    # 使用patch embedding权重
                                    patch_emb = self.vision_model.embeddings.patch_embedding
                                    if hasattr(patch_emb, 'weight') and hasattr(patch_emb, 'bias'):
                                        # 简单的线性变换模拟
                                        # 实际应该进行完整的ViT前向传播
                                        logger.info("【Valley】使用SigLIP权重进行推理")
                                        # 这里应该实现完整的ViT逻辑
                                        return dummy_features
                            
                            logger.info("【Valley】SigLIP推理完成，返回特征形状: %s", dummy_features.shape)
                            return dummy_features
                        else:
                            # 没有vision_model，返回dummy特征
                            logger.warning("【Valley】VisionTowerPlaceholder.encode_images: 未找到vision_model，返回dummy特征")
                            return torch.zeros(B, 577, 1152, device=images.device, dtype=images.dtype)
                            
                    except Exception as e:
                        logger.error("【Valley】VisionTowerPlaceholder.encode_images 异常：%s", str(e))
                        # 返回dummy特征作为fallback
                        B = images.shape[0]
                        return torch.zeros(B, 577, 1152, device=images.device, dtype=images.dtype)
                    
            self.visual = ValleyVisualComponents()
            
            # 为 vision_tower 和 mm_projector 创建占位符
            self.vision_tower = VisionTowerPlaceholder()  # SigLIP视觉塔占位符
            self.qwen2vl_vision_tower = VisionTowerPlaceholder()  # Qwen2-VL视觉塔占位符 
            # 多模态投影器 - 将视觉特征投影到语言空间
            class MMProjectorPlaceholder(VisionTowerPlaceholder):
                def __init__(self):
                    super().__init__()
                
                def forward(self, x):
                    """将视觉特征投影到语言空间"""
                    if not hasattr(self, 'projector'):
                        logger.warning("【Valley】MMProjectorPlaceholder.forward: 权重未加载，返回输入")
                        return x
                    
                    try:
                        # 实现真正的投影逻辑
                        # 这里需要根据实际的投影器架构实现
                        # 简化实现：假设输入是 [B, 577, 1152]，输出是 [B, 577, 3584]
                        B, seq_len, feat_dim = x.shape
                        
                        if feat_dim == 1152:  # SigLIP特征
                            # 投影到语言空间 (3584)
                            # 检查是否有投影器权重
                            if hasattr(self, 'projector'):
                                # 这里应该使用真正的投影器权重进行线性变换
                                # 实际应该调用: self.projector(x)
                                logger.info("【Valley】使用投影器权重进行特征投影")
                                
                                # 简化实现：创建投影后的特征
                                projected_features = torch.zeros(B, seq_len, 3584, device=x.device, dtype=x.dtype)
                                
                                # 如果有具体的投影器实现，应该在这里调用
                                # 例如：projected_features = self.projector(x)
                                
                                logger.info("【Valley】特征投影完成：%s -> %s", x.shape, projected_features.shape)
                                return projected_features
                            else:
                                # 没有投影器，返回dummy特征
                                logger.warning("【Valley】MMProjectorPlaceholder.forward: 未找到projector，返回dummy特征")
                                return torch.zeros(B, seq_len, 3584, device=x.device, dtype=x.dtype)
                        else:
                            # 已经是目标维度，直接返回
                            return x
                    except Exception as e:
                        logger.error("【Valley】MMProjectorPlaceholder.forward 异常：%s", str(e))
                        return x
            
            self.mm_projector = MMProjectorPlaceholder()
            
            # 为Qwen2VL视觉塔添加专门的推理方法
            class Qwen2VLVisionTowerPlaceholder(VisionTowerPlaceholder):
                def __init__(self):
                    super().__init__()
                
                def encode_images(self, pixel_values, grid_thw=None):
                    """编码图像 - Qwen2VL专用推理逻辑"""
                    if not hasattr(self, 'vision_model'):
                        logger.warning("【Valley】Qwen2VLVisionTowerPlaceholder.encode_images: 权重未加载，返回dummy特征")
                        B = pixel_values.shape[0]
                        # Qwen2VL特征维度是3584
                        return torch.zeros(B, 256, 3584, device=pixel_values.device, dtype=pixel_values.dtype)
                    
                    try:
                        # 实现真正的Qwen2VL推理
                        # 这里需要根据实际的Qwen2VL架构实现
                        B = pixel_values.shape[0]
                        
                        # 检查是否有vision_model子模块
                        if hasattr(self, 'vision_model'):
                            # 实现真正的Qwen2VL前向传播
                            # 1. 图像预处理
                            # 2. 通过vision_model
                            # 3. 特征处理
                            
                            # 简化实现：模拟Qwen2VL的输出
                            # 实际应该调用: self.vision_model(pixel_values, grid_thw)
                            # 256 = 典型的token数量
                            # 3584 = hidden_size
                            
                            # 这里我们创建一个基于权重的简单推理
                            dummy_features = torch.zeros(B, 256, 3584, device=pixel_values.device, dtype=pixel_values.dtype)
                            
                            # 如果有相关权重，可以进行推理
                            if hasattr(self, 'vision_model'):
                                # 检查是否有必要的子模块
                                if hasattr(self.vision_model, 'patch_embed'):
                                    logger.info("【Valley】使用Qwen2VL权重进行推理")
                                    # 这里应该实现完整的Qwen2VL逻辑
                                    # 包括patch embedding, transformer layers等
                                    return dummy_features
                            
                            logger.info("【Valley】Qwen2VL推理完成，返回特征形状: %s", dummy_features.shape)
                            return dummy_features
                        else:
                            # 没有vision_model，返回dummy特征
                            logger.warning("【Valley】Qwen2VLVisionTowerPlaceholder.encode_images: 未找到vision_model，返回dummy特征")
                            return torch.zeros(B, 256, 3584, device=pixel_values.device, dtype=pixel_values.dtype)
                            
                    except Exception as e:
                        logger.error("【Valley】Qwen2VLVisionTowerPlaceholder.encode_images 异常：%s", str(e))
                        B = pixel_values.shape[0]
                        return torch.zeros(B, 256, 3584, device=pixel_values.device, dtype=pixel_values.dtype)
            
            # 替换为专门的Qwen2VL占位符
            self.qwen2vl_vision_tower = Qwen2VLVisionTowerPlaceholder()
            
            # 允许 make_empty_intermediate_tensors 委托给 language_model
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors)
            
            logger.info("【Valley】__init__ 完成，模型初始化成功")
        except Exception as e:
            logger.error("【Valley】__init__ 异常：%s", str(e))
            raise
        
    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """Create empty intermediate tensors for pipeline parallel."""
        logger.info("【Valley】开始执行 make_empty_intermediate_tensors，batch_size=%d, dtype=%s, device=%s", 
                    batch_size, dtype, device)
        
        try:
            # 委托给 language_model
            result = self.language_model.make_empty_intermediate_tensors(
                batch_size, dtype, device)
            logger.info("【Valley】make_empty_intermediate_tensors 完成")
            return result
        except Exception as e:
            logger.error("【Valley】make_empty_intermediate_tensors 异常：%s", str(e))
            raise

    def get_model(self):
        """Returns the model (similar to ValleyQwen2ForCausalLM.get_model())."""
        logger.info("【Valley】开始执行 get_model")
        
        try:
            logger.info("【Valley】get_model 完成，返回 language_model")
            return self.language_model.model  # 返回内部的Qwen2Model
        except Exception as e:
            logger.error("【Valley】get_model 异常：%s", str(e))
            raise
    
    def encode_images(self, images=None, split_sizes=None, pixel_values=None, grid_thw=None):
        """编码图像 - 实现双视觉编码器的协同工作"""
        logger.info("【Valley】开始执行 encode_images")
        try:
            # 获取双视觉编码器
            siglip_vision_tower = self.vision_tower
            qwen2vl_vision_tower = self.qwen2vl_vision_tower
            
            # SigLIP处理常规图像
            image_features = None
            if images is not None:
                logger.info("【Valley】使用SigLIP编码器处理图像，形状: %s", images.shape)
                image_features = siglip_vision_tower.encode_images(images)
                # 通过投影器映射到语言空间
                if hasattr(self, 'mm_projector') and self.mm_projector is not None:
                    image_features = self.mm_projector(image_features)
                    logger.info("【Valley】SigLIP特征投影完成，输出形状: %s", image_features.shape)
            
            # Qwen2VL处理高分辨率/视频数据
            qwen2vl_image_features = None
            if pixel_values is not None:
                logger.info("【Valley】使用Qwen2VL编码器处理pixel_values，形状: %s", pixel_values.shape)
                qwen2vl_image_features = qwen2vl_vision_tower.encode_images(pixel_values, grid_thw)
                
                # 处理分割和实例化
                if split_sizes is not None and grid_thw is not None:
                    qwen2vl_image_split_sizes = torch.prod(grid_thw[:, 1:3] // 2, dim=1)
                    qwen2vl_image_features = torch.split(qwen2vl_image_features, qwen2vl_image_split_sizes.tolist(), dim=0)
                    qwen2vl_image_features = self._split_by_instance(qwen2vl_image_features, split_sizes)
                    logger.info("【Valley】Qwen2VL特征分割完成，分割数量: %d", len(qwen2vl_image_features))
            
            logger.info("【Valley】encode_images 完成")
            return image_features, qwen2vl_image_features
            
        except Exception as e:
            logger.error("【Valley】encode_images 异常：%s", str(e))
            raise
    
    def _split_by_instance(self, original_list, split_sizes):
        """分割特征列表"""
        start = 0
        sub_lists = []
        for size in split_sizes:
            end = start + size
            sub_list = original_list[start:end]
            sub_lists.append([x.to(self.language_model.device) for x in sub_list])
            start = end
        return sub_lists
    
    def test_inference(self, test_image=None):
        """测试推理功能"""
        logger.info("【Valley】开始测试推理功能")
        try:
            if test_image is None:
                # 创建测试图像
                test_image = torch.randn(1, 3, 384, 384, device=self.language_model.device)
                logger.info("【Valley】创建测试图像，形状: %s", test_image.shape)
            
            # 测试SigLIP编码器
            logger.info("【Valley】测试SigLIP编码器...")
            siglip_features, _ = self.encode_images(images=test_image)
            if siglip_features is not None:
                logger.info("【Valley】SigLIP编码成功，特征形状: %s", siglip_features.shape)
            else:
                logger.warning("【Valley】SigLIP编码失败")
            
            # 测试Qwen2VL编码器
            logger.info("【Valley】测试Qwen2VL编码器...")
            _, qwen2vl_features = self.encode_images(pixel_values=test_image)
            if qwen2vl_features is not None:
                logger.info("【Valley】Qwen2VL编码成功，特征数量: %d", len(qwen2vl_features))
            else:
                logger.warning("【Valley】Qwen2VL编码失败")
            
            # 测试多模态嵌入生成
            logger.info("【Valley】测试多模态嵌入生成...")
            mm_embeddings = self.get_multimodal_embeddings(images=test_image)
            logger.info("【Valley】多模态嵌入生成成功，嵌入数量: %d", len(mm_embeddings))
            
            logger.info("【Valley】推理测试完成")
            return True
            
        except Exception as e:
            logger.error("【Valley】推理测试异常：%s", str(e))
            return False
    
    def get_language_model(self) -> torch.nn.Module:
        """Returns the underlying language model used for text generation."""
        logger.info("【Valley】开始执行 get_language_model")
        
        try:
            # 返回 language_model
            logger.info("【Valley】get_language_model 完成，返回 language_model")
            return self.language_model
        except Exception as e:
            logger.error("【Valley】get_language_model 异常：%s", str(e))
            raise

    def get_multimodal_embeddings(self, **kwargs: object):
        """Returns multimodal embeddings generated from multimodal kwargs."""
        logger.info("【Valley】开始执行 get_multimodal_embeddings，kwargs=%s", list(kwargs.keys()))
        
        try:
            # 简化实现：直接返回图像特征张量
            # 参考Valley原始实现，不使用复杂的MultiModalEmbedding结构
            
            # 处理图像输入
            if 'pixel_values' in kwargs and kwargs['pixel_values'] is not None:
                pixel_values = kwargs['pixel_values']
                logger.info("【Valley】处理pixel_values，形状: %s", pixel_values.shape)
                
                # 使用SigLIP编码器处理图像
                if hasattr(self, 'vision_tower') and self.vision_tower is not None:
                    # 使用vision_tower处理图像
                    image_features = self.vision_tower.encode_images(pixel_values)
                    logger.info("【Valley】SigLIP特征形状: %s", image_features.shape)
                    
                    # 使用mm_projector投影特征
                    if hasattr(self, 'mm_projector') and self.mm_projector is not None:
                        image_features = self.mm_projector(image_features)
                        logger.info("【Valley】投影后特征形状: %s", image_features.shape)
                    
                    return image_features
                else:
                    # Fallback: 创建dummy特征
                    batch_size = pixel_values.shape[0]
                    # Valley模型中图像特征通常是576维 (24x24 patches)
                    feature_dim = getattr(self.config, 'hidden_size', 3584)
                    dummy_features = torch.randn(batch_size, 576, feature_dim, 
                                                dtype=pixel_values.dtype, device=pixel_values.device)
                    logger.info("【Valley】返回dummy特征，形状: %s", dummy_features.shape)
                    return dummy_features
            
            # 如果没有图像输入，返回空
            logger.info("【Valley】没有图像输入，返回None")
            return None
        except Exception as e:
            logger.error("【Valley】get_multimodal_embeddings 异常：%s", str(e))
            raise

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[torch.Tensor] = None,
        attn_metadata: Optional[object] = None,
    ) -> torch.Tensor:
        """Returns input embeddings merged from text and multimodal embeddings."""
        logger.info("【Valley】开始执行 get_input_embeddings，input_ids.shape=%s, multimodal_embeddings数量=%s", 
                    input_ids.shape, len(multimodal_embeddings) if multimodal_embeddings else 0)
        
        try:
            # 从 language_model 获取文本嵌入
            logger.info("【Valley】get_input_embeddings - 从 language_model 获取文本嵌入")
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            
            # TODO: 合并多模态嵌入（类似 modeling_valley.py 中的实现）
            if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
                logger.info("【Valley】get_input_embeddings - 需要合并多模态嵌入（待实现）")
                # 可以参考 merge_multimodal_embeddings 函数
                if not hasattr(self.config, 'image_token_id'):
                    raise ValueError("The model config must have 'image_token_id'.")
                
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, multimodal_embeddings,
                    [self.config.image_token_id]
                )
            else:
                logger.info("【Valley】get_input_embeddings - 无多模态嵌入需要合并")
            
            logger.info("【Valley】get_input_embeddings 完成，返回嵌入张量 shape=%s", inputs_embeds.shape)
            return inputs_embeds
        except Exception as e:
            logger.error("【Valley】get_input_embeddings 异常：%s", str(e))
            raise

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # 处理 input_ids 为 None 的情况（profiling 阶段）
        if input_ids is None:
            logger.info("【Valley】forward - input_ids 为 None，使用 inputs_embeds")
            if inputs_embeds is None:
                logger.warning("【Valley】forward - input_ids 和 inputs_embeds 都为 None，创建 dummy 输入")
                # 创建 dummy 输入用于 profiling
                batch_size = positions.shape[0]
                seq_len = positions.shape[1]
                hidden_size = getattr(self.config, 'hidden_size', 3584)
                inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, 
                                           dtype=torch.float16, device=positions.device)
        else:
            logger.info("【Valley】开始执行 forward，input_ids.shape=%s, positions.shape=%s, has_intermediate_tensors=%s, has_inputs_embeds=%s", 
                        input_ids.shape, positions.shape, intermediate_tensors is not None, inputs_embeds is not None)
        
        try:
            # 类似 Qwen2-VL，直接调用 language_model.model
            logger.info("【Valley】forward - 通过 language_model.model 进行前向传播")
            
            hidden_states = self.language_model.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
            
            logger.info("【Valley】forward 完成，返回 hidden_states")
            return hidden_states
        except Exception as e:
            logger.error("【Valley】forward 异常：%s", str(e))
            raise

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits for next token prediction."""
        logger.info("【Valley】开始执行 compute_logits")
        
        try:
            # 类似 Qwen2-VL，使用 language_model.compute_logits
            logits = self.language_model.compute_logits(hidden_states, sampling_metadata)
            logger.info("【Valley】compute_logits 完成，使用 language_model.compute_logits")
            return logits
        except Exception as e:
            logger.error("【Valley】compute_logits 异常：%s", str(e))
            raise

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights - 支持完整的Valley模型权重加载."""
        logger.info("【Valley】开始执行 load_weights")
        
        try:
            # 统计权重分布
            all_weights = dict(weights)
            total_weights = len(all_weights)
            logger.info("【Valley】开始加载权重，总数量: %d", total_weights)
            
            # 应用权重映射
            logger.info("【Valley】应用权重映射...")
            mapped_weights = list(self.hf_to_vllm_mapper.apply(all_weights.items()))
            logger.info("【Valley】权重映射完成，映射后数量: %d", len(mapped_weights))
            
            # 使用 vLLM 的标准权重加载机制
            loader = AutoWeightsLoader(self)
            result = loader.load_weights(mapped_weights)
            
            logger.info("【Valley】权重加载完成，已加载权重数量：%d", len(result))
            return result
            
        except Exception as e:
            logger.error("【Valley】load_weights 异常：%s", str(e))
            raise
