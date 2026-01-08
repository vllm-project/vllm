import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Any, Tuple
import warnings
from contextlib import contextmanager
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig, fp8_w8a8_moe_quant_config)
from vllm import _offload_C as moe_offload
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.config import get_current_vllm_config
import gc

from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id, dbo_enabled, dbo_switch_to_comm,
    dbo_switch_to_compute, dbo_switch_to_comm_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm)



class CpuOffloadInfer:
    """
    CPU Offload 管理器：支持双层 Miss Expert Buffer
    支持 DB0 模式和 Prefetch 模式
    """
    def __init__(self, total_expert_num, cache_expert_num, top_k, hidden_size, intermediate_size, max_batch_tokens, tp_rank, tp_size):
        
        vllm_config = get_current_vllm_config()
        self.total_expert_num = total_expert_num
        self.cache_expert_num = cache_expert_num
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        #self.max_batch_tokens = 1024
        self.weight_block_size = [128,128]
        self.block_quant = True
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_offload_cache_topk = vllm_config.model_config.moe_offload_cache_topk
        self.moe_offload_update_expert_num = vllm_config.model_config.moe_offload_update_expert_num
        self.moe_offload_context_num_threads = vllm_config.model_config.moe_offload_context_num_threads

        config = moe_offload.MOEConfig(
            tp_rank,  # tp_rank
            tp_size,  # tp_size
            total_expert_num,  # expert_num
            top_k,  # num_experts_per_tok
            hidden_size,  # hidden_size
            intermediate_size,  # intermediate_size
            max_batch_tokens,  # max_batch_token
            cache_expert_num,  # cache_expert_num
            128,  # block_size
            self.moe_offload_cache_topk,  # cache_topk
            self.moe_offload_update_expert_num,  # update_expert_num
            self.moe_offload_context_num_threads,  # context_num_threads
        )
        self.engine = moe_offload.MoeOffloadEngine(config)

        # ==================== 核心数据结构 ====================
        # 存储各层原始 CPU 权重
        self.cpu_weights: Dict[int, Dict[str, Tensor]] = {}

        # 存储各层配置信息
        self.layer_configs: Dict[int, Dict[str, Any]] = {}

        # 已处理层的记录
        self._processed_layers: set = set()

        # Miss Expert Buffer，索引 0 和 1 对应两个 layer
        self.temp_layer: List[Optional[nn.Module]] = [None, None]

        # layer_id 到 buffer 索引的映射
        self._layer_to_buffer_idx: Dict[int, int] = {}
        # 存储 Fp8MoEMethod 引用，用于获取 quant_config 等参数
        self._moe_method_refs: Dict[int, Any] = {}

        # ==================== DB0 相关 ====================



        self.sync_token = None

        # 是否启用 DB0 模式
        self.dbo_enabled: bool = False

        # Cache maps（用于 DB0）
        self.cache_maps: Dict[int, torch.Tensor] = {}
        self.miss_maps: Dict[int, torch.Tensor] = {}
        self.policy_sorts: Dict[int, torch.Tensor] = {}

        # GPU 缓存引用（在 layer 对象中）
        self.w13_caches: Dict[int, Tensor] = {}
        self.w2_caches: Dict[int, Tensor] = {}

        self.update_streams: Optional[torch.cuda.Stream] = None
        self.compute_event: List[Optional[torch.cuda.Event]] = [None, None]
        self.copy_event: List[Optional[torch.cuda.Event]] = [None, None]

    def init_expert_hit_stats(self):
        self.expert_miss = 0
        self.copy_count = 0
        self.expert_count = 0

    def update_expert_hit_stats(self, topk_ids:torch.Tensor, copy_map:torch.Tensor):
        self.expert_miss += topk_ids.sum().cpu().item()


    def setup_cache_maps(self, layer_id: int, total_experts: int, cache_num: int) -> None:
        """
        初始化指定层的 cache 映射结构
        Args:
            layer_id: 层ID
            total_experts: 总专家数量
            cache_num: cache中的专家数量

        说明:
            - cache_maps:  专家ID → cache位置，初始前cache_num个专家映射到0~cache_num-1，其余为-1
            - miss_maps:   专家ID → temp位置，初始前cache_num个专家设为-1（在cache中），其余映射到0~miss_num-1
            - policy_sorts: cache位置 → 优先级，长度为cache_num，初始值为0~cache_num-1
        """
        device = torch.device('cuda')
        miss_num = total_experts - cache_num

        # cache_maps: [total_experts]，前cache_num个在cache中
        cache_map = torch.full((total_experts,), -1, dtype=torch.int32, device=device)
        if cache_num > 0:
            cache_map[:cache_num] = torch.arange(cache_num, dtype=torch.int32, device=device)
        self.cache_maps[layer_id] = cache_map

        # ✅ miss_maps: [total_experts]，不在cache中的都在temp中
        miss_map = torch.full((total_experts,), -1, dtype=torch.int32, device=device)
        if miss_num > 0:
            # 对于不在cache中的专家（从cache_num开始），映射到0~miss_num-1
            miss_map[cache_num:] = torch.arange(miss_num, dtype=torch.int32, device=device)
        self.miss_maps[layer_id] = miss_map

        # policy_sorts: [cache_num]，初始优先级顺序
        if cache_num > 0:
            policy_sort = torch.arange(cache_num, dtype=torch.int32, device=device)
        else:
            policy_sort = torch.empty(0, dtype=torch.int32, device=device)
        self.policy_sorts[layer_id] = policy_sort

    # ==================== 初始化方法 ====================
    def setup_layer_cache(self, layer: nn.Module) -> None:
        """
        初始化 Layer 的 Offload 配置

        Args:
            layer: MoE 层实例
            layer_id: 层唯一标识
            num_cache_experts: GPU 缓存的专家数量
            total_experts: 总专家数
            moe_method: Fp8MoEMethod 实例
            buffer_idx: 手动指定 miss buffer 索引（0 或 1）
        """
        layer_id = layer.layer_idx
        buffer_idx = layer_id % 2

        if layer_id in self._processed_layers:
            warnings.warn(f"Layer {layer_id} 已处理，跳过")
            return

        if not getattr(layer, 'moe_offload', False):
            warnings.warn(f"Layer {layer_id} 未开启 offloading，跳过")
            return

        # 1. 保存 moe_method 引用
        #self._moe_method_refs[layer_id] = moe_method

        # 2. 保存 CPU 权重
        weight_names = self._get_weight_names(layer)
        cpu_state_dict = {}
        for name in weight_names:
            param = getattr(layer, name)
            if name == 'w13_weight' or name == 'w2_weight':
                M = param.data.shape[0]
                N = param.data.shape[1]
                K = param.data.shape[2]
                param.data = param.data.reshape(M,N//32,32,K//32,16,2).permute(0,1,3,4,2,5).contiguous().pin_memory()
                cpu_state_dict[name] = param.data
            else:
                cpu_state_dict[name] = param.data

        self.cpu_weights[layer_id] = cpu_state_dict

        # 3. 分配 Miss Buffer 索引
        if buffer_idx is None:
            used_indices = set(self._layer_to_buffer_idx.values())
            available = {0, 1} - used_indices
            if not available:
                raise RuntimeError("Miss Expert Buffer 已满（最多支持2个layer）")
            buffer_idx = min(available)

        self._layer_to_buffer_idx[layer_id] = buffer_idx

        # 4. 创建 Miss Buffer
        self._create_temp_layer(layer_id, weight_names)

        # 5. 创建 GPU 缓存
        self._create_gpu_cache(layer, layer_id, self.cache_expert_num, weight_names)

        # 6. 记录配置
        self.layer_configs[layer_id] = {
            'num_cache_experts': self.cache_expert_num,
            'total_experts': self.total_expert_num,
            'buffer_idx': buffer_idx,
            'weight_names': weight_names,
            'intermediate_size': layer.intermediate_size_per_partition,
            'hidden_size': layer.hidden_size,
            'dtype': layer.w13_weight.dtype,
            '_layer_ref': layer
        }

        # 7. 在 layer 上保存引用
        layer.cpu_offload_layer_id = layer_id
        layer.cpu_offload_manager = self
        '''
        if layer.expert_map is None:
            layer.expert_map = torch.arange(0, self.total_expert_num, dtype=torch.int32, device='cuda')
        layer.expert_map[self.cache_expert_num:] = -1
        '''
        if layer._expert_map is None:
            layer._expert_map = torch.arange(0, self.total_expert_num, dtype=torch.int32, device='cuda')
        layer._expert_map[self.cache_expert_num:] = -1

        self._processed_layers.add(layer_id)


        self.setup_cache_maps(layer_id=layer_id, total_experts=self.total_expert_num, cache_num=self.cache_expert_num)

        self.engine.create_layer(
            self.cpu_weights[layer_id]['w13_weight'],  # intptr_t gateUpWeights
            self.cpu_weights[layer_id]['w2_weight'],  # intptr_t downWeights
            self.cpu_weights[layer_id]['w13_weight_scale_inv'],  # intptr_t gateUpScales
            self.cpu_weights[layer_id]['w2_weight_scale_inv'],  # intptr_t downScales
            layer_id,
        )

        self._preload_experts(layer_id, self.cache_expert_num)

        if self.tp_rank == 0:
            print(f"✓ Layer {layer_id}: Miss Buffer={buffer_idx}, "
                  f"GPU Cache={self.cache_expert_num}/{self.total_expert_num} experts")

    def _create_temp_layer(
            self,
            layer_id: int,
            weight_names: List[str],
    ) -> None:
        """
        创建 Temp Layer（全局仅2个）
        ✅ 所有参数（权重+缩放系数）都预分配空tensor
        ✅ 后续由GPU kernel填充实际数据
        """
        buffer_idx = layer_id % 2
        if self.temp_layer[buffer_idx] is not None:
            return

        miss_num_experts = self.total_expert_num - self.cache_expert_num
        if miss_num_experts <= 0:
            self.temp_layer[buffer_idx] = None
            return

        temp_mod = nn.Module()
        device = torch.device('cuda')

        for name in weight_names:
            ref_shape = self.cpu_weights[layer_id][name].shape

            new_shape = (miss_num_experts,) + ref_shape[1:]
            if name == 'w13_weight' or name == 'w2_weight':
                N = new_shape[1] * 32
                K = new_shape[2] * 32
                new_shape = (miss_num_experts,N,K)

            # 注册到 temp layer
            temp_mod.register_parameter(
                name,
                nn.Parameter(
                    torch.zeros(new_shape, dtype=self.cpu_weights[layer_id][name].dtype, device=device),
                    requires_grad=False
                )
            )

        # ✅ quant_config 引用 temp_layer 中的空tensor
        temp_mod.quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=(temp_mod.w13_weight_scale_inv if self.block_quant else temp_mod.w13_weight_scale),
            w2_scale=(temp_mod.w2_weight_scale_inv if self.block_quant else temp_mod.w2_weight_scale),
            a1_scale=getattr(temp_mod, 'w13_input_scale', None),  # ✅ 安全获取
            a2_scale=getattr(temp_mod, 'w2_input_scale', None),
            block_shape=self.weight_block_size,
        )

        self.temp_layer[buffer_idx] = temp_mod
        if self.tp_rank == 0:
                print(f"✓ Temp Layer {buffer_idx} 预分配成功，{miss_num_experts} experts")

    def _get_temp_layer(self, buffer_idx: int) -> Optional[nn.Module]:
        """
        获取已创建的 temp_layer
        ✅ forward时仅获取，不创建
        """
        if buffer_idx not in [0, 1]:
            raise ValueError(f"buffer_idx must be 0 or 1, got {buffer_idx}")
        return self.temp_layer[buffer_idx]
    def _create_gpu_cache(
            self,
            layer: nn.Module,
            layer_id: int,
            num_cache_experts: int,
            weight_names: List[str],
    ) -> None:
        """创建 GPU 缓存并替换 layer 参数"""
        for name in weight_names:
            param = getattr(layer, name)
            orig_shape = param.shape
            if name == 'w13_weight' or name == 'w2_weight':
                N = orig_shape[1] * 32
                K = orig_shape[2] * 32
                new_shape = (num_cache_experts, N, K)
            else:
                new_shape = (num_cache_experts,) + orig_shape[1:]
            gpu_tensor = torch.empty(new_shape, dtype=param.dtype, device='cuda')
            param.data = gpu_tensor

    def _preload_experts(self, layer_id: int, num_experts: int) -> None:
        """初始加载前 N 个 expert 到 GPU 缓存"""
        cpu_state = self.cpu_weights[layer_id]
        config = self.layer_configs[layer_id]

        # ✅ 获取 layer 引用
        layer = config.get('_layer_ref')
        copy_map = self.cache_maps[layer_id]
        if not layer:
            return

        # ✅ 统一处理所有参数：w13_weight, w2_weight, scales, input_scales
        weight_names = config.get('weight_names', [])



        for name in weight_names:
            if name not in cpu_state:
                continue

            src = cpu_state[name]
            param = getattr(layer, name)
            # 计算要复制的数量
            num_to_copy = min(num_experts, src.shape[0])

            if name == 'w13_weight' or name == 'w2_weight':
                M = src.shape[0]
                N = src.shape[1] * 32
                K = src.shape[2] * 32
                tmp = src[:num_to_copy].permute(0,1,4,2,3,5).reshape(num_to_copy,N,K).contiguous().pin_memory()
            else:
                tmp = src

            if num_to_copy > 0:
                param.data[:num_to_copy].copy_(tmp[:num_to_copy])
                if name == 'w13_weight' or name == 'w2_weight':
                    del tmp
                    gc.collect() 


    def _get_weight_names(self, layer: nn.Module) -> List[str]:
        """自动检测所有相关权重名称"""
        names = []
        for name in ['w13_weight', 'w2_weight',
                     'w13_weight_scale', 'w2_weight_scale',
                     'w13_weight_scale_inv', 'w2_weight_scale_inv',
                     'w13_input_scale', 'w2_input_scale']:
            if hasattr(layer, name) and getattr(layer, name) is not None:
                names.append(name)
        return names

    def get_miss_buffer(self, buffer_idx: int) -> Dict[str, Tensor]:
        """通过 0 或 1 索引获取 Miss Expert Buffer"""
        if buffer_idx not in self.miss_buffers:
            raise KeyError(f"Miss buffer 索引 {buffer_idx} 未初始化")
        return self.miss_buffers[buffer_idx]

    def forward_dbo(
            self,
            hidden_states: Tensor,
            topk_weights: Tensor,
            topk_ids: Tensor,
            layer_id: int,
    ) -> [Tensor, Tensor]:
        """
        深度绑定优化 (DBO) 前向计算
        使用自定义 CPU 算子进行异步计算
        """

        # 1. 执行缓存策略
        cache_map = self.cache_maps.get(layer_id, [])
        config = self.layer_configs[layer_id]

        layer = config.get('_layer_ref')

        if not layer:
            raise RuntimeError(f"Layer {layer_id} 引用丢失")

        cpu_topk_ids, copy_map = torch.ops.moe_offload_ops.expert_cache_policy(
            topk_ids,
            self.cache_maps[layer_id],
            self.miss_maps[layer_id],
            self.policy_sorts[layer_id],
            self.engine.ptr()
        )

        if dbo_enabled():
            dbo_switch_to_comm_sync()

        # 3. 提交 CPU 任务
        torch.ops.moe_offload_ops.cpu_moe_submit(hidden_states, cpu_topk_ids, topk_weights, self.engine.ptr(), layer_id, 0)

        # 4. 更新专家缓存
        torch.ops.moe_offload_ops.update_expert_cache(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            copy_map,
            self.total_expert_num,
            layer_id,
            self.engine.ptr()
        )

        cpu_output = torch.zeros_like(hidden_states)
        cpu_output = torch.ops.moe_offload_ops.cpu_moe_sync(cpu_output, self.engine.ptr())

        # 5. 切换到另一个线程的计算，计算结束当前线程等待传输完成
        if dbo_enabled():
            dbo_yield_and_switch_from_comm_to_compute()

        return cpu_output, cache_map


    def forward_prefetch(
            self,
            hidden_states: Tensor,
            topk_weights: Tensor,
            topk_ids: Tensor,
            layer_id: int,
    ) -> [Tensor, Tensor]:
        """
        预取 (Prefetch) 模式前向计算
        使用 Miss Expert Buffer 进行 miss expert 计算
        """
        n_tok = hidden_states.shape[0]
        if n_tok > 8192:
            n_copy = 256
        elif n_tok >= 1024:
            n_copy = (n_tok // 1000) * 16 + 112
        else:
            n_copy = 80

        # ✅ 使用 layer_id % 2 获取 buffer 索引
        buffer_idx = layer_id % 2
        temp_layer = self._get_temp_layer(buffer_idx)
        cache_map = self.cache_maps.get(layer_id, [])
        config = self.layer_configs[layer_id]
        miss_map = self.miss_maps.get(layer_id, [])

        # 如果 buffer 为空，返回零张量
        if not temp_layer:
            return torch.zeros_like(hidden_states)

        # 为每个buffer创建独立的stream和事件
        if self.update_streams is None:
            self.update_streams = torch.cuda.Stream()

        if self.compute_event[buffer_idx] is None:
            self.compute_event[buffer_idx] = torch.cuda.Event()
            self.copy_event[buffer_idx] = torch.cuda.Event()

        current_stream = torch.cuda.current_stream()
        
        # 3. 当前层的update需要等待同buffer的前一个fused_experts完成
        self.update_streams.wait_event(self.compute_event[buffer_idx])

        # 1. 在专用stream中执行update_expert_cache
        with torch.cuda.stream(self.update_streams):
            torch.ops.moe_offload_ops.update_expert_cache(
                temp_layer.w13_weight,
                temp_layer.w2_weight,
                temp_layer.w13_weight_scale_inv,
                temp_layer.w2_weight_scale_inv,
                miss_map,
                n_copy,
                layer_id,
                self.engine.ptr()
            )
            # 记录update完成事件
            self.copy_event[buffer_idx].record(self.update_streams)

        # 2. fused_experts依赖update_expert_cache的执行结束
        current_stream.wait_event(self.copy_event[buffer_idx])

        # 4. 执行 fused_experts（使用 temp_layer 中的权重）
        # 注意：这里直接调用 fused_experts 函数，而不是 moe_method.apply()
        if n_copy < 256:
            map_modified = miss_map.clone()
            map_modified[n_copy:] = -1
            mask = (topk_ids > n_copy) & (miss_map[topk_ids] >= 0)
            cpu_topk_ids = torch.where(mask, topk_ids, -1)
            torch.ops.moe_offload_ops.cpu_moe_submit(hidden_states, cpu_topk_ids, topk_weights, self.engine.ptr(), layer_id, 0)

        from vllm.model_executor.layers.fused_moe import fused_experts

        miss_output = fused_experts(
            hidden_states=hidden_states,
            w1=temp_layer.w13_weight,  # 来自 miss buffer
            w2=temp_layer.w2_weight,  # 来自 miss buffer
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",  # 可从 layer 配置获取
            global_num_experts=config['total_experts'],
            apply_router_weight_on_input=False,
            expert_map=miss_map,  # 使用 layer 的 miss_map
            quant_config=temp_layer.quant_config,
            allow_deep_gemm=False,
        )

        cpu_output = torch.zeros_like(hidden_states)
        if n_copy < 256:
            cpu_output = torch.ops.moe_offload_ops.cpu_moe_sync(cpu_output, self.engine.ptr())
        # 记录fused_experts完成事件，供下一个同buffer层使用
        self.compute_event[buffer_idx].record(current_stream)

        return miss_output + cpu_output, cache_map


    def forward_offload(
            self,
            hidden_states: Tensor,
            topk_weights: Tensor,
            topk_ids: Tensor,
            layer_id: int,
    ) -> Tensor:
        """
        统一入口：根据配置选择 DBO 或 Prefetch 模式
        """
        if self.tp_rank == 1 and layer_id == 4:
            print(f"dbo_enable():{dbo_enabled()} and ntok = {hidden_states.shape[0]}")
        if dbo_enabled() or hidden_states.shape[0] == 1:
            return self.forward_dbo(hidden_states, topk_weights, topk_ids, layer_id)
        else:
            return self.forward_prefetch(hidden_states, topk_weights, topk_ids, layer_id)
