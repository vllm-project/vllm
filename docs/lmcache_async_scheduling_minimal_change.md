# LMCache 异步调度适配 - 最小修改方案

## 问题分析

### 能否只修改 lmcache 而不修改 vllm？

**结论：不能完全避免修改 vllm，但可以做到最小化修改。**

### 原因分析

1. **信息传递链断裂**
   ```
   Scheduler 侧:
   ├── scheduler.requests[req_id].num_output_placeholders  ✅ 有这个信息
   ├── scheduler_output  ❌ 但 SchedulerOutput 不包含 placeholders
   └── connector.build_connector_meta(scheduler_output)  ❌ lmcache 拿不到
   
   Worker 侧:
   ├── forward_context  ❌ 不包含 placeholders 信息
   └── connector.start_load_kv(forward_context)  ❌ lmcache 拿不到
   ```

2. **现有接口限制**
   - `build_connector_meta()` 只接收 `SchedulerOutput`
   - `SchedulerOutput` 不包含 `num_output_placeholders` 字段
   - LMCache connector 无法直接访问 `scheduler.requests`

3. **文件归属**
   - ✅ 可以修改：`lmcache` 仓库中的 `vllm_v1_adapter.py`
   - ❌ 需要避免：`vllm` 仓库中的所有文件

## 方案对比

### 方案 1: 完全不修改 vllm（不可行）
**问题**：无法获取 `num_output_placeholders` 信息

### 方案 2: 最小化修改 vllm（推荐）
**只需修改 3 处，约 20 行代码**

### 方案 3: 扩展性修改（原设计文档方案）
**需要修改多处，添加新的元数据字段**

## 推荐方案：最小化修改

### 核心思路

**在 `SchedulerOutput` 中传递 placeholders 信息，无需新增元数据结构。**

### 需要修改的地方（仅 3 处）

#### 修改 1: 扩展 SchedulerOutput（约 5 行）

**文件**: `vllm/v1/core/sched/output.py`

```python
@dataclass
class SchedulerOutput:
    # ... 现有字段 ...
    
    # 新增：用于 async scheduling 的 placeholder 信息
    # req_id -> num_output_placeholders
    # 只在 async scheduling 模式下有值，同步模式为空字典
    num_output_placeholders: dict[str, int] = field(default_factory=dict)
```

**影响范围**：
- 仅添加一个字段，默认值为空字典
- 向后兼容：同步 scheduler 不填充此字段
- 性能开销：可忽略（仅一个 dict 引用）

#### 修改 2: AsyncScheduler 填充 placeholders（约 10 行）

**文件**: `vllm/v1/core/sched/async_scheduler.py`

```python
class AsyncScheduler(Scheduler):

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        
        # 填充 num_output_placeholders 信息供 LMCache 使用
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            scheduler_output.num_output_placeholders[req_id] = (
                request.num_output_placeholders
            )
            
            if (request.num_computed_tokens == request.num_tokens +
                    request.num_output_placeholders):
                # The request will generate a new token in this scheduling step.
                # TODO(woosuk): Support speculative decoding.
                request.num_output_placeholders += 1
```

**影响范围**：
- 仅在 AsyncScheduler 中添加逻辑
- 同步 Scheduler 不受影响
- 性能开销：O(num_requests) 的字典填充，可忽略

#### 修改 3: LMCache 访问 placeholders（lmcache 仓库）

**文件**: `lmcache/integration/vllm/vllm_v1_adapter.py`

```python
class LMCacheConnectorV1Impl:
    
    def build_connector_meta(self, scheduler_output):
        """构建元数据，使用 scheduler_output 中的 placeholders 信息"""
        meta = self._build_base_meta(scheduler_output)
        
        # 计算每个请求的实际已计算 token 数
        # real_computed = num_computed - num_output_placeholders
        meta.real_computed_tokens = {}
        
        # 从 SchedulerOutput 获取 placeholders 信息
        placeholders = getattr(scheduler_output, 'num_output_placeholders', {})
        
        for req_id in scheduler_output.num_scheduled_tokens:
            # 从 scheduled_cached_reqs 获取 num_computed_tokens
            num_computed = self._get_num_computed_tokens(
                scheduler_output, req_id)
            
            # 获取 placeholders（如果是同步模式则为 0）
            num_placeholders = placeholders.get(req_id, 0)
            
            # 计算实际已计算的 tokens
            meta.real_computed_tokens[req_id] = num_computed - num_placeholders
        
        return meta
    
    def _get_num_computed_tokens(self, scheduler_output, req_id):
        """从 SchedulerOutput 中提取 num_computed_tokens"""
        # 检查是否在 scheduled_new_reqs 中
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id == req_id:
                return new_req.num_computed_tokens
        
        # 检查是否在 scheduled_cached_reqs 中
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if req_id in cached_reqs.req_ids:
            idx = cached_reqs.req_ids.index(req_id)
            return cached_reqs.num_computed_tokens[idx]
        
        return 0
    
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """保存 KV cache，使用 real_computed_tokens 作为边界"""
        # 从 metadata 获取实际计算边界
        meta = self._get_current_metadata()
        if meta and hasattr(meta, 'real_computed_tokens'):
            real_computed = meta.real_computed_tokens
            self._save_with_boundaries(layer_name, kv_layer, 
                                      attn_metadata, real_computed)
        else:
            # 同步模式，使用全部 computed tokens
            self._save_all_computed(layer_name, kv_layer, attn_metadata)
```

**影响范围**：
- 仅修改 lmcache 仓库代码
- 使用 `getattr()` 保证向后兼容
- 如果 vllm 版本较老，placeholders 字段不存在，会回退到同步模式

### 总结

**vllm 侧修改**：
1. `vllm/v1/core/sched/output.py` - 添加 1 个字段（1 行）
2. `vllm/v1/core/sched/async_scheduler.py` - 填充字段（约 10 行）

**lmcache 侧修改**：
3. `lmcache/integration/vllm/vllm_v1_adapter.py` - 使用新字段（约 30 行）

**总计**：约 50 行代码修改


### 选择：最小化修改方案

**原因**：
1. ✅ **修改量小**：vllm 侧仅 2 处修改，约 15 行
2. ✅ **清晰明确**：信息传递路径清楚
3. ✅ **向后兼容**：对同步模式零影响
4. ✅ **易于维护**：未来 vllm 更新不易失效
5. ✅ **性能优秀**：几乎零开销

**修改清单**：
```
vllm 仓库:
├── vllm/v1/core/sched/output.py          (+1 字段)
└── vllm/v1/core/sched/async_scheduler.py (+10 行逻辑)

lmcache 仓库:
└── integration/vllm/vllm_v1_adapter.py   (+30 行逻辑)
```

**实施步骤**：
1. 提交 PR 到 vllm，添加 `num_output_placeholders` 字段到 `SchedulerOutput`
2. 在 lmcache 中实现适配逻辑，使用 `getattr()` 保证兼容性
3. 添加版本检测，提示用户升级 vllm 版本

### PR 建议

**vllm PR 标题**：
```
[Core] Add num_output_placeholders to SchedulerOutput for async scheduling
```

**PR 描述**：
```
This PR adds a new field `num_output_placeholders` to `SchedulerOutput` to 
support KV cache connectors (e.g., LMCache) in async scheduling mode.

- Adds `num_output_placeholders: dict[str, int]` to SchedulerOutput
- AsyncScheduler populates this field with placeholder information
- Backward compatible: field is empty dict for sync scheduler
- Performance impact: negligible (single dict reference)
- Enables external KV cache systems to correctly handle async scheduling

Related: <LMCache issue link>
```

这样可以让 vllm 社区理解修改的必要性和最小化影响。

## 版本兼容策略

在 lmcache 中：

```python
# lmcache/integration/vllm/version_compat.py

def check_async_scheduling_support(scheduler_output):
    """检查 vllm 版本是否支持 async scheduling with LMCache"""
    if not hasattr(scheduler_output, 'num_output_placeholders'):
        logger.warning(
            "Your vLLM version does not support async scheduling with LMCache. "
            "Please upgrade to vLLM >= X.Y.Z for async scheduling support. "
            "Falling back to synchronous mode."
        )
        return False
    return True
```

这样即使用户使用旧版本 vllm，也能正常工作（降级到同步模式）。

