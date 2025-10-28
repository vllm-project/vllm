# LMCache 异步调度适配设计文档

## 1. 背景与目标

### 1.1 当前状况
- **Async Scheduling**: vLLM v1 支持异步调度（AsyncScheduler），允许在 token 实际生成之前预先调度和分配 KV cache 空间
- **LMCache 集成**: LMCache 通过 `LMCacheConnectorV1` 与 vLLM 集成，支持 KV cache 的外部存储、共享和传输
- **兼容性问题**: 目前 LMCache 主要针对同步调度（Scheduler）设计，尚未完全适配异步调度机制

### 1.2 设计目标
1. 使 LMCache 能够正确处理异步调度中的 KV cache 操作
2. 确保 `num_output_placeholders` 机制与 LMCache 的缓存逻辑兼容
3. 保持向后兼容性，不影响现有的同步调度功能
4. 优化性能，避免不必要的重复缓存操作

## 2. 核心概念分析

### 2.1 AsyncScheduler 的关键机制

```
Request Timeline in AsyncScheduler:
┌─────────────────────────────────────────────────────────────┐
│ Time    │ Action                │ num_computed │ placeholders│
├─────────┼───────────────────────┼──────────────┼─────────────┤
│ t0      │ Schedule (prefill)    │ 100          │ 0           │
│ t1      │ Schedule (decode)     │ 101          │ 1           │  ← placeholder++
│ t2      │ Token generated       │ 101          │ 0           │  ← placeholder--
│         │ cache_blocks(100)     │              │             │  ← Cache real tokens
│ t3      │ Schedule (decode)     │ 102          │ 1           │
│ t4      │ Token generated       │ 102          │ 0           │
│         │ cache_blocks(101)     │              │             │
└─────────────────────────────────────────────────────────────┘
```

**关键点**:
- `num_output_placeholders`: 追踪已调度但未实际生成的 token 数量
- **增加时机**: 在 `_update_after_schedule()` 中，当请求将生成新 token 时
- **减少时机**: 在 `_update_request_with_output()` 中，当 token 实际生成时
- **缓存时机**: 在 `_update_request_with_output()` 中，使用 `num_computed_tokens - num_output_placeholders` 作为参数调用 `cache_blocks()`

### 2.2 LMCache 的缓存机制

LMCache 在以下位置进行 KV cache 操作：

1. **Worker 侧** (forward pass 期间):
   - `start_load_kv()`: 开始异步加载 KV cache
   - `wait_for_layer_load()`: 等待某层 KV cache 加载完成
   - `save_kv_layer()`: 异步保存某层 KV cache
   - `wait_for_save()`: 等待所有保存操作完成

2. **Scheduler 侧** (调度期间):
   - `get_num_new_matched_tokens()`: 获取可从外部缓存加载的 token 数量
   - `update_state_after_alloc()`: 分配 blocks 后更新状态
   - `build_connector_meta()`: 构建连接器元数据
   - `request_finished()`: 请求完成时的清理

### 2.3 兼容性问题分析

#### 问题 1: 缓存边界不一致
在 AsyncScheduler 中:
```python
# AsyncScheduler._update_request_with_output (line 43-46)
if status_before_update == RequestStatus.RUNNING:
    self.kv_cache_manager.cache_blocks(
        request,
        request.num_computed_tokens - request.num_output_placeholders)
```

- 缓存的是 "已真正计算完成的 tokens"，而不是所有 `num_computed_tokens`
- LMCache 需要知道实际应该缓存到哪个 token 位置

#### 问题 2: Prefix Caching 与 Placeholders
- AsyncScheduler 中 `num_computed_tokens` 包含了 placeholder tokens
- LMCache 在查询缓存时需要区分"真正计算的"和"预分配的"tokens

#### 问题 3: 元数据传递
- LMCache 的 `build_connector_meta()` 需要知道每个请求的实际缓存位置
- 当前接口可能没有传递 `num_output_placeholders` 信息

## 3. 设计方案

### 3.1 方案概览

**核心思路**: 让 LMCache 感知 async scheduling 的 placeholder 机制，在所有相关操作中使用"实际已计算 tokens 数"而不是"调度 tokens 数"。

### 3.2 修改点清单

#### 修改 1: 扩展 Request 对象信息传递

在 LMCache connector 的所有接口中，需要能够获取到请求的真实计算状态：

```python
def get_real_computed_tokens(request: Request) -> int:
    """获取请求实际已计算的 token 数量（排除 placeholders）"""
    return request.num_computed_tokens - request.num_output_placeholders
```

#### 修改 2: 更新 `cache_blocks()` 调用逻辑

在 `AsyncScheduler._update_request_with_output()` 中:

```python
# 当前实现 (line 43-46)
if status_before_update == RequestStatus.RUNNING:
    self.kv_cache_manager.cache_blocks(
        request,
        request.num_computed_tokens - request.num_output_placeholders)

# 如果使用 LMCache，需要确保 LMCache 也知道这个边界
# LMCache connector 应该在 Worker 侧的 save_kv_layer() 中使用这个信息
```

#### 修改 3: LMCache Connector 适配

##### 3.2.1 Scheduler 侧修改

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

```python
# 新增方法：获取请求的实际计算 token 数
def get_real_computed_tokens(self, request: Request) -> int:
    """
    获取请求实际已计算的 token 数量（排除 async scheduling placeholders）
    
    这个方法在 async scheduling 模式下返回：
        num_computed_tokens - num_output_placeholders
    在同步模式下返回：
        num_computed_tokens
    """
    return self._lmcache_engine.get_real_computed_tokens(request)

# 修改 build_connector_meta() 
def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
    """
    构建连接器元数据，需要传递每个请求的实际计算位置
    """
    # 需要为每个请求添加 real_computed_tokens 信息
    return self._lmcache_engine.build_connector_meta(
        scheduler_output, 
        include_placeholder_info=True)
```

##### 3.2.2 Worker 侧修改

**在 save_kv_layer() 时使用正确的边界**:

```python
def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                  attn_metadata: "AttentionMetadata", **kwargs) -> None:
    """
    保存 KV cache 时，只保存实际计算的部分，不包括 placeholder tokens
    """
    # 从 attn_metadata 或 kwargs 中获取每个请求的实际计算边界
    real_computed_info = kwargs.get('real_computed_tokens', {})
    
    self._lmcache_engine.save_kv_layer(
        layer_name, kv_layer, attn_metadata,
        real_computed_tokens=real_computed_info,
        **kwargs)
```

##### 3.2.3 元数据传递链

```
AsyncScheduler._update_request_with_output()
    ↓ 计算 real_computed = num_computed_tokens - num_output_placeholders
    ↓
KVCacheManager.cache_blocks(request, real_computed)
    ↓
Scheduler → build_connector_meta() 
    ↓ 将 real_computed 信息编码到 metadata
    ↓
Worker → start_load_kv() / save_kv_layer()
    ↓ 从 metadata 获取 real_computed 信息
    ↓
LMCache → 使用 real_computed 作为缓存边界
```

### 3.3 详细接口设计

#### 3.3.1 修改 KVConnectorMetadata

```python
@dataclass
class KVConnectorMetadata:
    """
    包含 KV transfer 所需的所有元数据
    """
    # ... 现有字段 ...
    
    # 新增：每个请求的实际计算 token 数（用于 async scheduling）
    real_computed_tokens: dict[str, int] = field(default_factory=dict)
    """
    request_id -> 实际已计算的 token 数（排除 placeholders）
    在 async scheduling 模式下，这个值 <= num_computed_tokens
    在同步模式下，这个值 == num_computed_tokens
    """
```

#### 3.3.2 修改 LMCacheConnectorV1Impl

**在 lmcache 侧的适配器实现** (假设在 `lmcache` 仓库中):

```python
class LMCacheConnectorV1Impl:
    def __init__(self, vllm_config, role, connector):
        self.vllm_config = vllm_config
        self.role = role
        self.connector = connector
        
        # 检测是否启用 async scheduling
        self.is_async_scheduling = self._detect_async_scheduling(vllm_config)
        
        # 追踪每个请求的实际计算边界
        self.request_real_computed: dict[str, int] = {}
    
    def _detect_async_scheduling(self, vllm_config) -> bool:
        """检测是否启用 async scheduling"""
        # 可以通过 config 或环境变量检测
        return getattr(vllm_config.scheduler_config, 
                      'use_async_scheduling', False)
    
    def build_connector_meta(self, scheduler_output, 
                           include_placeholder_info=False):
        """构建元数据，包含 async scheduling 信息"""
        meta = self._build_base_meta(scheduler_output)
        
        if include_placeholder_info and self.is_async_scheduling:
            # 从 scheduler_output 提取每个请求的实际计算位置
            meta.real_computed_tokens = {}
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.connector.get_request(req_id)
                if request:
                    meta.real_computed_tokens[req_id] = (
                        request.num_computed_tokens - 
                        request.num_output_placeholders
                    )
        
        return meta
    
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata,
                     real_computed_tokens=None, **kwargs):
        """
        保存 KV cache，考虑 async scheduling 的边界
        
        Args:
            real_computed_tokens: dict[request_id -> actual_computed_position]
                                 用于 async scheduling
        """
        if self.is_async_scheduling and real_computed_tokens:
            # 使用 real_computed_tokens 作为保存边界
            self._save_with_boundaries(layer_name, kv_layer, 
                                      attn_metadata, 
                                      real_computed_tokens)
        else:
            # 同步模式，使用全部 computed tokens
            self._save_all_computed(layer_name, kv_layer, attn_metadata)
```

### 3.4 执行流程图

#### 3.4.1 Async Scheduling + LMCache 的完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     调度阶段 (Scheduler)                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. AsyncScheduler.schedule()                                    │
│    - 调度请求，分配 tokens                                        │
│    - 调用 allocate_slots() 分配 KV cache blocks                  │
│    - 生成 SchedulerOutput                                        │
│                                                                  │
│ 2. AsyncScheduler._update_after_schedule()                      │
│    - 更新 num_computed_tokens += num_scheduled_tokens            │
│    - 如果将生成新 token: num_output_placeholders += 1            │
│                                                                  │
│ 3. LMCache: build_connector_meta()                              │
│    - 为每个请求计算: real_computed = computed - placeholders     │
│    - 将 real_computed_tokens 编码到 metadata                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     执行阶段 (Worker)                            │
├─────────────────────────────────────────────────────────────────┤
│ 4. ModelRunner.execute_model()                                  │
│    - 从 metadata 获取 real_computed_tokens                       │
│                                                                  │
│ 5. LMCache: start_load_kv()                                     │
│    - 使用 real_computed 作为已缓存边界                            │
│    - 异步加载需要的 KV cache                                     │
│                                                                  │
│ 6. Forward pass                                                 │
│    - 每层调用 wait_for_layer_load() 确保数据就绪                  │
│    - 计算 attention                                             │
│    - 每层调用 save_kv_layer() 保存新计算的 KV                     │
│      * 只保存 [old_real_computed, new_real_computed) 范围         │
│      * 不保存 placeholder 对应的 KV                               │
│                                                                  │
│ 7. LMCache: wait_for_save()                                     │
│    - 等待所有异步保存完成                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     更新阶段 (Scheduler)                         │
├─────────────────────────────────────────────────────────────────┤
│ 8. AsyncScheduler.update_from_output()                          │
│    - 接收 ModelRunnerOutput (包含生成的 tokens)                   │
│                                                                  │
│ 9. AsyncScheduler._update_request_with_output()                 │
│    - append_output_token_ids()                                  │
│    - num_output_placeholders -= len(new_token_ids)              │
│    - 调用 cache_blocks(request, computed - placeholders)         │
│      * 这会更新 prefix cache 的 hash table                       │
│      * LMCache 的实际缓存已在 Worker 侧完成                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 关键同步点

```
Timeline:
t0: Schedule → num_computed=101, placeholders=1, real=100
    ↓ (metadata: real_computed_tokens[req_id] = 100)
t1: Worker starts → Load KV up to token 100
    ↓
t2: Forward pass → Generate token at position 100
    ↓
t3: Worker saves → Save KV for token 100 only (not 101)
    ↓
t4: Update → placeholders=0, real=101
    ↓
t5: cache_blocks(request, 101) → Update prefix cache
```

### 3.5 边界情况处理

#### 情况 1: 多个 Placeholders
```python
# 如果一次调度多个 decode steps (pipeline parallelism)
num_output_placeholders = 3
num_computed_tokens = 103

# LMCache 应该只缓存到 token 100
real_computed = 103 - 3 = 100
```

#### 情况 2: Chunked Prefill + Async Scheduling
```python
# Prefill 阶段不应该有 placeholders
assert num_output_placeholders == 0 during prefill

# 只在 decode 阶段使用 placeholders
```

#### 情况 3: Speculative Decoding
```python
# Spec tokens 和 placeholders 是不同的概念
# Spec tokens: 可能被拒绝的推测 tokens
# Placeholders: 还未生成的 future tokens

# LMCache 应该：
# 1. 不缓存 placeholder tokens
# 2. 只缓存被接受的 spec tokens
```

#### 情况 4: Preemption 和 Resumption
```python
# Preemption 时:
# - num_computed_tokens = 0
# - num_output_placeholders = 0
# - LMCache 应该保留已保存的缓存

# Resumption 时:
# - 从 LMCache 重新加载缓存
# - 使用 get_num_new_matched_tokens() 获取可重用的 tokens
```

## 4. 实现步骤

### Phase 1: 基础设施准备
1. ✅ 分析现有代码，理解 AsyncScheduler 和 LMCache 的交互点
2. 📝 设计接口扩展方案
3. 🔧 在 `KVConnectorMetadata` 中添加 `real_computed_tokens` 字段

### Phase 2: Scheduler 侧修改
1. 修改 `LMCacheConnectorV1.build_connector_meta()`
   - 添加 `include_placeholder_info` 参数
   - 计算并填充 `real_computed_tokens`
2. 确保 `AsyncScheduler` 的 `cache_blocks()` 调用正确传递信息

### Phase 3: Worker 侧修改
1. 修改 `start_load_kv()` 以使用 `real_computed_tokens`
2. 修改 `save_kv_layer()` 以只保存实际计算的 tokens
3. 添加日志和断言，验证边界正确性

### Phase 4: LMCache 适配器实现
1. 在 `lmcache` 仓库中实现 `LMCacheConnectorV1Impl` 的相关方法
2. 添加 `is_async_scheduling` 检测逻辑
3. 实现 `_save_with_boundaries()` 方法

### Phase 5: 测试验证
1. 单元测试：测试 placeholder 边界计算
2. 集成测试：Async scheduling + LMCache 端到端测试
3. 性能测试：对比同步 vs 异步的 throughput
4. 正确性测试：验证缓存内容一致性

### Phase 6: 文档和发布
1. 更新 LMCache 使用文档
2. 添加 async scheduling + LMCache 的示例
3. 发布 release notes

## 5. 兼容性考虑

### 5.1 向后兼容性
- 同步 Scheduler 不受影响（`num_output_placeholders` 始终为 0）
- 现有 LMCache 配置继续工作
- 如果不使用 async scheduling，新字段被忽略

### 5.2 配置选项
```yaml
# vllm config
scheduler:
  use_async_scheduling: true  # 启用异步调度

kv_transfer:
  kv_connector: "LMCacheConnectorV1"
  kv_connector_extra_config:
    # LMCache 会自动检测 async scheduling
    # 无需额外配置
```

### 5.3 性能影响
- **额外开销**: 传递 `real_computed_tokens` 字典（O(num_requests)）
- **内存**: 每个请求额外 8 bytes（int64）
- **计算**: 每次调度多一次减法操作
- **预期影响**: < 1% overhead

## 6. 测试策略

### 6.1 单元测试
```python
# test_async_scheduler_lmcache.py

def test_placeholder_boundary():
    """测试 placeholder 边界计算"""
    request = create_test_request()
    request.num_computed_tokens = 105
    request.num_output_placeholders = 3
    
    real_computed = get_real_computed_tokens(request)
    assert real_computed == 102

def test_metadata_generation():
    """测试元数据生成包含正确的边界信息"""
    scheduler_output = create_test_scheduler_output()
    connector = LMCacheConnectorV1(config, role)
    
    meta = connector.build_connector_meta(scheduler_output)
    assert "real_computed_tokens" in meta
    assert meta.real_computed_tokens[req_id] == expected_value
```

### 6.2 集成测试
```python
def test_async_scheduling_with_lmcache():
    """端到端测试：async scheduling + LMCache"""
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        kv_connector="LMCacheConnectorV1",
        enable_async_scheduling=True,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # 验证输出正确性
    assert_outputs_valid(outputs)
    
    # 验证缓存被正确使用
    cache_stats = llm.get_cache_stats()
    assert cache_stats.num_cached_tokens > 0
```

### 6.3 压力测试
```python
def test_high_concurrency():
    """高并发场景下的测试"""
    llm = LLM(model="...", kv_connector="LMCacheConnectorV1",
             enable_async_scheduling=True)
    
    # 1000 个并发请求
    prompts = generate_random_prompts(1000)
    outputs = llm.generate(prompts)
    
    assert len(outputs) == 1000
    assert_no_cache_corruption()
```

## 7. 风险与缓解

### 风险 1: 边界计算错误
- **影响**: 缓存位置错误，导致生成错误的 tokens
- **缓解**: 
  - 添加大量断言检查边界合法性
  - 在 debug 模式下记录详细日志
  - 单元测试覆盖所有边界情况

### 风险 2: 性能退化
- **影响**: Async scheduling 的性能优势被抵消
- **缓解**:
  - 性能基准测试对比
  - Profile 找出瓶颈
  - 优化元数据传递（使用 shared memory）

### 风险 3: 与其他特性冲突
- **影响**: Speculative decoding、Pipeline parallelism 等特性不兼容
- **缓解**:
  - 逐个特性进行兼容性测试
  - 文档明确列出支持的特性组合
  - 不支持的组合给出清晰的错误提示

## 8. 未来扩展

### 8.1 优化方向
1. **Zero-copy 元数据传递**: 使用 shared memory 减少开销
2. **Adaptive caching**: 根据 placeholder 数量动态调整缓存策略
3. **Speculative caching**: 预测性地缓存 future tokens

### 8.2 其他 KV Connector 适配
相同的设计模式可应用于其他 KV connectors:
- NIXL Connector
- P2P NCCL Connector
- Custom connectors

## 9. 总结

### 9.1 核心设计原则
1. **最小侵入**: 只修改必要的接口，保持现有逻辑不变
2. **信息透明**: 在元数据中显式传递 async scheduling 信息
3. **边界清晰**: 明确区分"调度的"和"实际计算的"tokens
4. **向后兼容**: 对同步调度零影响

### 9.2 关键技术点
- 在所有 KV cache 操作中使用 `num_computed_tokens - num_output_placeholders`
- 通过 `KVConnectorMetadata.real_computed_tokens` 传递边界信息
- Worker 侧只保存实际计算的 tokens，忽略 placeholders
- Scheduler 侧在 token 真正生成后更新 prefix cache

### 9.3 预期收益
- ✅ LMCache 完全支持 async scheduling
- ✅ 保持 async scheduling 的性能优势
- ✅ 支持 disaggregated prefill + async decode
- ✅ 为其他 connector 提供参考实现

---

## 附录

### A. 相关文件清单
```
vllm/v1/core/sched/
  ├── scheduler.py              # 基础 Scheduler
  ├── async_scheduler.py        # AsyncScheduler (需要修改)
  └── interface.py              # Scheduler 接口

vllm/distributed/kv_transfer/kv_connector/v1/
  ├── base.py                   # KVConnectorBase_V1 (需要修改 metadata)
  └── lmcache_connector.py      # LMCacheConnectorV1 (需要修改)

vllm/v1/core/
  └── kv_cache_manager.py       # KVCacheManager

lmcache/ (外部仓库)
  └── integration/vllm/
      └── vllm_v1_adapter.py    # LMCacheConnectorV1Impl (需要修改)
```

### B. 配置示例
```python
# 启用 async scheduling + LMCache
from vllm import LLM, SamplingParams
import os

# LMCache 环境变量
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",  # 既生产也消费
    # Async scheduling 相关
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
)

outputs = llm.generate(prompts, sampling_params)
```

### C. 参考资料
- [vLLM v1 Architecture](https://github.com/vllm-project/vllm/tree/main/vllm/v1)
- [LMCache Documentation](https://docs.lmcache.ai/)
- [AsyncScheduler Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/async_scheduler.py)
- [KV Connector Interface](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

