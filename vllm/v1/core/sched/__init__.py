# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM V1 调度器模块。

本模块实现 vLLM V1 的调度器功能，负责：
- 调度请求的执行顺序（FCFS 或优先级）
- 分配 KV 缓存块和 token 预算
- 处理推测解码、前缀缓存和多模态输入
- 管理编码器缓存和 LoRA 适配器

主要类：
- Scheduler: 主调度器实现
- AsyncScheduler: 异步调度器实现
- SchedulerInterface: 调度器抽象接口

主要数据类：
- SchedulerOutput: 调度器输出
- NewRequestData: 新请求数据
- CachedRequestData: 缓存请求数据
- GrammarOutput: 文法输出
"""