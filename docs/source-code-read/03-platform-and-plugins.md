# 第3章：平台抽象与插件系统

> 一句话：Platform 接口将硬件差异（CUDA/ROCm/CPU/TPU）隐藏在统一抽象后面，插件系统则允许第三方扩展模型、平台和 API 端点。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/platforms/interface.py` | ~1301 | Platform 抽象基类：定义 device_type、attention backend 选择等接口 |
| `vllm/platforms/cuda.py` | ~1020 | CudaPlatform：CUDA 平台实现，包含 compute capability 检测 |
| `vllm/platforms/__init__.py` | ~304 | 平台自动检测与全局 `current_platform` 单例 |
| `vllm/plugins/` | ~200 | 插件加载：基于 entry_points 的发现与注册机制 |
| `vllm/envs.py` | ~400+ | 环境变量定义：VLLM_* 全局开关汇总 |

## 关键问题（带着这些问题读）

1. `current_platform` 是如何在启动时确定的？平台检测的优先级顺序是什么？
2. Platform 接口中 `check_and_update_config()` 在什么时机被调用？它能修改哪些配置？
3. 第三方硬件如何通过 platform plugin 接入 vLLM？需要实现哪些最小接口？
4. `VLLM_*` 环境变量与 VllmConfig 之间是什么关系？谁的优先级更高？

## 调用链概览

```
vllm 启动
  → plugins/__init__.py: load_plugins_by_group("vllm.platform_plugins")
    → 发现并执行注册函数
  → platforms/__init__.py: 探测当前硬件 → 实例化 current_platform
  → VllmConfig 初始化时:
    → current_platform.check_and_update_config(vllm_config)
    → current_platform.get_attn_backend_cls() → 选择 attention 后端
```

## 官方文档参考

- `docs/design/plugin_system.md` — 插件类型、注册方式、兼容性保证的完整说明

## 详细笔记

> （实际阅读后填充）
