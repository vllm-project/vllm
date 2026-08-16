# 第4章：入口与 API 层

> 一句话：vLLM 提供三大入口（LLM 离线、OpenAI API Server、CLI），它们最终都收敛到 Engine Core，本章追踪请求从用户到引擎的完整路径。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/entrypoints/llm.py` | ~916 | LLM 类：离线推理的主入口，generate() / encode() / chat() |
| `vllm/entrypoints/openai/api_server.py` | ~804 | OpenAI 兼容 API 服务器：FastAPI 路由、请求转发 |
| `vllm/entrypoints/cli/serve.py` | ~408 | CLI `vllm serve` 命令：解析参数、启动 API Server |
| `vllm/entrypoints/chat_utils.py` | ~2057 | Chat 消息格式转换：OpenAI messages → prompt tokens |
| `vllm/entrypoints/openai/dp_supervisor.py` | ~557 | DP Supervisor：数据并行模式下的多 API Server 管理 |

## 关键问题（带着这些问题读）

1. `LLM.generate()` 的同步接口内部是如何驱动异步引擎的？
2. OpenAI API Server 如何处理 streaming（SSE）响应？数据从 Engine 到 HTTP chunk 的路径？
3. `vllm serve` 启动时，API Server 进程与 Engine Core 进程是如何建立 ZMQ 通信的？
4. 数据并行（DP）模式下，多个 API Server 如何做负载均衡？请求如何路由到不同的 Engine Core？

## 调用链概览

```
用户侧:
  LLM(model="...").generate(prompts)     # 离线
  curl http://localhost:8000/v1/chat/...  # 在线

离线路径:
  LLM.generate() → LLMEngine → EngineCoreClient (同进程)
    → EngineCore.step()

在线路径:
  vllm serve → api_server.py: create_app()
    → FastAPI routes → AsyncLLM
      → EngineCoreClient (ZMQ) → EngineCore (独立进程)
```

## 官方文档参考

- `docs/design/arch_overview.md` — Entrypoints 一节
- `docs/serving/online_serving/` — 在线服务配置

## 详细笔记

> （实际阅读后填充）
