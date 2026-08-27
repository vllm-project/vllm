"""压力场景 benchmark：触发抢占 + 前缀共享，测 KV-aware 调度收益。

场景设计要点（对应设计文档的压力测试章节）：
1. 共享前缀池：8 个长 system prompt (~900 tokens)，请求随机选一个 → prefix cache 有命中空间
2. 混合输出长度 [512, 1024, 1536] → 长短请求竞争 KV cache，解码期持续膨胀
3. 服务端压小 KV cache 触发抢占（关键配置，客户端无法做到）：
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-7B-Instruct \
       --gpu-memory-utilization 0.75 \
       --num-gpu-blocks-override 1250 \
       --max-model-len 8192 --enable-prefix-caching
   ※ 坑：新版本 vLLM 默认 scheduler_reserve_full_isl=True，准入时检查完整输入序列，
     单纯高并发/KV 压到 45k 都触发不了抢占；必须 --num-gpu-blocks-override 压到
     20k tokens（1250 块）才稳定复现抢占
4. 从 /metrics 抓 vllm:num_preemptions_total 和 prefix cache 命中率（差值）

用法: python stress_bench.py [输出json路径]
"""

import asyncio
import json
import random
import sys
import time

import aiohttp

API = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ---- 场景参数 ----
NUM_PREFIX_POOLS = 8          # 共享前缀池数量
PREFIX_TOKENS = 900           # 每个 system prompt 约 900 token
SHARED_RATIO = 0.7            # 70% 请求使用共享前缀（30% 用独享前缀，制造对比）
CONCURRENCY = 48             # 并发数（压满 KV cache）
TOTAL_REQUESTS = 192          # 总请求数
OUTPUT_LENS = [512, 1024, 1536]  # 混合输出长度（偏长，制造解码期 KV 膨胀→抢占）
RNG_SEED = 42

random.seed(RNG_SEED)

# 预生成共享前缀（模拟多轮对话/同 system prompt 的业务负载）
SHARED_PREFIXES = []
for i in range(NUM_PREFIX_POOLS):
    # 重复文本凑 ~900 token，内容带编号确保 hash 不同
    base = (
        f"你是一个专业领域助手（角色 {i}）。你的任务是处理第 {i} 类业务问题，"
        "包括需求分析、方案设计、实施规划和结果评估等多个环节。"
    ) * 30
    SHARED_PREFIXES.append(base)

UNIQUE_PREFIX = "以下是一次性独立任务的上下文，不与其他请求共享："


def build_prompt(req_id: int) -> tuple[str, int]:
    """返回 (prompt, prefix_pool_id)；-1 表示独享前缀"""
    if random.random() < SHARED_RATIO:
        pid = random.randrange(NUM_PREFIX_POOLS)
        prefix = SHARED_PREFIXES[pid]
    else:
        pid = -1
        prefix = UNIQUE_PREFIX
    user_msg = f"[请求{req_id}] 请基于以上背景，详细阐述你的方案。"
    return f"{prefix}\n{user_msg}", pid


async def fetch_metrics(session: aiohttp.ClientSession) -> dict:
    """抓 vLLM /metrics 中与 KV/抢占相关的指标"""
    ret = {}
    try:
        async with session.get(f"{API}/metrics") as r:
            text = await r.text()
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in (
                "vllm:num_preemptions_total",
                "vllm:gpu_prefix_cache_hits_total",
                "vllm:gpu_prefix_cache_queries_total",
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
            ):
                if line.startswith(key + "{") or line.startswith(key + " "):
                    try:
                        ret[key] = float(line.rsplit(" ", 1)[1])
                    except (ValueError, IndexError):
                        pass
                    break
    except Exception as e:
        ret["error"] = str(e)
    return ret


async def one_request(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore, req_id: int, results: list
):
    prompt, pid = build_prompt(req_id)
    out_len = random.choice(OUTPUT_LENS)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_len,
        "temperature": 0.7,
        "stream": True,
    }
    rec = {
        "req_id": req_id,
        "prefix_pool": pid,
        "output_len": out_len,
        "ttft": None,
        "itl_mean": None,
        "itl_p99": None,
        "n_tokens": 0,
        "elapsed": None,
        "error": None,
    }
    t0 = time.perf_counter()
    first = True
    stamps = []
    try:
        async with sem:
            async with session.post(
                f"{API}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                    results.append(rec)
                    return
                async for line in resp.content:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        now = time.perf_counter()
                        if first:
                            rec["ttft"] = now - t0
                            first = False
                        else:
                            stamps.append(now)
                        rec["n_tokens"] += 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed"] = time.perf_counter() - t0
    # ITL 统计
    if stamps:
        prev = rec["ttft"] + t0  # 第一个 content 之后
        itls = []
        # stamps 是相对 time.perf_counter() 的绝对值，逐差分
        prev = None
        for s in stamps:
            if prev is not None:
                itls.append(s - prev)
            prev = s
        if itls:
            itls.sort()
            rec["itl_mean"] = sum(itls) / len(itls)
            rec["itl_p99"] = itls[min(len(itls) - 1, int(0.99 * len(itls)))]
    results.append(rec)


def pct(xs: list, p: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


async def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "stress_bench_baseline.json"
    print(f"[stress_bench] 并发={CONCURRENCY} 总请求={TOTAL_REQUESTS} "
          f"共享前缀池={NUM_PREFIX_POOLS}(~{PREFIX_TOKENS}tok) 共享比例={SHARED_RATIO}")
    print(f"[stress_bench] 输出长度混合: {OUTPUT_LENS}")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 8)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=1800)) as s:
        # 等服务就绪
        for _ in range(60):
            try:
                async with s.get(f"{API}/v1/models"):
                    break
            except Exception:
                await asyncio.sleep(2)
        else:
            print("服务不可达"); sys.exit(1)

        m_before = await fetch_metrics(s)
        print(f"[stress_bench] 指标(前): {m_before}")

        t_start = time.perf_counter()
        await asyncio.gather(*[
            one_request(s, sem, i, results) for i in range(TOTAL_REQUESTS)
        ])
        wall = time.perf_counter() - t_start

        m_after = await fetch_metrics(s)
        print(f"[stress_bench] 指标(后): {m_after}")

    ok = [r for r in results if r["error"] is None]
    err = [r for r in results if r["error"] is not None]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    itl_means = [r["itl_mean"] for r in ok if r["itl_mean"] is not None]
    itl_p99s = [r["itl_p99"] for r in ok if r["itl_p99"] is not None]
    total_tokens = sum(r["n_tokens"] for r in ok)

    def delta(key):
        return (m_after.get(key, 0) - m_before.get(key, 0)) if key in m_before and key in m_after else None

    summary = {
        "config": {
            "concurrency": CONCURRENCY, "total_requests": TOTAL_REQUESTS,
            "prefix_pools": NUM_PREFIX_POOLS, "shared_ratio": SHARED_RATIO,
            "output_lens": OUTPUT_LENS, "seed": RNG_SEED,
        },
        "wall_time_s": wall,
        "ok": len(ok), "errors": len(err),
        "ttft": {"mean": sum(ttfts) / len(ttfts) if ttfts else None,
                 "p50": pct(ttfts, 0.5), "p90": pct(ttfts, 0.9), "p99": pct(ttfts, 0.99)},
        "itl_mean": {"mean": sum(itl_means) / len(itl_means) if itl_means else None,
                     "p99": pct(itl_p99s, 0.99)},
        "output_throughput_tok_s": total_tokens / wall if wall else None,
        "request_throughput": len(ok) / wall if wall else None,
        "preemptions_delta": delta("vllm:num_preemptions_total"),
        "prefix_hits_delta": delta("vllm:gpu_prefix_cache_hits_total") or delta("vllm:prefix_cache_hits_total"),
        "prefix_queries_delta": delta("vllm:gpu_prefix_cache_queries_total") or delta("vllm:prefix_cache_queries_total"),
        "errors_detail": [r["error"] for r in err[:5]],
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "detail": results}, f, indent=2, ensure_ascii=False)

    print("\n===== 压力场景基线 =====")
    print(f"完成 {len(ok)}/{TOTAL_REQUESTS}  (失败 {len(err)})  总耗时 {wall:.1f}s")
    if ttfts:
        print(f"TTFT  mean={summary['ttft']['mean']:.2f}s  p50={summary['ttft']['p50']:.2f}s  "
              f"p90={summary['ttft']['p90']:.2f}s  p99={summary['ttft']['p99']:.2f}s")
    if itl_means:
        print(f"ITL   mean={summary['itl_mean']['mean']*1000:.1f}ms  p99={summary['itl_mean']['p99']*1000:.1f}ms")
    print(f"吞吐  output={summary['output_throughput_tok_s']:.1f} tok/s  request={summary['request_throughput']:.2f} req/s")
    print(f"抢占  preemptions_delta={summary['preemptions_delta']}")
    print(f"前缀  hits={summary['prefix_hits_delta']}  queries={summary['prefix_queries_delta']}")
    if err:
        print(f"错误示例: {summary['errors_detail'][:2]}")
    print(f"\n结果已存 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
