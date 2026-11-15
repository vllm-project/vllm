# Validation on November 15, 2026

Base source: vLLM `919d45f`. This is an experimental local patch, not an accepted
upstream fix for #62574.

## Hardware and blocked work

The Windows host reports AMD Radeon integrated graphics. WSL PyTorch
`2.13.0+cu130` reports `torch.cuda.is_available() == False` and zero CUDA devices.
No live vLLM GPU server or GB10/DeepSeek reproduction was run. Real GPU decode
p99, TTFT and throughput are **not available**.

Full engine scheduler pytest cases were added for sync/async and enabled/disabled
operation, but could not execute: this checkout lacks pytest and the complete
vLLM test environment (configuration import also requires missing pydantic).
No GPU result may be inferred from syntax validation or the tests below.

## Executed checks

- 16 standard-library unittest cases pass: controller admission/credits,
  context cost, pending reservations, rollback refunds, invalid feedback,
  bounded state, aged minimum progress, fair shares, and local HTTP protocol
  handling (including truncated and multi-token streams).
- Modified Python files compile; `git diff --check` passes.
- The streaming fixture is only a protocol test, not inference evidence.

## Synthetic policy experiment only

`PYTHONPATH=. python benchmarks/adaptive_prefill/simulate.py` compares both
policies under the explicitly declared artificial device-cost formula:

`0.5 + 0.18 * decode_requests + 0.006 * context_weighted_prefill_tokens` ms.

Mixed scenario: four 16,384-token prompts and 32 short prompts; prompt chunks
capped at 1,024 tokens, aggregate token budget 8,192, identical arrival trace.
Percentiles use nearest rank.

| Synthetic metric | Fixed-budget baseline | Adaptive |
| --- | ---: | ---: |
| Short decode gap p99, ms | 52.040 | 25.045 |
| Short TTFT p95, ms | 90.484 | 35.985 |
| Long TTFT maximum, ms | 685.528 | 851.958 |
| Output tokens/s | 1767.433 | 1732.728 |
| All long prefills completed | yes | yes |

Short-only controls are identical. Long-only output throughput decreases from
40.822 to 39.997 synthetic tokens/s. Mixed-load throughput decreases about 2.0%;
long-request TTFT increases about 24.3%. These tradeoffs must remain visible.

The experiment supports testing the control logic, not a claim about vLLM GPU
performance, real p99 protection or starvation freedom under arbitrary overload.
The original issue's speculative hybrid-model configuration is intentionally
unsupported by this first implementation.

## Next required gate

Install this commit in a CUDA-capable vLLM environment. Run the added real
streaming benchmark against baseline and opt-in servers on identical hardware,
model, token budgets and arrival traces; repeat and sweep targets, then execute
the original reported configuration only after adding its missing support.
