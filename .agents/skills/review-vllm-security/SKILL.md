---
name: review-vllm-security
description: Prevent recurring vLLM security defects when designing, implementing, or reviewing models, processors, media loaders, endpoints and request fields, tool parsers, deployment settings, and changes to shared engine state, caches, or worker transports. Use when input changes how vLLM spends resources, crosses a trust boundary, or affects other requests.
---

# Review vLLM security

Use this before choosing an implementation and again when reviewing the diff.
Read the project's [threat model](../../../docs/usage/security.md) and
[security policy](../../../SECURITY.md). Follow the
[vulnerability management process](../../../docs/contributing/vulnerability_management.md)
for a reported vulnerability.

## Review workflow

1. Trace every affected input from its source to the first operation that trusts
   it or spends resources on it. Distinguish inference callers, model output,
   artifact authors, operators, and trusted peers. Include conversions,
   defaults, expansion, and the error path.
2. Scan the quick screen below. Use the change-type table to open only the
   relevant sections of the detailed checklist. Include any additional family
   flagged by the quick screen for this change.
3. Search for sibling consumers by field or operation. When applicable, compare
   Python and Rust, HTTP and gRPC, streaming and buffered flows, model or backend
   overrides, and opt-in and fallback paths. Identify where each path enforces
   the control and which paths share it.
4. Confirm where rejected work, cancellation, or a timeout leaves resources and
   shared state. For each applicable risk, record the enforcing boundary, the
   sibling paths checked, and the observed behavior or remaining verification.

## Quick screen

| Review area | Question to answer |
| --- | --- |
| [Resource budgets][budgets] | What bounds the bytes, counts, work, and retained state before each stage incurs the cost? |
| [Input invariants and fault containment][input] | Which checks prevent invalid inputs from reaching the engine, scheduler, or native consumer? |
| [Cache, request, and peer ownership][ownership] | What preserves ownership when requests, buffers, cache keys, and transfer IDs are reused? |
| [Parser and rendering complexity][parsing] | What bounds parsing, compilation, and rendering under malformed or fragmented input and concurrent work? |
| [Artifact trust and unsafe interpretation][artifacts] | What permits executable loading, and are trust choices and revision pins enforced at every actual load? |
| [Fetch, filesystem, and decoder policy][fetch] | What confines every network destination, file access, and decoder invocation? |
| [Authorization and capability gates][access] | Who can reach the action, set privileged state, or select and invoke a tool? |
| [Sensitive output and error responses][output] | What can enter responses, logs, metrics, or model context, and how large can it become? |

## Detailed checks by change

Start with the families for each affected change type.

| Change type | Checklist sections |
| --- | --- |
| Endpoint, request field, or frontend | [Authorization][access], [resource budgets][budgets], [input invariants][input], [sensitive output][output] |
| Multimodal model, processor, or media backend | [Fetch and decoder policy][fetch], [resource budgets][budgets], [input invariants][input], [artifact trust][artifacts] |
| Model, tokenizer, quantization, or artifact loading | [Artifact trust][artifacts], [path confinement][fetch], [native input invariants][input] |
| Structured output, template, reasoning, or tool calls | [Parser complexity][parsing], [authorization][access], [resource budgets][budgets], [sensitive output][output] |
| Engine, scheduler, sampler, or native kernel | [Input invariants][input], [resource budgets][budgets], [shared-state ownership][ownership] |
| Cache, batching, IPC, or disaggregated serving | [Ownership][ownership], [authorization][access], [resource budgets][budgets], [safe IPC decoding][artifacts] |
| Deployment, configuration, or plugin | [Authorization][access], [artifact trust][artifacts] |
| Errors, logging, metrics, or usage reporting | [Sensitive output][output], [resource budgets][budgets] |

Use the relevant section of [Checks by change](references/checks-by-change.md)
when you need vLLM entry points or examples of what to verify.

For a limit regression, use a small configured cap or a stubbed consumer.
Check valid boundary behavior and rejection before expensive work. For
shared-state changes, follow a failed or cancelled request with a valid one,
or interleave two requests and verify their results and cleanup independently.
See [Published advisory examples](references/advisory-evidence.md) for
cases behind these checks.

[budgets]: references/failure-modes.md#resource-budgets
[input]: references/failure-modes.md#input-invariants-and-fault-containment
[ownership]: references/failure-modes.md#cache-request-and-peer-ownership
[parsing]: references/failure-modes.md#parser-and-rendering-complexity
[artifacts]: references/failure-modes.md#artifact-trust-and-unsafe-interpretation
[fetch]: references/failure-modes.md#fetch-filesystem-and-decoder-policy
[access]: references/failure-modes.md#authorization-and-capability-gates
[output]: references/failure-modes.md#sensitive-output-and-error-responses
