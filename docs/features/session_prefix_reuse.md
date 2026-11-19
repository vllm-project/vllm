# Session prefix reuse

Session prefix reuse uses the existing session ID to select a recent candidate
for exact input tokenization reuse.
Enable it explicitly; the default budget is zero (disabled):

```console
vllm serve Qwen/Qwen3-0.6B \
    --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
    --session-token-cache-mb 32
```

Budgets are MiB per renderer. Histories
occupy CPU memory; they do not retain physical KV blocks or replace APC.
Accounting includes reservations and in-flight readers, and conservatively
charges each token occurrence. The budget is not a process RSS limit.

Send the existing `session_id` request field or `X-Session-ID` header.
Precedence is request field, header, then `vllm_xargs.session_id`.
Session identity selects a candidate; exact content validation remains required.
Each process keeps one successfully published candidate per session and salt.
Concurrent completion order selects the retained candidate. Routing and forks
can reduce reuse. Cancellation, errors, reset and shutdown discard pending
publication as appropriate. Snapshots retain text and IDs, not media payloads.

## Tokenizer admission

Token reuse supports text Chat Completions with the following exact library
versions and audited tokenizer configurations. Each Transformers version in
the table can be paired with either listed tokenizers version:

| Transformers | tokenizers |
| --- | --- |
| 5.16.1 | 0.23.2, 0.23.3 |
| 5.17.0 | 0.23.2, 0.23.3 |
| 5.18.0 | 0.23.2, 0.23.3 |
| 5.19.0 | 0.23.2, 0.23.3 |
| 5.20.0 | 0.23.2, 0.23.3 |
| 5.20.1 | 0.23.2, 0.23.3 |
| 5.20.2 | 0.23.2, 0.23.3 |

When auditing new library pairs, update the production admission list and the
independent expected-version list in the renderer tests together. Run both
positive reuse and unsupported-version fallback checks.

These are explicit supported pairs, not a continuous version range. Install
mutually compatible dependencies: tokenizers 0.23.2 requires
`huggingface-hub<2.0`; compatibility validation used Hub 1.31.0.

| Tokenizer asset repository | Audited revision |
| --- | --- |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| unsloth/Llama-3.2-1B-Instruct | `5a8abab4a5d6f164389b1079fb721cfab8d7126c` |
| zai-org/GLM-4.5 | `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` |
| deepseek-ai/DeepSeek-V3.1 | `c0781d039fb7a1ba2abc4add0bdc293e92d2b8db` |

These repositories identify tokenizer assets, not a guarantee for every model
in a family. Qwen3 and Qwen3-Coder load the same audited backend. Compatibility
validation checks complete input token IDs, real chat templates, history edits,
length errors and fallback behavior; it does not certify GPU model execution
or performance for every listed model.

For DeepSeek-V3.1 tool calling, use vLLM's documented configuration:
`--tool-call-parser deepseek_v31` together with
`--chat-template examples/tool_chat_template_deepseekv31.jinja`.
See [DeepSeek-V3.1 tool calling](tool_calling.md#deepseek-v31-models-deepseek_v31).
The repository template serializes dictionary arguments and includes tool
definitions. The model-bundled template is not equivalent: it expects string
arguments and can reject tool-call histories after vLLM parses the arguments.
Token reuse neither substitutes templates automatically nor repairs such errors.
Mistral-7B-Instruct-v0.3's Metaspace backend is not admitted: encoding a suffix
independently can change its IDs. Other tokenizer configurations, including
ones that passed a preliminary raw-JSON survey, are not automatically admitted.

Other model revisions using an identical audited configuration can pass the same
checks. Activation checks the complete backend configuration
and public encoding implementation; a model name alone is insufficient.
The tokenizer must remain immutable. Unsupported configurations or library
versions disable token reuse with a warning and use native tokenization, even
when those library versions satisfy vLLM's installation requirements.

The implementation requires GIL-enabled CPython. Token accounting uses Python
and does not depend on a native helper.

## Exactness and scope

Token reuse accepts only unchanged prefixes ending at audited complete added
token boundaries. Unsafe overlap, whitespace stripping, unsupported options,
encoding overrides or content mismatches take the native path. Full input IDs
remain the correctness reference.

Requests containing images or other media use native tokenization.
They do not publish media reuse histories. This stage optimizes text chat only.

No performance improvement is guaranteed. Misses, short inputs, frequent edits,
concurrency and routing can make admission, validation and publication cost
more than recomputation. Measure the enabled configuration on the target load.
