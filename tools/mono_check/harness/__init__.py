# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug / bring-up harnesses of the GLM-5.2 MonoKernel (not shipped in the vllm
package).

Worker extensions (``worker_extension_cls="tools.mono_check.harness.<module>.<Class>"``;
the repo root must be importable in the workers, as with the dev overlay):
``live_worker_ext`` (install + control RPCs), ``serve_ext`` (``vllm serve`` health RPC +
fail-stop watch), ``perf_ext`` (per-step GPU timestamps), ``worker_ext`` (check mode,
``check.py`` + its FP32 golden ``reference.py``), ``idx_shadow`` / ``idx_equiv``
(indexer-mode shadows), ``live_selftest`` (eager self-tests attached from
``LiveConfig.extra``)."""
