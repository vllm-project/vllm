# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(f"vllm.{__name__}")


class MonokernelCheckExtension:
    """Proves an AMD monokernel served the eval: ``start_monokernel_count`` (a
    startup check) takes its device launch counters, ``check_monokernel_ran``
    (a final check) asserts that every one of them moved since."""

    model_runner: "GPUModelRunner"

    def _modules(self) -> list:
        models = [self.model_runner.model, self.model_runner.get_draft_model()]
        return [m for model in models if model is not None for m in model.modules()]

    def _specialized_decode(self):
        """MiniMax-M3's or Kimi-K3's specialized decode, None for other models."""
        return next(
            (
                decode
                for m in self._modules()
                if (decode := getattr(m, "specialized_decode", None)) is not None
            ),
            None,
        )

    def _launch_counters(self) -> dict[str, int] | str:
        """Each monokernel runner's device launches so far, or why this rank
        has none."""
        decode = self._specialized_decode()
        if decode is not None:
            executor = getattr(decode, "executor", None)
            if executor is None:  # Kimi-K3: one epoch step a kernel forward
                return {"kimi_k3": int(decode.runner.epoch[0])}
            # MiniMax-M3: one epoch step a launch, cleared every ~2000 steps
            epoch = int(executor.runner.epoch[0])
            return {"minimax_m3": getattr(executor, "cleared", 0) + epoch}
        layers = [m.mono for m in self._modules() if getattr(m, "mono", None)]
        if not layers:
            return "no monokernel is enabled on this rank"
        from vllm.models.deepseek_v4.amd import mono_decode as dsv4

        if isinstance(layers[0], dsv4.MonoDecodeLayer):
            return {"dsv4": 0 if dsv4._runner is None else int(dsv4._runner.epoch[0])}
        from vllm.models.deepseek_v41.amd import mono_decode

        runners = {"dsv41": mono_decode._runner}
        if any(layer.draft for layer in layers):
            runners["dsv41_draft"] = mono_decode._draft_runner
        return {
            name: 0 if runner is None else int(runner.epoch[0])
            for name, runner in runners.items()
        }

    def start_monokernel_count(self) -> str | None:
        counters = self._launch_counters()
        if isinstance(counters, str):
            return counters
        executor = getattr(self._specialized_decode(), "executor", None)
        if executor is not None:
            # count the launches each MiniMax-M3 epoch reset clears
            clear, executor.cleared = executor._clear, 0

            def count_and_clear():
                executor.cleared += int(executor.runner.epoch[0])
                clear()

            executor._clear = count_and_clear
        self._monokernel_start = counters
        return None

    def check_monokernel_ran(self) -> str | None:
        counters = self._launch_counters()
        if isinstance(counters, str):
            return counters
        moved = {
            name: count - self._monokernel_start[name]
            for name, count in counters.items()
        }
        logger.info("Monokernel launches during the eval: %s", moved)
        idle = [name for name, launches in moved.items() if launches <= 0]
        return f"no monokernel launches for {idle}: {moved}" if idle else None
