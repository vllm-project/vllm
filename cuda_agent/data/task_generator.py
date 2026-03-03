"""
data/task_generator.py

Generates agent_workdir-style task directories from CUDA-Agent-Ops-6K samples.

Each task directory mirrors the agent_workdir layout:
  <task_dir>/
    model.py          — reference PyTorch model (the sample's code)
    model_new.py      — stub for the agent to fill in
    SKILL.md          — symlink / copy of the canonical SKILL.md
    binding.cpp       — (copied from template)
    binding_registry.h
    kernels/          — empty; agent writes here
    utils/            — copied from template

Usage:
    from cuda_agent.data.task_generator import TaskGenerator
    gen = TaskGenerator(template_dir="cuda_agent/agent_workdir",
                        output_dir="/tmp/tasks")
    gen.generate_task(sample, task_id="task_0001")
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

from cuda_agent.data.dataset_loader import CUDAAgentSample


MODEL_NEW_STUB = textwrap.dedent("""\
    \"\"\"
    Optimized model — implement custom CUDA kernels here.

    Rules:
      - Must expose class ModelNew with identical __init__ / forward signatures
        as Model in model.py.
      - May NOT use torch.nn.functional operations.
      - Must call cuda_extension.* for computation.

    After implementation:
      1. Add kernel files to kernels/
      2. Run: TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
      3. Verify: python3 -m utils.verification
      4. Profile: python3 -m utils.profiling
    \"\"\"

    import torch
    import torch.nn as nn

    # TODO: import cuda_extension after compilation
    # import cuda_extension


    class ModelNew(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # TODO: store parameters
            raise NotImplementedError("Implement ModelNew.__init__")

        def forward(self, *args, **kwargs):
            # TODO: call cuda_extension.*_forward(...)
            raise NotImplementedError("Implement ModelNew.forward")
""")


class TaskGenerator:
    """Creates self-contained task directories from dataset samples."""

    def __init__(self, template_dir: str | Path, output_dir: str | Path):
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_task(self, sample: CUDAAgentSample, task_id: str) -> Path:
        """
        Create a task directory for a single sample.

        Returns the path to the generated directory.
        """
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        self._copy_template_files(task_dir)
        self._write_model(task_dir, sample)
        self._write_model_new_stub(task_dir)
        self._write_task_metadata(task_dir, sample, task_id)

        return task_dir

    def generate_all_tasks(
        self,
        samples: list[CUDAAgentSample],
        prefix: str = "task",
    ) -> list[Path]:
        """Generate task directories for a list of samples."""
        paths: list[Path] = []
        for i, sample in enumerate(samples):
            task_id = f"{prefix}_{i:05d}"
            paths.append(self.generate_task(sample, task_id))
        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _copy_template_files(self, task_dir: Path) -> None:
        """Copy immutable infrastructure files from the template workdir."""
        immutable_files = [
            "binding.cpp",
            "binding_registry.h",
        ]
        for fname in immutable_files:
            src = self.template_dir / fname
            if src.exists():
                shutil.copy(src, task_dir / fname)

        # Copy utils/ and SKILL.md
        for item in ("utils", "SKILL.md"):
            src = self.template_dir / item
            dst = task_dir / item
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy(src, dst)

        # Create empty kernels/ directory
        (task_dir / "kernels").mkdir(exist_ok=True)

    def _write_model(self, task_dir: Path, sample: CUDAAgentSample) -> None:
        """Write the reference model.py from the dataset sample."""
        model_path = task_dir / "model.py"
        header = (
            '"""\n'
            'Reference PyTorch baseline model.\n'
            'Do NOT modify this file.\n'
            '"""\n\n'
        )
        # Prepend the header only if the sample code doesn't already have one
        code = sample.code
        if '"""' not in code[:50] and "'''" not in code[:50]:
            code = header + code
        model_path.write_text(code, encoding="utf-8")

    def _write_model_new_stub(self, task_dir: Path) -> None:
        """Write the empty model_new.py stub for the agent to fill in."""
        (task_dir / "model_new.py").write_text(MODEL_NEW_STUB, encoding="utf-8")

    def _write_task_metadata(
        self,
        task_dir: Path,
        sample: CUDAAgentSample,
        task_id: str,
    ) -> None:
        """Write a metadata JSON file describing the task."""
        import json
        meta = {
            "task_id": task_id,
            "sample_id": sample.sample_id,
            "ops": sample.ops,
            "data_source": sample.data_source,
            "code_length": len(sample.code),
            "num_ops": len(sample.ops),
        }
        (task_dir / "task_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import logging
    from cuda_agent.data.dataset_loader import load_cuda_agent_dataset

    logging.basicConfig(level=logging.INFO)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cuda_agent_tasks"

    samples = load_cuda_agent_dataset(max_samples=n)
    gen = TaskGenerator(
        template_dir=Path(__file__).resolve().parent.parent / "agent_workdir",
        output_dir=out,
    )
    paths = gen.generate_all_tasks(samples)
    print(f"Generated {len(paths)} task(s) in {out}")
    for p in paths:
        print(f"  {p}")
