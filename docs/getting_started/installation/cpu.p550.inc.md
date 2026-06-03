<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM can be brought up on the SiFive HiFive Premier P550 board through the CPU backend. Treat this as an experimental RISC-V source build path. The first milestone is a native CPU build and tiny-model offline inference; API serving and performance tuning are separate follow-up steps.

The P550 baseline is the RISC-V scalar CPU path. If the board reports RVV support with `zvl128b` or `zvl256b`, the existing RISC-V RVV attention path can be used. If those ISA strings are absent, force scalar with `VLLM_RVV_VLEN=0`.

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: Ubuntu 24.04 on the P550 board
- Compiler: `gcc/g++ >= 12.3.0`
- Build tools: `cmake`, `ninja-build`, `python3-dev`, `python3-venv`, `git`, `curl`
- Runtime libraries: `libnuma-dev`, `libtcmalloc-minimal4`
- PyTorch: a working `riscv64` PyTorch install is required before building vLLM

Run the probe script first:

```bash
bash tools/p550_probe.sh | tee p550_probe.log
```

The script prints the detected ISA string, a recommended `VLLM_RVV_VLEN` value, toolchain versions, and whether `torch` can be imported.

If the standard dependency path is not available on the board, follow the recorded component plan in [SiFive P550 Component Bring-up Plan](p550_component_plan.md) before adding compatibility shims or building missing packages from source.

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

There are no pre-built vLLM RISC-V CPU wheels for the P550 target. Build vLLM from source on the board.

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

Install system packages on the board:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential gcc g++ \
    cmake ninja-build python3-dev python3-venv \
    libnuma-dev libtcmalloc-minimal4
```

Create a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Install build dependencies. Keep `VLLM_TARGET_DEVICE=cpu` set so setup does not attempt any accelerator backend:

```bash
export VLLM_TARGET_DEVICE=cpu
python -m pip install -r requirements/build/cpu.txt
python -m pip install -r requirements/cpu.txt
```

If `torch==2.11.0` is not available for `riscv64`, install a board-compatible PyTorch build first, then rerun the requirements command with the already-installed PyTorch preserved.

The P550 bring-up used `torch==2.4.1` from `https://ext.kmtea.eu/simple` and a
venv-local `pyzmq>=26`. Ubuntu's system `pyzmq 24.0.1` has an older
`Socket.shadow(int)` API and is not sufficient for the vLLM V1 engine process
client.

Select the RISC-V CPU path:

```bash
# Use scalar when /proc/cpuinfo does not report zvl128b or zvl256b.
export VLLM_RVV_VLEN=0

# If the probe reports zvl128b or zvl256b, leave VLLM_RVV_VLEN unset and let
# CMake auto-detect, or set it explicitly to 128 or 256.
```

Build and install vLLM:

```bash
VLLM_TARGET_DEVICE=cpu python -m pip install -e . --no-build-isolation
```

Run the minimal P550 smoke test:

```bash
bash tools/p550_smoke_test.sh
```

The smoke test uses `hmellor/tiny-random-LlamaForCausalLM` by default. Override it with `VLLM_P550_MODEL=<model-id-or-path>` if the default model is not available.

If the board cannot download from Hugging Face, create the local tiny model
used during bring-up:

```bash
python tools/p550_make_tiny_llama.py
bash tools/p550_smoke_test.sh
```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

There are no pre-built vLLM RISC-V CPU images for the P550 target.

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

Docker bring-up is not part of the P550 minimum target. Use the native source build path above.

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]

For the first P550 milestone, validate only native build, `import vllm`, CPU attention tests that fit on the board, and tiny-model offline inference. Tune RVV and larger models after the scalar path is reliable.

--8<-- [end:extra-information]
