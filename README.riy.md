```
                    ██████╗ ██╗██╗   ██╗
                    ██╔══██╗██║╚██╗ ██╔╝
                    ██████╔╝██║ ╚████╔╝
                    ██╔══██╗██║  ╚██╔╝
                    ██║  ██║██║   ██║
                    ╚═╝  ╚═╝╚═╝   ╚═╝
          Runtime Expert Masking for MoE Models
                   ── Pruning on Air ──
```

# vllm-riy

**See which experts your model actually uses. Mask the rest. No restart needed.**

RIY hooks into vLLM's MoE routing layer and gives you two things:

1. **Live statistics** — activation frequency and routing weight magnitude per `(layer, expert)`, collected on your actual workload
2. **Runtime masking** — deactivate experts via HTTP API, instantly reversible, no checkpoint modification

---

## Quick Start

```bash
# Start vLLM with any MoE model (the riy branch adds the hook automatically)
vllm serve Qwen/Qwen3.5-122B-A10B --trust-remote-code

# Start collecting stats
curl -X POST http://localhost:8019/riy/stats/start

# Run your workload, then check
curl http://localhost:8019/riy/stats | python3 -m json.tool

# Or use the live TUI
python3 tools/riy_live.py
```

```
  riy live | qwen3.5-122b | localhost:8019 | layers:48 exp:256 | 2.0s | ?=help
          |0    0    0    0    0    0    0    0    1    1    1    1    1
          |0    1    2    3    4    5    6    7    8    9    0    1    2
  L0001   |▓░▓█░░▒░░·░░▒░·░░░░░▒·░░░░·░░·░░░░▒░░··░░░░··░░░░░▓░··░░
  L0002   |█▒▓▓░░▒░░·░░▒░░░░░░░▒·░░░░·░░·░░░░▒░░··░░░░··░░░░░▓░··░░
  L0003   |▓░▓█░░▒░░·░░░░·░░░░░░·░░░░·░░·░░░░▒░░··░░░░··░░░░░▒░··░░
  L0004   |▓▒▒▓░░▒░░·░░▒░·░░░░░▒·░░░░·░░·░░░░░░░··░░░░··░░░░░▓░··░░
  ...
  dead:1840(15%)  rare:4211  low:3892  active:2345  |  prunable:6051(49%)  specialists:23
  freq: · dead  ░ rare  ▒ low  ▓ high  █ dominant  |  color=contribution: ■■■■■
  prune potential: [████████████████████████░░░░░░░░░░░░░░░░░░░░░░░] 49%
  q=quit  r=reset  e=enable  d=disable  m=mask  s=save  ?=help
```

---

## Why

MoE models activate only a fraction of their experts per token. Many experts
are rarely or never called for a given workload — but they still consume VRAM.

Existing pruning tools (like Cerebras REAP) use generic benchmarks to decide
which experts to cut. **RIY lets you measure on your own workload and decide
yourself.**

| | Cerebras REAP | vllm-riy |
|--|--------------|---------|
| Calibration data | Generic benchmarks | Your workload |
| Output | Static pruned model | Profile JSON, model unchanged |
| Reversibility | No | Yes, any time |
| Quantization-dependent | Yes | No — same profile, any quant |
| Automatic decisions | Yes | No — operator decides |

---

## API

RIY runs a standalone HTTP server on port **8019** inside the vLLM engine process.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/riy/health` | Status: enabled, collecting, layers, experts |
| `GET` | `/riy/stats` | Raw per-(layer, expert) frequency + weight sum |
| `POST` | `/riy/stats/start` | Start collecting |
| `POST` | `/riy/stats/stop` | Stop collecting |
| `POST` | `/riy/stats/reset` | Zero all counters |
| `GET` | `/riy/mask` | Current mask |
| `POST` | `/riy/mask` | Set mask: `{"pruned_experts": [[0,3],[4,7]]}` |
| `DELETE` | `/riy/mask` | Clear mask |
| `POST` | `/riy/profile/load` | Load from file: `{"path": "/data/profile.json"}` |

---

## Profile Format

```json
{
  "version": 1,
  "model": "Qwen3.5-397B-A17B",
  "workload": "municipal German administrative",
  "pruned_experts": [[0, 3], [0, 11], [4, 7], [12, 2]]
}
```

Profiles are quantization-agnostic. Same profile works on BF16, FP8, INT4.
Share them, version them, publish them on HuggingFace.

---

## Workflow

```
1.  Start model — fully loaded, no mask
2.  curl -X POST :8019/riy/stats/start
3.  curl -X POST :8019/riy/stats/reset    ← clean slate
4.  Run your actual workload
5.  curl :8019/riy/stats > stats.json      ← export raw data
6.  Analyze offline, build profile
7.  curl -X POST :8019/riy/mask -d @profile.json  ← apply live
8.  Observe quality — clear mask if degraded
9.  Satisfied → save profile, use --riy-expert-profile next start
```

---

## Expert Categories

| Frequency | Contribution | Assessment |
|-----------|-------------|------------|
| Never | — | Dead — safe to prune |
| Rare | Low | Candidate — prune |
| Rare | High | Specialist — workload dependent, caution |
| Frequent | Low | Redundant — candidate |
| Frequent | High | Essential — keep |

The TUI shows these as:
- **Fill** `·░▒▓█` = activation frequency (log scale)
- **Color** dark → cyan → green → orange → red = routing weight magnitude (log scale)

---

## Installation

This is a branch on a vLLM fork. Apply the patch to any vLLM image:

```bash
# Generate patch
cd vllm-riy && git diff main -- vllm/ > riy.patch

# Apply to running container
podman run --name patch -v riy.patch:/tmp/riy.patch:ro your-vllm-image \
  bash -c "cd /usr/local/lib/python3.12/dist-packages && patch -p1 < /tmp/riy.patch"
podman commit patch your-vllm-image-riy
```

Or use `--riy-expert-profile` CLI flag for load-time masking.

---

## Limitations

- **No VRAM savings at runtime.** Masked experts still occupy memory.
  Load-time masking zeros weights but doesn't skip allocation (follow-up).
- **`--enforce-eager` required for stats.** CUDA Graphs replay the captured
  graph without executing Python — the stats hook doesn't fire during replay.
  Masking works with CUDA Graphs; stats collection does not.
- **Single-process stats.** Stats are collected in the EngineCore worker process
  and served via a separate HTTP server on port 8019.

---

## License

Same as vLLM — Apache 2.0.

Part of the [flash7777/vllm](https://github.com/flash7777/vllm) fork, branch `riy`.
