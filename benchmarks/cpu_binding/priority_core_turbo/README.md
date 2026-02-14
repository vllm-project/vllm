# Enabling Priority Core Turbo (PCT) for vLLM GPU Performance

## Overview

**Intel® Priority Core Turbo (PCT)** is part of **Intel® Speed Select Technology – Turbo Frequency (SST-TF)**.
It allows a subset of CPU cores to operate at **higher turbo frequencies**, while remaining cores run closer to base frequency.

This is particularly effective for **GPU-accelerated vLLM inference**, where a small number of CPU threads handle
**latency-critical, mostly serial tasks** such as tokenization, scheduling, and feeding GPUs.
Running these threads on **High-Priority (HP) cores** improves GPU utilization, TTFT, and tail latency.

Validated platforms:

- **Intel® Xeon® 6776P**

## How PCT Works

PCT relies on **two Intel Speed Select features**:

- **SST-TF (Turbo Frequency)**
  Defines how many cores are allowed to run at higher turbo frequencies (HP cores).

- **SST-CP (Core Power / CLOS)**
  Assigns CPUs to **Classes of Service (CLOS)**.
  Only CPUs assigned to **CLOS0** are treated as **High-Priority** by PCT.

> **Important:** PCT is only effective when CPUs are explicitly assigned to **CLOS0**.

## 1. Build the Environment

Build the Docker image with required tools:

```bash
docker compose build --no-cache
```

## 2. Check PCT Status (Read-Only)

This step verifies:

- Hardware support for Intel® Speed Select features
- BIOS enablement of PCT (SST-TF) and Core Power (SST-CP)
- Whether CPUs are already assigned to **CLOS0** (required for PCT)

Run:

```bash
docker compose --profile check up --abort-on-container-exit
```

Example results when PCT is enabled successfully.

```bash
------------------------------------------------------------
CPU and Intel Speed Select Capability
------------------------------------------------------------
Intel(R) SST-PP (feature perf-profile) is supported
Intel(R) SST-TF (feature turbo-freq) is supported
Intel(R) SST-BF (feature base-freq) is not supported
Intel(R) SST-CP (feature core-power) is supported
Intel(R) Speed Select Technology
Executing on CPU model:173[0xad]

------------------------------------------------------------
PCT (Turbo-Frequency) Feature Status
------------------------------------------------------------
✅ PCT (Turbo-Frequency) data present.

------------------------------------------------------------
Core Power (CLOS) Feature Status
------------------------------------------------------------
✅ Core Power feature ENABLED
✅ CLOS ENABLED

------------------------------------------------------------
CPU list for TARGET_CLOS=0
------------------------------------------------------------
clos:0 CPU list: 0-7,32-39,64-71,96-103,128-135,160-167,192-199,224-231

------------------------------------------------------------
Summary
------------------------------------------------------------
✅ PCT turbo tables detected (turbo-freq reports high-priority data)
✅ Core Power enabled
✅ CLOS enabled
```

## 3. Set PCT (Assign CPUs to CLOS0)

This step **activates PCT in practice** by assigning CPUs to the correct
**Class of Service (CLOS)**.

The setup script automatically performs the following actions:

- Detects how many **High-Priority (HP) cores** are supported by the platform
  (from `intel-speed-select perf-profile info`)
- Selects HP cores **per NUMA node** to maintain locality
- Expands the HP set to include **Hyper-Threading siblings** when required
- Assigns:
    - **HP CPUs → CLOS0** (eligible for Priority Core Turbo)
    - **All remaining CPUs → CLOS2** (non-HP cores)

Run the setup:

```bash
docker compose --profile set up --abort-on-container-exit
```

Example results when PCT is set successfully based on power-domains.

```bash
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | Config
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | HP_PER_DOMAIN=8 (HP_BUCKET=0)
intel-speed-select-set-1  | INCLUDE_HT=0
intel-speed-select-set-1  | HP_CLOS=0  OTHER_CLOS=2
intel-speed-select-set-1  | DEBUG_MODE=0  DRY_RUN=0  DEBUG_VERBOSE=0  DEBUG_MAP=0
intel-speed-select-set-1  |
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | HP selection per NUMA node (initial pick)
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | node 0 -> 0 1 2 3 4 5 6 7
intel-speed-select-set-1  | node 1 -> 32 33 34 35 36 37 38 39
intel-speed-select-set-1  | node 2 -> 64 65 66 67 68 69 70 71
intel-speed-select-set-1  | node 3 -> 96 97 98 99 100 101 102 103
intel-speed-select-set-1  |
intel-speed-select-set-1  | HP initial ranges      : 0-7,32-39,64-71,96-103
intel-speed-select-set-1  | HP effective (with HT) : 0-7,32-39,64-71,96-103,128-135,160-167,192-199,224-231
intel-speed-select-set-1  |
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | Computed CPU lists
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | HP (effective) : 0-7,32-39,64-71,96-103,128-135,160-167,192-199,224-231
intel-speed-select-set-1  | Non-HP         : 8-31,40-63,72-95,104-127,136-159,168-191,200-223,232-255
intel-speed-select-set-1  |
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | Apply CLOS assignments (quiet)
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | Setting HP -> CLOS0, Non-HP -> CLOS2
intel-speed-select-set-1  | Applied.
intel-speed-select-set-1  |
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | Verification (concise CPU->CLOS)
intel-speed-select-set-1  | ------------------------------------------------------------
intel-speed-select-set-1  | HP list should be clos:0
intel-speed-select-set-1  | cpu-0 clos:0
intel-speed-select-set-1  | … (showing first 40 lines)
intel-speed-select-set-1  |
intel-speed-select-set-1  | Non-HP list should be clos:2
intel-speed-select-set-1  | cpu-8 clos:2
intel-speed-select-set-1  | … (showing first 40 lines)
intel-speed-select-set-1  |
intel-speed-select-set-1  | Done.
intel-speed-select-set-1 exited with code 0
```

## 4. Debug / Manual Inspection (Optional)

This section is useful for **troubleshooting**, **validation**, or **manual experimentation**
with Intel® Speed Select and PCT behavior.

Start an interactive shell with the required tools installed:

```bash
docker compose run --rm intel-speed-select-shell
```
