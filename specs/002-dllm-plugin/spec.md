# Feature Specification: dLLM (Blocked Masked Diffusion LLM) Plugin

**Feature Branch**: `002-dllm-plugin`  
**Created**: 2025-03-01  
**Status**: Draft  
**Input**: Integrate blocked masked diffusion LLMs (dLLMs) into vLLM as a plugin using the vLLM plugin system, so that each diffusion step is one worker iteration and existing continuous batching applies. Support block-based dLLM architectures (e.g. SDAR, LLaDA2.0/2.1, Fast-dLLMv2, WeDLM) via plugin registration without modifying vLLM core.

## Context

Block-based dLLMs generate text in fixed-size blocks (e.g. 32 tokens) using mask-then-fill decoding. Each inference step consumes one block; the model returns a variable number of “committed” tokens (0 to block size) and the next block’s input. This fits naturally with “one step, one forward” and continuous batching. vLLM’s [plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/) allows extending vLLM with custom models via entry points (e.g. `vllm.general_plugins`) and model registration. This feature specifies delivering dLLM support **as a plugin**: users install a plugin package to run dLLM models; new dLLM architectures can be added by publishing new plugins without changing vLLM’s core codebase.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run dLLM inference via an installed plugin (Priority: P1)

As a user, I want to install a dLLM plugin package and run vLLM with a supported dLLM model (e.g. LLaDA2.0) so that I get correct block-step inference: each step corresponds to one forward over a block, and the generated sequence advances by 0 to block-size tokens per step according to the model’s commit rule, without modifying vLLM source code.

**Why this priority**: This is the minimal viable outcome; without it, plugin-based dLLM support does not exist.

**Independent Test**: Install a dLLM plugin, start vLLM with a model name provided by that plugin, send one request, and verify that (1) the model loads, (2) generation completes with total output length = prompt length + sum of committed tokens per step, and (3) step semantics match block-based decoding (one forward per step, variable tokens per step).

**Acceptance Scenarios**:

1. **Given** a published dLLM plugin package and vLLM installed, **When** the user installs the plugin and starts the server with the plugin’s model name, **Then** the server loads the model and accepts requests.
2. **Given** a running vLLM instance with a dLLM model from a plugin, **When** the user sends a completion request, **Then** the response contains generated text and the number of tokens produced is consistent with block-step semantics (e.g. prompt length + committed tokens across steps).
3. **Given** the same setup, **When** the user does not install the plugin, **Then** the model name is not available (or a clear error indicates the model is not supported), so that plugin boundary is explicit.

---

### User Story 2 - Register multiple dLLM architectures from one plugin (Priority: P2)

As a plugin author, I want my plugin to register one or more dLLM model architectures (e.g. LLaDA2.0 and LLaDA2.1) with vLLM so that users can choose among them by model name without installing multiple packages.

**Why this priority**: Reduces fragmentation and simplifies deployment when several related dLLM variants exist.

**Independent Test**: Ship a plugin that registers two dLLM model names; install the plugin and verify both model names are available and runnable with correct block-step behavior.

**Acceptance Scenarios**:

1. **Given** a plugin that registers multiple dLLM architectures, **When** the plugin is loaded by vLLM, **Then** all registered model names appear as supported (or are discoverable) and can be selected at startup.
2. **Given** two different dLLM model names from the same plugin, **When** the user runs inference with each model in turn, **Then** each run produces correct block-step behavior for that architecture (e.g. correct block size and commit semantics).

---

### User Story 3 - Batching and prefix caching with dLLM plugin (Priority: P2)

As a user, I want multiple requests to a dLLM model (loaded via plugin) to be batched in the same step where possible, and prefix caching to behave correctly (valid only up to committed length; disabled when the model uses fully bidirectional prefill) so that throughput and memory use match expectations.

**Why this priority**: Ensures plugin-based dLLMs integrate with vLLM’s batching and caching instead of degrading to single-request behavior.

**Independent Test**: Run multiple concurrent requests against a dLLM model from a plugin; verify batching occurs (e.g. multiple requests in one step). If the model supports prefix caching, verify that only committed prefix is reused; if the model uses bidirectional prefill, verify prefix caching is disabled or documented.

**Acceptance Scenarios**:

1. **Given** several concurrent requests to the same dLLM model (from a plugin), **When** the engine schedules them, **Then** multiple requests can be processed in the same inference step where the engine supports it.
2. **Given** a dLLM model that allows prefix reuse (non-bidirectional prefill), **When** two requests share the same prompt prefix, **Then** prefix caching applies only up to the committed length and does not reuse uncommitted tail positions.
3. **Given** a dLLM model that uses a fully bidirectional prefill mask, **When** the user runs inference, **Then** prefix caching is not used for that model (or is clearly documented as unsupported), so that correctness is preserved.

---

### Edge Cases

- What happens when the user specifies a dLLM model name but no plugin providing that model is installed? The system should report a clear error (e.g. unknown model or “install plugin X”).
- What happens when two plugins register the same model name? Behavior should be deterministic (e.g. first-wins or last-wins) and documented; conflicts should be avoidable by naming.
- How does the system behave when the plugin’s model returns invalid step output (e.g. wrong number of committed or next-block tokens)? The engine or plugin should validate and fail the request or step with a clear error rather than corrupting state.
- How is version compatibility between the plugin and vLLM handled? Plugins target a vLLM version; mismatches should be detectable (e.g. at load time) and documented so users can align versions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support loading block-based dLLM models that are registered by a plugin (via the vLLM plugin mechanism) so that users can run dLLM inference without changing vLLM core code.
- **FR-002**: When a dLLM model from a plugin is loaded, the system MUST run it with block-step semantics: one inference step per diffusion step, variable committed tokens (0 to block size) per step, and fixed-size next-block input, so that behavior matches the block-based dLLM contract.
- **FR-003**: The system MUST allow a single plugin to register multiple model architectures (e.g. multiple model names) so that one package can provide several dLLM variants.
- **FR-004**: The system MUST integrate batching for dLLM requests (same model from plugin) so that multiple requests can be processed in the same step where the engine supports it.
- **FR-005**: The system MUST apply prefix caching only where valid (e.g. up to committed length for that model) and MUST disable or disallow prefix caching when the model uses a fully bidirectional prefill mask, so that correctness is preserved.
- **FR-006**: The system MUST surface a clear error when a user requests a dLLM model that no installed plugin provides, so that users know to install the appropriate plugin.
- **FR-007**: Plugin registration MUST follow the documented plugin contract (e.g. entry point group, re-entrant registration) so that plugins load reliably across processes and environments.

### Key Entities

- **dLLM plugin**: A separate installable package that registers one or more block-based dLLM model architectures with vLLM via the plugin system; it is discovered by vLLM at runtime and must conform to the plugin and model contracts.
- **Block-based dLLM model**: A model that produces text in fixed-size blocks, returning per step a variable number of committed tokens and a fixed-size next-block input; it may have model-specific attention masks and prefill behavior (e.g. bidirectional prefill disables prefix caching).
- **Block step**: One diffusion step = one inference step over one block of tokens; the engine advances the sequence by 0 to block-size tokens per step based on the model’s committed output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can run at least one block-based dLLM model (e.g. LLaDA2.0 or a reference stub) by installing a plugin and starting vLLM with that model name, without modifying vLLM source code.
- **SC-002**: Generated output length matches block-step semantics (prompt length + sum of committed tokens per step) for every request to a plugin-registered dLLM model.
- **SC-003**: Adding a new dLLM architecture does not require changes to vLLM core; it can be delivered by publishing and installing a new (or updated) plugin.
- **SC-004**: When multiple requests are sent to the same plugin-registered dLLM model, batching occurs where the engine supports it, so that throughput is not limited to single-request execution.
- **SC-005**: Prefix caching is either applied correctly (up to committed length) or disabled (e.g. for bidirectional prefill models), with behavior documented so that users and plugin authors can reason about memory and reuse.

## Assumptions

- vLLM’s plugin system (e.g. `vllm.general_plugins`, model registration) remains the supported extension mechanism for adding custom models; this feature does not introduce a new plugin type unless the existing mechanism is insufficient for dLLM.
- One model per vLLM instance remains the rule; when a dLLM model from a plugin is loaded, all requests in that instance are for that model (no mixing of dLLM and non-dLLM models in the same batch).
- Plugin authors are responsible for ensuring their plugin and model implementation match the block-based dLLM contract (block size, committed tokens, next-block input) and for documenting prefill mask behavior (e.g. bidirectional vs not) so that prefix caching can be applied or disabled correctly.
- Compatibility between a plugin and a vLLM version is the responsibility of the plugin author; the system may provide detection or documentation to reduce version mismatch issues.
