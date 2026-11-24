# Kimi server source snapshot: attempt34

The retained result is attempt34 (2026-11-23 20:20:05 +08:00), four layers,
16 experts/top16, BF16, one 8-device server using <term>Ascend 950PR&950DT系列产品</term> (950DT). Actor/reference use
FSDP8/EP8/eFSDP1; vLLM uses TP8/EP8/DP1. Two steps each had 1024 valid response
tokens, no nonfinite values, no differing independent old/rollout logprobs and
zero mean/max absolute difference. Old/gradient and actor/reference checks each
covered 32 pairs / 2048 valid tokens with zero differences.

Same-side checks covered one image/prompt, eight generated positions, A/A,
actual batch1/batch2, and one full-prefill/decode corresponding history.
The actual scheduler target row did not change. Full BI/PD coverage (longer
trajectories, batch16, actual row reordering, chunked prefill) is NOT complete.
Reward, advantage and gradient norm were zero: effective updates and subsequent
weight synchronization are NOT validated. Neither 93-layer nor model-quality
or performance acceptance is claimed.

## Source provenance

The source branches retain the original server-era ancestry, not current main.
This is the user's requested source-review snapshot, not an upstream-ready port.
Inherited fork/baseline changes are not alignment fixes. Do not resolve conflicts
against current main by rewriting these source files.

Original provenance: verl uses wangdongleix/verl d854d22c3a01359970e6b0fb05a59ce74555d4dd;
vLLM uses ced6857afa0ea7b2e3f0846a62e1394e90f15607; FSDPTurbo uses GitCode
c43ebadabb74f85ca131189c8d439ab2fae617b6. Installed vLLM-Ascend reported
0.30.0rc2.dev252+g9a0b3fbd6; that exact public Git commit is unavailable.
Its Git scaffold is d1da39542603433dc8251b30f12af0131a74589e with inherited
runtime overlays and captured server source. The scaffold is NOT asserted to
be the installed runtime commit.

The run manifest hashed 53 files, all reproduced exactly across these four PRs.
Other package source files are a later read-only capture with capture-time hashes;
they are NOT retroactively covered by the attempt34 manifest. Source mtimes were
checked, but mtimes alone are not proof of historical content. The recovered
serving_invariance_probe.py matches its attempt34 hash. vLLM engine/core.py is
the pre-attempt35 backup, also byte-identical to the pre-task captured baseline;
its scheduler-budget method was added only for the unsuccessful attempt35.
Attempt35 failed HCCL initialization before probes and supplies no new acceptance.

Original runtime imports and APIs are preserved. Standalone helpers reside in
verl/examples/kimi_alignment/runtime and correspond to source/alignment_runtime
on the server. Checkpoint Python/config files reside in
FSDPTurbo/examples/kimi_alignment/checkpoint-code and correspond to the server's
model/ directory; do not merge these files into fsdp_turbo/models. The package
MoE source and the checkpoint MC2 adapter intentionally remain separate.
These are export storage locations, not changes to the server's loading path.

The three manifest-hashed launcher/environment scripts retain their original
absolute paths. Restore the original workspace layout to use them. Ancillary
scripts are current captured files, not independently historically attested;
record_run.py includes the later added core.py hash entry, without changing model
execution. No loader, import, signature, environment-variable or operator-layout
refactor has been introduced by this reconstruction.

Weights, tokenizer assets, compiled libraries, installed CANN custom-op artifacts,
and other external dependencies are not vendored. The installed CANN text-file
inventory is retained separately as provenance; it is not task-authored source.
This snapshot is not a self-contained environment or fresh-install validation.

No tests, runtime imports, training, inference, device jobs or writes to server423 were
performed for this update. Only source reading, SHA256/AST inspection, Git changes
and draft-PR updates were performed. New checks do not constitute numerical reruns.
