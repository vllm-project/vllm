# Mooncake layerwise static delivery report

Original static delivery recorded on 2026-10-23 (Asia/Shanghai).
The control rollback described below supersedes its lifecycle claims; the
original commit/check tables are retained as historical evidence.

## Baseline and change history

- Delivery branch: `dev/mooncake-layerwise-capabilities`.
- Original local head: `42f7a6008829644aea25e35e50ea58fe98ff10a8`.
- PR #62504 head observed before development: `7913c96894cacc348397a377411acde96b63e68a`.
- Rebase target fetched for this delivery: `90aa823cbae28a560b5aecc88e110fcca70aa628`.
- Rebased original layerwise patch: `ed227fdbc053bfb16d73425f75cbafeeca1f5884`.
- Backup branch: `dev/mooncake-layerwise-before-capabilities-20261022`.

`git range-diff 42f7a60088^..42f7a60088 origin/main..ed227fdbc0` reported the
original patch unchanged. The original branch was preserved. No branch was
pushed and no PR content was updated during delivery.

| Commit | Reason |
| --- | --- |
| `a612b45d71` | Separate drained data from control submission; retain ACK context on send failure |
| `21912e97ae` | Bind immutable destination attempts and replay outcomes without another WRITE |
| `7998e1264d` | Hold old allocations through connector step completion, including empty steps |
| `c6268f702d` | Query/cancel on deadlines and surface unconfirmed DMA failures to the worker |
| `9c83af922b` | Identify PP endpoints separately and require full cache-name coverage |
| `aef73bd8fc` | Plan independent groups without flattening block identities or token units |
| `8012482ba5` | Preserve SWA logical positions and transfer the final required window |
| `dee764b89b` | Send final Mamba resident states without FA kernel-block expansion |
| `23c6e20517` | Transfer one CircularBuffer state page across ring wrap boundaries |
| `eb603a7727` | Route explicitly complete PCP replicas through source replica zero |
| `a9564c980c` | Close cross-feature identity/fence gaps, update regressions, and add pure mapping checks |
| `9870a7353e` | Ensure shared CI dependencies do not exclude KV-transfer source changes |
| `1f4f0ba82c` | Preserve full-manager prefix-cache contents for windowed FullAttentionSpec |

The original integration correction added both functional layout fixes and
lifecycle protection. The latter has since been withdrawn; see below.

## Control rollback and unresolved cases

At the author's request, the additional control safety changes from commits
1-4 and the control portions of commit 11 were removed, without rewriting
history. The max-tokens guard was removed too. Functional PP/group/cache/PCP
changes remain, including their readiness and completion aggregation.
`scheduler.py`, `base.py`, and the scheduler test file match `ed227fdbc0`
exactly. The cache planners and shared Mooncake functional hooks are unchanged.

The fence regression and six tests of the withdrawn safety mechanisms were
removed with their implementations. The original WRITE-failure regression was
restored. Existing feature tests remain; this is a contract rollback, not an
attempt to make failing safety tests pass.

These are source-derived review cases, not reproduced device failures:

| Situation | Behavior after rollback / unresolved risk |
| --- | --- |
| D client cancels while waiting for KV, P finishes normally | The original scheduler retains D blocks until receive completion. Cancellation alone does not immediately recycle them, but does not immediately stop P work either. |
| D receive times out or its control connection fails while P can still WRITE | Ordinary receive failure settles the peer task without a remote drain certificate. Once all tasks settle, cancelled requests can free blocks or live requests can recompute; old P writes may overlap reused memory. |
| P is cancelled/preempted during a forward/send, especially with overlapping PP batches | The new producer allocation fence is gone. Scheduler release can precede worker handling of the abort/preemption; executor block snapshots do not retain block-pool ownership. Source reads or model writes may touch reused blocks. |
| Late D-ready starts a transfer on a zero-token step | No connector-specific step fence covers this work. A concurrent source release is not prevented merely by the model thread collecting its future. |
| Final window/state handoff races scheduler trimming | The original `remove_skipped_blocks()` call runs before connector finalization. Final block-table capture cannot resurrect already retired pages; verify every SWA/Mamba/ring manager's actual lifetime. |
| Request is preempted/recomputed, or messages/identities are reused | Original preemption reset remains, but there is no generation/attempt check or immutable allocation ledger. Duplicate live ready is rejected, not replayed; late messages/results are not independently tied to an allocation generation. |
| ACK submission fails or ACK is lost | Context is removed before submission and there is no terminal replay/query. P may retire while D waits and eventually follows the ordinary timeout path. |
| One of several WRITEs fails after earlier WRITEs were submitted | A batch error clears progress and reports failure without a drain certificate. Request completion/error does not prove all possible writers stopped; source and destination release need review. |
| Device event or transport call never returns | `future.result()` and shutdown can block indefinitely; there is no layerwise wait deadline or explicit worker-fatal transition. |
| P decodes beyond the handoff while D-ready is late | No layerwise max-tokens guard remains (base model-specific behavior is unchanged). Final recurrent/ring state may advance before transfer; use the intended one-token P response contract. |
| Restart or mixed old/new connector revisions | There is no new attempt protocol/version rejection. Both peers must use the same revision; old writers must not be assumed harmless after D restarts. |

No replacement fix was implemented. Choosing which cases to support, reject, or
handle through deployment constraints is deferred to review.

Rollback checks: pinned Ruff lint/format, Python AST parsing, and
`git diff --check`; baseline comparison for the three restored files and
unchanged functional mapping/shared-hook files. Device-dependent connector and
scheduler tests were not run.

## Original delivery checks (before control rollback)

The environment has no GPU/NPU execution coverage for this delivery.

| Check | Result |
| --- | --- |
| Repository-pinned Ruff 0.14.0 lint on changed Python code/tests | Passed |
| Ruff format check on changed Python code/tests | Passed |
| Python AST parsing of changed Python files | Passed |
| `git diff --check` | Passed |
| Standalone, standard-library-only block mapping suite | 6 tests passed |
| Full pre-commit hook suite / repository-wide type checking | Not run |
| vLLM connector unit suite | Not run |
| Scheduler allocation-fence regression | Not run |
| GPU/NPU serving, payload inspection, faults, accuracy, performance | Not run |

Commands used (the available `uv` binary was invoked by absolute path):

```bash
uv tool run --from ruff==0.14.0 ruff check <changed-python-files>
uv tool run --from ruff==0.14.0 ruff format --check <changed-python-files>
.venv/bin/python tests/v1/kv_connector/unit/test_mooncake_layerwise_mapping.py
git diff --check
```

The standalone suite loads the actual standard-library-only mapping module via
`runpy`, avoiding package initialization. It does not replace torch or inject
fake vLLM modules. It checks independent block sizes, cached suffix offsets,
final tails, null-preserving window mapping, resident states, missing slots,
and ring wrap positions.

Existing host-logic connector tests were updated for component and worker
identities. Added regression sources cover terminal replay, ACK-send failure,
allocation replacement rejection, worker-visible receive failure, and real
scheduler allocation retention during cancellation. These tests remain
**unexecuted**; the older connector fixture still uses mocked events/transport
and a lightweight worker shell, so even passing it would not establish device
correctness. CI selection includes Mooncake/base connector changes; runtime
connector tests remain in the existing KV-connector unit job, and the scheduler
regression follows the existing scheduler suite's CPU-test selection.

## Device acceptance / future redesign cases

| Contract | Required evidence |
| --- | --- |
| Original FA behavior | Same-layer indexer, multi-chunk complete blocks, final tail, late ready, prefix hits, supported TP ratios |
| PP | P1/D2, P2/D1, different cuts with equal counts, empty intersections, absent required layers, all worker completions |
| Groups/storage | Different block sizes and prefixes, independent byte strides, shared storage, last physical page, null slots |
| SWA | Long prompts, moving windows, partial blocks, extra retained tokens, bounded replay, late ready |
| Mamba | h(N-1) handoff and D first consumption, none/align, all Conv/SSM components, checkpoint/speculative slot demand |
| CircularBuffer | Raw state bytes and logical handoff position before D consumption, wrap and compression boundaries |
| PCP | P1/DN, PN/D1, PN/DM with actual complete replicas, identical TP ownership, all-rank completion, non-replica rejection |
| Lifecycle | Pause WRITE with explicit synchronization; cancel/preempt/recompute; prove source/target allocations are not reused before drain |
| Failures | Lose ACK, replay ready, replace allocation, stale generation, one failed peer with another writing, timeout and worker-stop behavior |
| Product behavior | Real serving/proxy requests and deterministic output/logit comparisons against ordinary Mooncake; no unsupported accuracy claims |

Do not infer Mamba2/model-wide support, successful PCP collectives, or safe
transport recovery from source-level feasibility. Inspect the pinned transfer
engine's cancellation/registration teardown guarantees before validating
recovery. Throughput and overlap measurements follow correctness acceptance.
