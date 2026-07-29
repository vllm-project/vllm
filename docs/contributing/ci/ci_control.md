# `/ci` control API

This is the user contract for the new pull-request comment API. The command
gateway must be deployed from `ci-infra` before these commands take effect.
Repository-owned policy lives in
[`.buildkite/ci_control.toml`](../../../.buildkite/ci_control.toml); job commands,
dependencies, and secrets remain in the existing CI definitions.

The existing exact `/ci run` and `/ci retry` commands remain on the legacy
workflow during the read-only rollout. At cutover, repository administrators
set `CI_CONTROL_GATEWAY_MODE=active`, which disables that workflow before the
new gateway enables mutations. The two mutation paths must never be active at
the same time because the legacy path has no credit ledger.

Use one exact, lowercase command per PR comment. Editing a processed comment
does not run it again.

## Commands

| Command | What it does |
| --- | --- |
| `/ci help` | Show syntax and valid selectors. |
| `/ci status` | Summarize the exact current PR HEAD, latest `main` snapshot, and your credits. |
| `/ci status main [<selection>]` | Show active, recovering, and recently resolved failures on `main`. |
| `/ci status pr [<selection>]` | Classify each failure from completed CI on the exact current PR HEAD. |
| `/ci status request:<id>` | Show one run or retry request and its cost. |
| `/ci status refresh` | Refresh CI evidence without rerunning tests. |
| `/ci list groups` | List selectable groups. |
| `/ci list areas` | List stable test areas. |
| `/ci list jobs` | List stable job keys and their group/area membership. |
| `/ci plan <selection>` | Show exact jobs, dependencies, and cost without running them. |
| `/ci run <selection>` | Run selected CI on the exact current PR HEAD. |
| `/ci retry failures [<selection>]` | Retry each matching current failure once. |
| `/ci credits` | Show available, reserved, granted, and spent credits. |
| `/ci credits add @user <amount> <reason>` | Add an audited credit grant; maintainers/admins only. |

Unknown commands or selectors do no work and return help.

## Selectors

```text
all
groups:<group>[,<group>...]
areas:<area>[,<area>...] [groups:<group>[,<group>...]]
jobs:<stable-job-key>[,<stable-job-key>...]
```

The initial groups are:

- `upstream`: vLLM-owned upstream tests;
- `cpu`: CPU-only tests; and
- `amd`: keyed AMD mirrors plus the native AMD lane.

Comma-separated values are alternatives; different selector types are
intersected. For example, `groups:upstream,cpu areas:models` selects model tests
from either catalog. A job selected more than once runs once.

The legacy native AMD pipeline is represented by the single
`native-amd-lane` job and `native-amd` area. Either name selects the whole
lane. Its individual jobs cannot be selected by area or job until that
generated pipeline publishes stable metadata. Keyed AMD mirror jobs remain
individually selectable, including by their inherited test areas.
Status for the native lane is correspondingly lane-level, not a claim about
the identity of an individual native job.

Job keys and area names are public API. The catalog check compares a PR with
its base checkout; a rename requires an alias and a removal requires a
tombstone in `ci_control.toml`. Derived AMD mirror keys are checked too.

Examples:

```text
/ci run groups:cpu
/ci run groups:upstream areas:attention,models
/ci run groups:amd
/ci retry failures groups:cpu
/ci plan jobs:cpu-kernel-tests
/ci credits add @alice 100 investigate intermittent AMD failures
```

## Credits and access

Each eligible GitHub user receives a one-time balance of 300 job credits. One
credit pays for one executable job attempt or parallel shard accepted by the
provider. Retries and newly required dependencies cost again. Read commands,
wait/group nodes, skipped jobs, and ordinary automatic CI are free. There is
no automatic reset in version 1.

Insufficient credit rejects the complete plan; it never runs an arbitrary
subset. `/ci retry failures` has no lineage-depth ceiling, but one comment
still creates at most one new attempt for each current failure and still
requires credits.

Read-only status and catalog views may be public. Commands that mutate
authoritative state or request compute require live `write` access or
membership in the vLLM committer team. Credit grants require
`maintain` or `admin` access, and the recipient must independently have
`write` access or committer membership.

## Understanding failures

Buildkite is the source of run facts. Completion webhooks update status
quickly, while an overlapping scheduled reconciler repairs missed or
out-of-order events. `/ci status refresh` queues that same reconciliation; it
does not run tests. An update is conclusive only after every provider page,
expected shard, and retry predecessor is accounted for.

`/ci status main` reports an immutable evidence snapshot, its per-lane
watermarks, freshness, and incident state. A group becomes known after
failures on two distinct canonical `main` SHAs. After a known failure starts
passing, it remains recovering until three clean distinct SHAs resolve it.
Retries and rebuilds of one SHA count once. A force-push starts a new main
epoch, so unconfirmed failure and recovery counters do not cross rewritten
history. A changed executable test definition also starts fresh confirmation
counters. Evidence older than 72 hours is stale and cannot prove that a
failure is PR-only.

`/ci status pr` compares each failure independently with one immutable `main`
snapshot and reports one of:

- known on `main`;
- candidate on `main`;
- seen flaky on `main`;
- recovering or resolved on `main`;
- a different failure on `main`;
- the same group is red but the underlying failure is unverified;
- not matched on fresh, compatible `main` evidence from the PR base ancestry;
  or
- unable to classify because evidence is missing, partial, stale, or
  incompatible.

A red group on both branches is not enough to call failures equal. The API
uses exact fingerprints only when the lane produces them and the execution
profile and test-definition digest match. Otherwise it reports the weaker
group-level result explicitly. Mixed groups keep each result visible, and all
classifications are advisory.
