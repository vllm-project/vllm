# Remote draft placement (experimental)

Remote placement separates a target-conditioned speculator from the target
process. The target remains responsible for verification and sampling. Placement
is configured independently of the speculative method; native local execution
remains the default.

This foundation provides configuration, feature negotiation, bounded inline
frames and request lifecycles. It deliberately rejects GPU serving until the
standalone GPU runtime is installed. The protocol test adapter is for development,
not a model-serving backend.

## Configuration and lifecycle

`speculative_config.remote_draft` takes a single `endpoint`, a `transport`
(`auto` or `inline`) and a `failure_policy` (`target_only` or `error`). `auto`
selects inline. Use a Unix socket or loopback TCP endpoint on the same host.
The protocol does not authenticate peers; non-loopback server binding requires
explicit `--allow-remote-connections`.

The default request budget is 100 ms and the initial connection budget is
30,000 ms. A request timeout under `target_only` permanently disables remote
speculation for affected requests, returning zero valid drafts while the target
continues. `error` surfaces the failure. Reconnection may admit new identities;
it does not rebuild the old requests' draft KV. Generation and session fencing
prevent late replies from reactivating retired state. Replies for another session,
request, or batch are discarded before their ordering counters or draft tokens are
used.

Feature schema and frame byte budgets are checked before tensor serialization.
Connections, partial frames, pending bytes and adapter waiters have server-side
bounds. A shutdown grace timeout does not authorize freeing GPU storage whose
completion is unknown; the process must be restarted in that state.

This placement does not provide predictive caching, cross-node execution or
N:M routing. Moving a speculator to another GPU does not by itself overlap its
computation with target verification or guarantee higher throughput.
