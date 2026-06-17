//! Tracking of the external→internal request-id mapping for in-flight requests.
//!
//! When request-id randomization is enabled (the default), [`crate::Llm`]
//! rewrites the external (user-supplied) request id into a unique internal
//! engine id before reaching engine-core. Engine-core only ever knows the
//! internal id, so aborting a request by its external id requires resolving it
//! back to the internal id(s) first.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

/// external id → internal id → number of live guards holding that edge.
type InflightMap = HashMap<String, HashMap<String, usize>>;

/// Maps external (user-supplied) request ids to the set of live internal engine
/// request ids they currently expand into.
///
/// One external id may map to multiple internal ids: duplicate external ids
/// submitted concurrently each get their own randomized internal id, and an
/// abort by the shared external id must reach all of them. Edges are
/// refcounted: with randomization disabled the same (external, internal) pair
/// can be tracked by several guards in sequence (e.g. a finished request whose
/// stream is still held alongside a fresh submission reusing the id), and the
/// edge must survive until the last guard drops.
#[derive(Default)]
pub(crate) struct InflightRequests {
    map: Arc<Mutex<InflightMap>>,
}

impl InflightRequests {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record that `internal` is now an in-flight engine request for the
    /// `external` request id, returning a guard that removes the edge when the
    /// request's output stream is dropped (on clean finish or cancellation).
    pub(crate) fn track(&self, external: String, internal: String) -> RequestGuard {
        *self
            .map
            .lock()
            .entry(external.clone())
            .or_default()
            .entry(internal.clone())
            .or_insert(0) += 1;
        RequestGuard {
            map: Arc::downgrade(&self.map),
            external,
            internal,
        }
    }

    /// Resolve external request ids to the internal engine ids currently
    /// in-flight for them. Unknown or already-finished ids contribute nothing.
    pub(crate) fn resolve(&self, external_ids: &[String]) -> Vec<String> {
        let map = self.map.lock();
        external_ids
            .iter()
            .filter_map(|external| map.get(external))
            .flat_map(|internal_ids| internal_ids.keys())
            .cloned()
            .collect()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.map.lock().is_empty()
    }
}

/// RAII guard that releases one refcount on a single external→internal edge
/// when dropped, removing the edge once no live guard holds it.
///
/// Held by the per-request output stream, so cleanup runs whether the stream
/// terminates cleanly or is cancelled. A [`Weak`] handle is used so a stream
/// outliving its owning [`InflightRequests`] does not keep the map alive.
pub(crate) struct RequestGuard {
    map: Weak<Mutex<InflightMap>>,
    external: String,
    internal: String,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let Some(map) = self.map.upgrade() else {
            return;
        };
        let mut map = map.lock();
        if let Some(internal_ids) = map.get_mut(&self.external) {
            if let Some(count) = internal_ids.get_mut(&self.internal) {
                *count -= 1;
                if *count == 0 {
                    internal_ids.remove(&self.internal);
                }
            }
            if internal_ids.is_empty() {
                map.remove(&self.external);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_external_to_internal() {
        let inflight = InflightRequests::new();
        let _guard = inflight.track("ext".to_string(), "ext-abc".to_string());

        assert_eq!(
            inflight.resolve(&["ext".to_string()]),
            vec!["ext-abc".to_string()]
        );
        assert!(inflight.resolve(&["unknown".to_string()]).is_empty());
    }

    #[test]
    fn one_external_maps_to_many_internal() {
        let inflight = InflightRequests::new();
        let _g1 = inflight.track("dup".to_string(), "dup-1".to_string());
        let _g2 = inflight.track("dup".to_string(), "dup-2".to_string());

        let mut resolved = inflight.resolve(&["dup".to_string()]);
        resolved.sort();
        assert_eq!(resolved, vec!["dup-1".to_string(), "dup-2".to_string()]);
    }

    #[test]
    fn dropping_guard_removes_only_its_own_edge_then_cleans_empty_key() {
        let inflight = InflightRequests::new();
        let g1 = inflight.track("dup".to_string(), "dup-1".to_string());
        let g2 = inflight.track("dup".to_string(), "dup-2".to_string());

        drop(g1);
        assert_eq!(
            inflight.resolve(&["dup".to_string()]),
            vec!["dup-2".to_string()]
        );

        drop(g2);
        assert!(inflight.resolve(&["dup".to_string()]).is_empty());
        assert!(
            inflight.is_empty(),
            "empty external key must be removed, not left dangling"
        );
    }

    #[test]
    fn identical_edges_are_refcounted_across_guards() {
        // With request-id randomization disabled, internal == external, so two
        // tracked requests can share the exact same edge. Dropping one guard
        // (e.g. a stale stream, or the error path of a rejected duplicate
        // submission) must not untrack the other still-live request.
        let inflight = InflightRequests::new();
        let g1 = inflight.track("x".to_string(), "x".to_string());
        let g2 = inflight.track("x".to_string(), "x".to_string());

        drop(g1);
        assert_eq!(inflight.resolve(&["x".to_string()]), vec!["x".to_string()]);

        drop(g2);
        assert!(inflight.resolve(&["x".to_string()]).is_empty());
        assert!(inflight.is_empty());
    }

    #[test]
    fn guard_drop_is_a_noop_after_inflight_is_gone() {
        let guard = {
            let inflight = InflightRequests::new();
            inflight.track("ext".to_string(), "ext-abc".to_string())
        };
        // Dropping the guard after the owning map is gone must not panic.
        drop(guard);
    }
}
