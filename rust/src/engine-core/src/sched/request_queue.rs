// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/core/sched/request_queue.py`.
//!
//! Queues hold request handles (e.g. slab keys) rather than requests, so the
//! scheduler can own the requests. Python's queues hold `Request` objects
//! compared by identity, which handles reproduce with `Eq`. The priority
//! queue reads each request through a [`RequestLookup`] on every comparison,
//! like Python's `Request.__lt__` reads live attributes.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, Hash};

use rustc_hash::FxHashSet;

use super::heapq;
use super::request::Request;

/// Enum for scheduling policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    Fcfs,
    Priority,
}

impl SchedulingPolicy {
    /// The policy named as in `SchedulingPolicy(value)`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "fcfs" => Some(Self::Fcfs),
            "priority" => Some(Self::Priority),
            _ => None,
        }
    }
}

/// Resolves a request handle to the request it names.
pub trait RequestLookup<H> {
    fn request(&self, handle: H) -> &Request;
}

impl RequestLookup<usize> for [Request] {
    fn request(&self, handle: usize) -> &Request {
        &self[handle]
    }
}

impl RequestLookup<usize> for Vec<Request> {
    fn request(&self, handle: usize) -> &Request {
        &self[handle]
    }
}

impl<H: Eq + Hash, S: BuildHasher> RequestLookup<H> for HashMap<H, Request, S> {
    fn request(&self, handle: H) -> &Request {
        &self[&handle]
    }
}

impl<H, T: RequestLookup<H> + ?Sized> RequestLookup<H> for &T {
    fn request(&self, handle: H) -> &Request {
        (**self).request(handle)
    }
}

/// A first-come-first-served queue that supports deque operations.
#[derive(Clone, Debug)]
pub struct FcfsRequestQueue<H> {
    queue: VecDeque<H>,
}

impl<H> Default for FcfsRequestQueue<H> {
    fn default() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl<H: Copy + Eq + Hash> FcfsRequestQueue<H> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a request to the queue according to FCFS policy.
    pub fn add_request(&mut self, request: H) {
        self.queue.push_back(request);
    }

    /// Pop a request from the queue according to FCFS policy.
    pub fn pop_request(&mut self) -> Option<H> {
        self.queue.pop_front()
    }

    /// Peek at the next request in the queue without removing it.
    pub fn peek_request(&self) -> Option<H> {
        self.queue.front().copied()
    }

    /// Prepend a request to the front of the queue.
    pub fn prepend_request(&mut self, request: H) {
        self.queue.push_front(request);
    }

    /// Prepend all requests from another queue to the front of this
    /// queue.
    ///
    /// Note: The requests will be prepended in reverse order of their
    /// appearance in the `requests` queue.
    pub fn prepend_requests(&mut self, requests: impl IntoIterator<Item = H>) {
        for request in requests {
            self.queue.push_front(request);
        }
    }

    /// Remove a specific request from the queue.
    ///
    /// Panics if the request is not queued (`ValueError` in Python).
    pub fn remove_request(&mut self, request: H) {
        let pos = self
            .queue
            .iter()
            .position(|&r| r == request)
            .expect("deque.remove(x): x not in deque");
        self.queue.remove(pos);
    }

    /// Remove multiple specific requests from the queue.
    pub fn remove_requests(&mut self, requests: impl IntoIterator<Item = H>) {
        let requests_to_remove: FxHashSet<H> = requests.into_iter().collect();
        self.queue.retain(|r| !requests_to_remove.contains(r));
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Iterate over the queue according to FCFS policy.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = H> + '_ {
        self.queue.iter().copied()
    }
}

/// A priority queue that supports heap operations.
///
/// Respects the ordering defined in the Request class, where
/// requests with a smaller value of `priority` are processed first.
/// If multiple requests have the same priority, the one with the earlier
/// `arrival_time` is processed first.
#[derive(Clone, Debug)]
pub struct PriorityRequestQueue<H> {
    heap: Vec<H>,
}

impl<H> Default for PriorityRequestQueue<H> {
    fn default() -> Self {
        Self { heap: Vec::new() }
    }
}

fn request_lt<H: Copy>(store: &(impl RequestLookup<H> + ?Sized)) -> impl Fn(&H, &H) -> bool + '_ {
    move |&a, &b| store.request(a).lt(store.request(b))
}

impl<H: Copy + Eq + Hash> PriorityRequestQueue<H> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a request to the queue according to priority policy.
    pub fn add_request(&mut self, request: H, store: &(impl RequestLookup<H> + ?Sized)) {
        heapq::heappush(&mut self.heap, request, request_lt(store));
    }

    /// Pop a request from the queue according to priority policy.
    pub fn pop_request(&mut self, store: &(impl RequestLookup<H> + ?Sized)) -> Option<H> {
        heapq::heappop(&mut self.heap, request_lt(store))
    }

    /// Peek at the next request in the queue without removing it.
    pub fn peek_request(&self) -> Option<H> {
        self.heap.first().copied()
    }

    /// Add a request to the queue according to priority policy.
    ///
    /// Note: In a priority queue, there is no concept of prepending to the
    /// front. Requests are ordered by (priority, arrival_time).
    pub fn prepend_request(&mut self, request: H, store: &(impl RequestLookup<H> + ?Sized)) {
        self.add_request(request, store);
    }

    /// Add all requests from another queue according to priority policy.
    ///
    /// Note: In a priority queue, there is no concept of prepending to the
    /// front. Requests are ordered by (priority, arrival_time).
    pub fn prepend_requests(
        &mut self,
        requests: impl IntoIterator<Item = H>,
        store: &(impl RequestLookup<H> + ?Sized),
    ) {
        for request in requests {
            self.add_request(request, store);
        }
    }

    /// Remove a specific request from the queue.
    ///
    /// Panics if the request is not queued (`ValueError` in Python).
    pub fn remove_request(&mut self, request: H, store: &(impl RequestLookup<H> + ?Sized)) {
        let pos = self
            .heap
            .iter()
            .position(|&r| r == request)
            .expect("list.remove(x): x not in list");
        self.heap.remove(pos);
        heapq::heapify(&mut self.heap, request_lt(store));
    }

    /// Remove multiple specific requests from the queue.
    pub fn remove_requests(
        &mut self,
        requests: impl IntoIterator<Item = H>,
        store: &(impl RequestLookup<H> + ?Sized),
    ) {
        let requests_to_remove: FxHashSet<H> = requests.into_iter().collect();
        self.heap.retain(|r| !requests_to_remove.contains(r));
        heapq::heapify(&mut self.heap, request_lt(store));
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Iterate over the queue according to priority policy.
    ///
    /// Pops from a copy of the heap, lazily, as Python's generator does.
    pub fn iter<'a, S: RequestLookup<H> + ?Sized>(
        &self,
        store: &'a S,
    ) -> impl Iterator<Item = H> + use<'a, H, S> {
        let mut heap_copy = self.heap.clone();
        std::iter::from_fn(move || heapq::heappop(&mut heap_copy, request_lt(store)))
    }

    /// The heap list in its internal order (Python's `_heap`).
    pub fn as_heap(&self) -> &[H] {
        &self.heap
    }
}

/// A request queue for either [`SchedulingPolicy`]
/// (`create_request_queue` in Python).
///
/// Methods take the request store for the priority policy's comparisons;
/// the FCFS queue ignores it.
#[derive(Clone, Debug)]
pub enum RequestQueue<H> {
    Fcfs(FcfsRequestQueue<H>),
    Priority(PriorityRequestQueue<H>),
}

impl<H: Copy + Eq + Hash> RequestQueue<H> {
    /// Create request queue based on scheduling policy.
    pub fn new(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::Priority => Self::Priority(PriorityRequestQueue::new()),
            SchedulingPolicy::Fcfs => Self::Fcfs(FcfsRequestQueue::new()),
        }
    }

    /// Add a request to the queue according to the policy.
    pub fn add_request(&mut self, request: H, store: &(impl RequestLookup<H> + ?Sized)) {
        match self {
            Self::Fcfs(q) => q.add_request(request),
            Self::Priority(q) => q.add_request(request, store),
        }
    }

    /// Pop a request from the queue according to the policy. `None` when
    /// empty (`IndexError` in Python).
    pub fn pop_request(&mut self, store: &(impl RequestLookup<H> + ?Sized)) -> Option<H> {
        match self {
            Self::Fcfs(q) => q.pop_request(),
            Self::Priority(q) => q.pop_request(store),
        }
    }

    /// Peek at the request at the front of the queue without removing it.
    /// `None` when empty (`IndexError` in Python).
    pub fn peek_request(&self) -> Option<H> {
        match self {
            Self::Fcfs(q) => q.peek_request(),
            Self::Priority(q) => q.peek_request(),
        }
    }

    /// Prepend a request to the front of the queue.
    pub fn prepend_request(&mut self, request: H, store: &(impl RequestLookup<H> + ?Sized)) {
        match self {
            Self::Fcfs(q) => q.prepend_request(request),
            Self::Priority(q) => q.prepend_request(request, store),
        }
    }

    /// Prepend all requests from another queue to the front of this
    /// queue.
    pub fn prepend_requests(
        &mut self,
        requests: impl IntoIterator<Item = H>,
        store: &(impl RequestLookup<H> + ?Sized),
    ) {
        match self {
            Self::Fcfs(q) => q.prepend_requests(requests),
            Self::Priority(q) => q.prepend_requests(requests, store),
        }
    }

    /// Remove a specific request from the queue.
    pub fn remove_request(&mut self, request: H, store: &(impl RequestLookup<H> + ?Sized)) {
        match self {
            Self::Fcfs(q) => q.remove_request(request),
            Self::Priority(q) => q.remove_request(request, store),
        }
    }

    /// Remove multiple specific requests from the queue.
    pub fn remove_requests(
        &mut self,
        requests: impl IntoIterator<Item = H>,
        store: &(impl RequestLookup<H> + ?Sized),
    ) {
        match self {
            Self::Fcfs(q) => q.remove_requests(requests),
            Self::Priority(q) => q.remove_requests(requests, store),
        }
    }

    /// Get number of requests in queue.
    pub fn len(&self) -> usize {
        match self {
            Self::Fcfs(q) => q.len(),
            Self::Priority(q) => q.len(),
        }
    }

    /// Check if queue has no requests (`not queue` in Python).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over the queue according to the policy.
    pub fn iter<'a, S: RequestLookup<H> + ?Sized>(
        &'a self,
        store: &'a S,
    ) -> impl Iterator<Item = H> + 'a {
        let (fcfs, priority) = match self {
            Self::Fcfs(q) => (Some(q.iter()), None),
            Self::Priority(q) => (None, Some(q.iter(store))),
        };
        fcfs.into_iter().flatten().chain(priority.into_iter().flatten())
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::sched::request::NewRequest;

    #[derive(Deserialize)]
    struct Case {
        policy: String,
        requests: Vec<CaseRequest>,
        ops: Vec<Op>,
    }

    #[derive(Deserialize)]
    struct CaseRequest {
        id: String,
        priority: i64,
        arrival_time: f64,
    }

    #[derive(Deserialize)]
    struct Op {
        op: String,
        args: Vec<serde_json::Value>,
        result: serde_json::Value,
        state: Vec<usize>,
    }

    fn arg(op: &Op, i: usize) -> usize {
        op.args[i].as_u64().unwrap() as usize
    }

    /// Replays random operation sequences recorded from Python's
    /// `FCFSRequestQueue` / `PriorityRequestQueue` and compares every result
    /// and the internal deque / heap order after each step.
    #[test]
    fn matches_python_queues() {
        let cases: Vec<Case> = serde_json::from_str(include_str!(
            "../../tests/fixtures/request_queue_cases.json"
        ))
        .unwrap();
        for (case_idx, case) in cases.iter().enumerate() {
            let mut store: Vec<Request> = case
                .requests
                .iter()
                .map(|r| {
                    Request::new(NewRequest {
                        priority: r.priority,
                        arrival_time: Some(r.arrival_time),
                        ..NewRequest::for_test(&r.id, vec![1])
                    })
                    .unwrap()
                })
                .collect();
            let policy = SchedulingPolicy::from_name(&case.policy).unwrap();
            let mut queue = RequestQueue::<usize>::new(policy);
            for (op_idx, op) in case.ops.iter().enumerate() {
                let ctx = format!("case {case_idx} op {op_idx} ({})", op.op);
                let all_args = || op.args.iter().map(|a| a.as_u64().unwrap() as usize);
                let result = match op.op.as_str() {
                    "add" => {
                        queue.add_request(arg(op, 0), &store);
                        serde_json::Value::Null
                    }
                    "prepend" => {
                        queue.prepend_request(arg(op, 0), &store);
                        serde_json::Value::Null
                    }
                    "prepend_many" => {
                        queue.prepend_requests(all_args(), &store);
                        serde_json::Value::Null
                    }
                    "pop" => queue.pop_request(&store).into(),
                    "peek" => queue.peek_request().into(),
                    "remove" => {
                        queue.remove_request(arg(op, 0), &store);
                        serde_json::Value::Null
                    }
                    "remove_many" => {
                        queue.remove_requests(all_args(), &store);
                        serde_json::Value::Null
                    }
                    "iter" => queue.iter(&store).collect::<Vec<_>>().into(),
                    "set_arrival" => {
                        store[arg(op, 0)].arrival_time = op.args[1].as_f64().unwrap();
                        serde_json::Value::Null
                    }
                    "set_priority" => {
                        store[arg(op, 0)].priority = op.args[1].as_i64().unwrap();
                        serde_json::Value::Null
                    }
                    other => panic!("unknown op {other}"),
                };
                assert_eq!(result, op.result, "{ctx}");
                let state: Vec<usize> = match &queue {
                    RequestQueue::Fcfs(q) => q.iter().collect(),
                    RequestQueue::Priority(q) => q.as_heap().to_vec(),
                };
                assert_eq!(state, op.state, "{ctx}");
                assert_eq!(queue.len(), op.state.len(), "{ctx}");
            }
        }
    }

    fn requests(specs: &[(&str, i64, f64)]) -> Vec<Request> {
        specs
            .iter()
            .map(|&(id, priority, arrival_time)| {
                Request::new(NewRequest {
                    priority,
                    arrival_time: Some(arrival_time),
                    ..NewRequest::for_test(id, vec![1])
                })
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn fcfs_prepend_requests_reverses() {
        let store = requests(&[("a", 0, 0.0), ("b", 0, 0.0), ("c", 0, 0.0)]);
        let mut q = RequestQueue::new(SchedulingPolicy::Fcfs);
        q.add_request(2, &store);
        q.prepend_requests([0, 1], &store);
        assert_eq!(q.iter(&store).collect::<Vec<_>>(), [1, 0, 2]);
        assert_eq!(q.pop_request(&store), Some(1));
        q.remove_request(2, &store);
        assert_eq!(q.peek_request(), Some(0));
        q.remove_requests([0, 1], &store);
        assert!(q.is_empty());
        assert_eq!(q.pop_request(&store), None);
        assert_eq!(q.peek_request(), None);
    }

    #[test]
    fn priority_orders_by_priority_then_arrival() {
        let store = requests(&[
            ("low", 2, 0.0),
            ("late", 1, 5.0),
            ("early", 1, 1.0),
            ("high", 0, 9.0),
        ]);
        let mut q = RequestQueue::new(SchedulingPolicy::Priority);
        q.prepend_requests(0..4, &store);
        assert_eq!(q.peek_request(), Some(3));
        assert_eq!(q.iter(&store).collect::<Vec<_>>(), [3, 2, 1, 0]);
        // Iterating does not consume.
        assert_eq!(q.len(), 4);
        q.remove_request(2, &store);
        let mut popped = Vec::new();
        while let Some(r) = q.pop_request(&store) {
            popped.push(r);
        }
        assert_eq!(popped, [3, 1, 0]);
    }
}
