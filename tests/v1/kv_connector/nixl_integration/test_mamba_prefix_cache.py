# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify D-side prefix cache hits reduce transfer for Mamba hybrid PD.

Sends the same long prompt twice through P/D and asserts that the second
request transfers fewer bytes (because cached blocks are skipped).
"""

import os
import time

import openai
import regex as re
import requests

PREFILL_HOST = os.getenv("PREFILL_HOST", "localhost")
PREFILL_PORT = os.environ["PREFILL_PORT"]
DECODE_HOST = os.getenv("DECODE_HOST", "localhost")
DECODE_PORT = os.environ["DECODE_PORT"]
PROXY_HOST = os.getenv("PROXY_HOST", "localhost")
PROXY_PORT = os.environ["PROXY_PORT"]

# Long prompt (~9000 tokens) to span many blocks so prefix caching kicks in.
_BASE_PROMPT = """\
The following is a comprehensive overview of distributed systems, covering \
their history, design principles, and modern applications.

Distributed systems emerged from the need to connect multiple computers to \
work together on shared tasks. In the 1960s, ARPANET demonstrated that \
geographically dispersed machines could communicate through packet switching. \
This laid the groundwork for decades of research into fault tolerance, \
consistency, and performance.

Leslie Lamport's 1978 paper on logical clocks introduced the concept of \
causal ordering in distributed systems. His later work on the Paxos algorithm \
provided a practical solution to the consensus problem, enabling multiple \
nodes to agree on a single value despite failures. The Byzantine Generals \
Problem, also formulated by Lamport, addressed the challenge of reaching \
agreement when some participants may be malicious.

The CAP theorem, proposed by Eric Brewer in 2000 and formally proved by Seth \
Gilbert and Nancy Lynch in 2002, states that a distributed system cannot \
simultaneously provide Consistency, Availability, and Partition tolerance. \
This fundamental trade-off has guided the design of distributed databases and \
storage systems ever since. Systems like Google's Bigtable chose consistency \
and partition tolerance, while Amazon's Dynamo prioritized availability and \
partition tolerance.

Google's MapReduce framework, published in 2004, popularized the concept of \
processing large datasets across clusters of commodity hardware. The \
programming model was simple: users specified a map function to process \
key-value pairs and a reduce function to merge intermediate values. The \
framework handled distribution, fault tolerance, and load balancing \
automatically. This inspired the open-source Hadoop ecosystem, which became \
the foundation for big data processing throughout the 2010s.

The Google File System (GFS) and its open-source counterpart HDFS provided \
the distributed storage layer beneath MapReduce. These systems replicated \
data across multiple nodes, using a single master for metadata management \
and chunk servers for actual data storage. The master maintained a mapping \
from files to chunks and tracked which chunk servers held each replica.

Apache Kafka, developed at LinkedIn and open-sourced in 2011, introduced a \
distributed commit log that could handle millions of messages per second. \
Its design separated producers from consumers through topic-based \
publish-subscribe semantics. Partitioning allowed horizontal scaling, while \
replication ensured durability. Kafka's exactly-once semantics, achieved \
through idempotent producers and transactional writes, made it suitable for \
financial and mission-critical applications.

Raft, published by Diego Ongaro and John Ousterhout in 2014, provided an \
understandable alternative to Paxos for consensus. Its key insight was \
decomposing consensus into leader election, log replication, and safety. A \
leader would be elected through randomized timeouts, then would replicate \
its log entries to followers. Committed entries were guaranteed to be present \
on a majority of servers. Raft's clarity led to its adoption in systems like \
etcd, CockroachDB, and TiKV.

Container orchestration systems like Kubernetes, released by Google in 2014, \
brought distributed systems concepts to application deployment. Kubernetes \
managed clusters of machines, scheduling containers across nodes while \
maintaining desired state. Its control plane used etcd for consistent state \
storage, an API server for client communication, a scheduler for placement \
decisions, and controllers for reconciliation loops.

Service meshes emerged to handle the networking complexity of microservices \
architectures. Istio, Linkerd, and Envoy provided transparent proxying, load \
balancing, circuit breaking, and observability without requiring application \
code changes. They implemented the sidecar pattern, deploying a proxy \
alongside each service instance to intercept all network traffic.

Modern distributed databases like CockroachDB, TiDB, and YugabyteDB combine \
the SQL interface that developers expect with the horizontal scalability of \
NoSQL systems. They use Raft for consensus, multi-version concurrency control \
for transactions, and range-based sharding for data distribution. These \
systems can span multiple data centers while providing serializable isolation.

Stream processing frameworks evolved from batch-oriented MapReduce to \
real-time systems. Apache Flink provided exactly-once processing with \
event-time semantics, handling out-of-order data through watermarks. Its \
checkpoint mechanism, based on Chandy-Lamport distributed snapshots, allowed \
recovery without data loss. Google's Dataflow model unified batch and \
streaming under a single programming model.

The rise of machine learning at scale introduced new distributed systems \
challenges. Training large neural networks required distributing computation \
across hundreds or thousands of GPUs. Data parallelism split batches across \
workers, while model parallelism partitioned the network itself. Pipeline \
parallelism overlapped computation stages to maximize utilization. \
Ring-allreduce and parameter server architectures provided different \
trade-offs for gradient synchronization.

Inference serving systems like vLLM, TensorRT-LLM, and SGLang optimized the \
deployment of large language models. They introduced techniques like \
continuous batching to maximize GPU utilization, PagedAttention for efficient \
KV cache memory management, and speculative decoding to reduce latency. \
Prefill-decode disaggregation separated the compute-intensive prefill phase \
from the memory-bound decode phase across different GPU pools.

KV cache transfer in disaggregated serving requires careful coordination \
between prefill and decode nodes. The prefill node computes the full KV cache \
for a request's prompt and transfers it to the decode node via high-bandwidth \
interconnects like NVLink, InfiniBand, or RDMA. The decode node then uses \
this transferred cache to generate tokens autoregressively without \
recomputing the prefix.

Prefix caching optimizes this further by recognizing that multiple requests \
often share common prefixes, such as system prompts or few-shot examples. \
When a decode node receives a new request whose prefix matches a previously \
transferred KV cache, it can skip the transfer for those shared blocks and \
only fetch the new, unique portion. This dramatically reduces both network \
bandwidth consumption and time-to-first-token latency.

For hybrid architectures combining attention mechanisms with state-space \
models like Mamba, prefix caching becomes more complex. Attention layers \
maintain a KV cache that can be trivially split into independent blocks, \
making prefix matching straightforward. However, Mamba layers maintain a \
recurrent hidden state that represents the entire sequence history in a \
single fixed-size tensor. This state cannot be meaningfully split into \
prefix-aligned blocks the way attention KV caches can.

The challenge in disaggregated serving of hybrid models is that the cache \
coordination logic must handle these heterogeneous cache types simultaneously. \
A naive approach that requires all cache groups to agree on a single prefix \
hit length will always report zero hits for the Mamba group on a cold decode \
node, dragging the entire prefix cache hit rate to zero even when the \
attention layers have perfect cache hits.

The solution is to evaluate each cache group independently, allowing the \
attention groups to report their actual cache hits while the Mamba group \
reports zero. The transfer logic then only fetches the blocks that each group \
actually needs: for attention, only the new uncached blocks; for Mamba, \
always the full state. This per-group evaluation preserves the prefix caching \
benefits for attention layers while correctly handling the all-or-nothing \
nature of Mamba state.

Consistency models in distributed systems range from strong linearizability \
to weak eventual consistency. Linearizability requires that operations appear \
to occur atomically at some point between their invocation and response. \
Sequential consistency relaxes this by only requiring that operations from \
each process appear in program order. Causal consistency preserves causal \
relationships between operations. Eventual consistency only guarantees that \
all replicas will eventually converge to the same state.

Vector clocks extend Lamport timestamps to capture causality precisely. Each \
process maintains a vector of logical clocks, one per process in the system. \
When a process performs a local event, it increments its own entry. When \
sending a message, it attaches its current vector. Upon receiving a message, \
a process takes the element-wise maximum of its vector and the received \
vector, then increments its own entry. Two events are concurrent if and only \
if neither vector dominates the other.

Conflict-free replicated data types (CRDTs) provide eventual consistency \
without coordination. They achieve this through mathematical properties: \
either operations are commutative and idempotent (operation-based CRDTs), or \
states form a join-semilattice where merging always produces a valid result \
(state-based CRDTs). Examples include grow-only counters, positive-negative \
counters, grow-only sets, observed-remove sets, and last-writer-wins registers.

Distributed hash tables (DHTs) like Chord, Kademlia, and Pastry provide \
decentralized key-value lookup. Chord arranges nodes on a circular identifier \
space, using finger tables for O(log n) routing. Kademlia uses XOR distance \
for routing, enabling parallel lookups and natural load balancing. These \
systems underpin peer-to-peer networks, content distribution, and \
decentralized storage.

Leader election algorithms ensure that exactly one node acts as coordinator \
at any time. The Bully algorithm selects the node with the highest \
identifier. Ring-based algorithms pass election messages around a logical \
ring. In practice, systems often use lease-based leadership where a leader \
must periodically renew its lease, allowing automatic failover when a leader \
becomes unresponsive.

Distributed transactions spanning multiple partitions require coordination \
protocols. Two-phase commit (2PC) provides atomicity but blocks if the \
coordinator fails. Three-phase commit (3PC) adds a prepare-to-commit phase \
to avoid blocking but does not handle network partitions. Saga patterns \
decompose long-running transactions into compensable sub-transactions, \
providing eventual consistency without global locks.

Load balancing in distributed systems takes many forms. Round-robin \
distributes requests evenly but ignores server capacity. Weighted round-robin \
accounts for heterogeneous servers. Least-connections routes to the server \
with fewest active requests. Consistent hashing minimizes redistribution when \
servers join or leave. Power-of-two-choices selects the less loaded of two \
randomly chosen servers, providing near-optimal balance with minimal \
coordination.

Observability in distributed systems requires correlated telemetry across \
service boundaries. Distributed tracing, pioneered by Google's Dapper and \
standardized through OpenTelemetry, propagates trace context through request \
chains. Each service adds spans representing its processing, creating a tree \
structure that reveals latency bottlenecks and error sources. Combined with \
metrics and structured logs, traces provide the visibility needed to operate \
complex distributed systems reliably."""

# Pad to ~23000 chars (~9000 tokens) to fill many blocks.
PROMPT = _BASE_PROMPT
while len(PROMPT) < 23000:
    n = len(PROMPT)
    PROMPT += f" The value at position {n} is {n * 7 % 9973}."


METRICS_OF_INTEREST = [
    "vllm:nixl_bytes_transferred_sum",
    "vllm:nixl_bytes_transferred_count",
    "vllm:nixl_num_descriptors_sum",
    "vllm:nixl_num_descriptors_count",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
]


def get_metric(host: str, port: str, metric_name: str) -> float:
    """Scrape a single Prometheus metric from /metrics."""
    url = f"http://{host}:{port}/metrics"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    total = 0.0
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(metric_name):
            match = re.search(r"[\d.eE+\-]+$", line)
            if match:
                total += float(match.group())
    return total


def get_all_metrics(host: str, port: str) -> dict[str, float]:
    """Scrape all metrics of interest."""
    return {name: get_metric(host, port, name) for name in METRICS_OF_INTEREST}


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"\n  [{label}]")
    for name, val in metrics.items():
        print(f"    {name} = {val}")


def test_mamba_prefix_cache_hit():
    """Repeated prompts through PD should transfer fewer bytes on D-side."""
    proxy_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{PROXY_HOST}:{PROXY_PORT}/v1",
    )
    decode_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{DECODE_HOST}:{DECODE_PORT}/v1",
    )

    models = decode_client.models.list()
    MODEL = models.data[0].id
    print(f"\nModel: {MODEL}")
    print(f"Prompt length: {len(PROMPT)} chars")

    # Baseline
    m_baseline = get_all_metrics(DECODE_HOST, DECODE_PORT)
    print_metrics("D-side baseline", m_baseline)

    # Request 1: cold, primes the D-side cache
    print("\n--- Request 1 (cold) ---")
    resp1 = proxy_client.completions.create(
        model=MODEL, prompt=PROMPT, max_tokens=10, temperature=0, seed=42
    )
    output1 = resp1.choices[0].text
    print(f"  Output: {output1!r}")
    time.sleep(2)

    m_after_req1 = get_all_metrics(DECODE_HOST, DECODE_PORT)
    print_metrics("D-side after req1", m_after_req1)

    transfer_req1 = (
        m_after_req1["vllm:nixl_bytes_transferred_sum"]
        - m_baseline["vllm:nixl_bytes_transferred_sum"]
    )
    descs_req1 = (
        m_after_req1["vllm:nixl_num_descriptors_sum"]
        - m_baseline["vllm:nixl_num_descriptors_sum"]
    )
    print(f"  Transfer: {transfer_req1 / 1e6:.2f} MB, {descs_req1:.0f} descs")

    # Request 2: same prompt, should hit D-side prefix cache
    print("\n--- Request 2 (warm, same prompt) ---")
    resp2 = proxy_client.completions.create(
        model=MODEL, prompt=PROMPT, max_tokens=10, temperature=0, seed=42
    )
    output2 = resp2.choices[0].text
    print(f"  Output: {output2!r}")
    time.sleep(2)

    m_after_req2 = get_all_metrics(DECODE_HOST, DECODE_PORT)
    print_metrics("D-side after req2", m_after_req2)

    transfer_req2 = (
        m_after_req2["vllm:nixl_bytes_transferred_sum"]
        - m_after_req1["vllm:nixl_bytes_transferred_sum"]
    )
    descs_req2 = (
        m_after_req2["vllm:nixl_num_descriptors_sum"]
        - m_after_req1["vllm:nixl_num_descriptors_sum"]
    )
    print(f"  Transfer: {transfer_req2 / 1e6:.2f} MB, {descs_req2:.0f} descs")

    # P-side metrics (informational)
    m_prefill = get_all_metrics(PREFILL_HOST, PREFILL_PORT)
    print_metrics("P-side final", m_prefill)

    # Summary
    print("\n--- Summary ---")
    print(f"  Req 1: {transfer_req1 / 1e6:.2f} MB ({descs_req1:.0f} descs)")
    print(f"  Req 2: {transfer_req2 / 1e6:.2f} MB ({descs_req2:.0f} descs)")
    if transfer_req1 > 0:
        reduction_pct = (1 - transfer_req2 / transfer_req1) * 100
        print(f"  Reduction: {reduction_pct:.1f}%")

    # Assertions
    assert transfer_req1 > 0, (
        f"First request should transfer data, got {transfer_req1} bytes"
    )
    assert transfer_req2 < transfer_req1, (
        f"Second request should transfer fewer bytes due to D-side prefix "
        f"cache hits. Got req1={transfer_req1 / 1e6:.2f} MB, "
        f"req2={transfer_req2 / 1e6:.2f} MB (no reduction)."
    )
    assert output1 == output2, f"Outputs differ: {output1!r} vs {output2!r}"
