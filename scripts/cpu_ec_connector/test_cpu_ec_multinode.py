#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-node CPU EC connector e2e test for OpenShift.

Producer and consumer run as OpenShift Deployments in separate pods.
The driver (this script) runs outside the cluster and communicates with
both pods via `oc port-forward`. Pod-to-pod NIXL and ZMQ traffic flows
directly over the cluster network.

Run from the repo root::

    python scripts/cpu_ec_connector/test_cpu_ec_multinode.py \\
        --namespace my-ns \\
        --image quay.io/my-org/vllm:tag

Prerequisites:
  - `oc` is installed and logged in to the target cluster.
  - A secret named `llm-d-hf-token` (key HF_TOKEN) exists in the namespace.
  - At least two GPU nodes are available (or one node with 2 GPUs).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from k8s_harness import K8sHarness
from shared import (
    ALL_TESTS,
    DEFAULT_MODEL,
    TESTS_REQUIRING_CUSTOM_HARNESS,
    TESTS_REQUIRING_DEFAULT_HARNESS,
    ServerSpec,
    test_baseline,
    test_cache_reuse,
    test_concurrent_ec,
    test_multi_image,
    test_pool_exhaustion,
    test_producer_restart,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = Path(__file__).resolve().parent / "k8s"

DEFAULT_PRODUCER_PORT = 8001
DEFAULT_CONSUMER_PORT = 8002
DEFAULT_PRODUCER_SIDE_PORT = 5601
DEFAULT_CONSUMER_SIDE_PORT = 5602
DEFAULT_IMAGE_PATH = REPO_ROOT / "tests/v1/ec_connector/integration/hato.jpg"


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------


def make_k8s_specs(
    args,
    log_dir: Path,
    *,
    producer_num_ec_blocks: int = 80000,
    consumer_num_ec_blocks: int = 80000,
) -> tuple[ServerSpec, ServerSpec]:
    producer = ServerSpec(
        role="producer",
        gpu=0,  # unused in K8s mode; GPU assignment is in pod spec
        http_port=args.producer_port,
        side_channel_port=DEFAULT_PRODUCER_SIDE_PORT,
        engine_id="ec-producer-0",
        gpu_memory_utilization=0.05,
        log_path=log_dir / "producer.log",
        num_ec_blocks=producer_num_ec_blocks,
    )
    consumer = ServerSpec(
        role="consumer",
        gpu=0,  # unused in K8s mode
        http_port=args.consumer_port,
        side_channel_port=DEFAULT_CONSUMER_SIDE_PORT,
        engine_id="ec-consumer-0",
        gpu_memory_utilization=0.5,
        log_path=log_dir / "consumer.log",
        num_ec_blocks=consumer_num_ec_blocks,
    )
    return producer, consumer


def _make_harness(args):
    """Return a callable (producer, consumer, model, **kw) -> K8sHarness."""

    def factory(producer, consumer, model, *, keep_on_exit: bool = False):
        return K8sHarness(
            producer,
            consumer,
            model,
            namespace=args.namespace,
            image=args.image,
            k8s_dir=K8S_DIR,
            different_nodes=args.different_nodes,
            log_delay=args.log_delay,
            keep_on_exit=keep_on_exit,
        )

    return factory


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--namespace", required=True, help="OpenShift namespace to deploy into"
    )
    parser.add_argument(
        "--image", required=True, help="vLLM container image (registry/image:tag)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--image-path",
        dest="image_path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Local path to the test image (hato.jpg or similar)",
    )
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument(
        "--producer-port",
        type=int,
        default=DEFAULT_PRODUCER_PORT,
        help="Local port forwarded to producer's HTTP server",
    )
    parser.add_argument(
        "--consumer-port",
        type=int,
        default=DEFAULT_CONSUMER_PORT,
        help="Local port forwarded to consumer's HTTP server",
    )
    parser.add_argument(
        "--different-nodes",
        action="store_true",
        help="Add podAntiAffinity to force producer+consumer on "
        "different Kubernetes nodes",
    )
    parser.add_argument(
        "--log-delay",
        type=float,
        default=0.5,
        help="Seconds to wait after each generate() call before "
        "reading the log window (oc logs has streaming latency)",
    )
    parser.add_argument(
        "--keep-servers",
        action="store_true",
        help="Leave the shared-harness deployments running on success",
    )
    parser.add_argument(
        "--test",
        action="append",
        choices=("all", *ALL_TESTS),
        default=None,
        help="Test(s) to run (repeatable). Default: all",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"image not found: {args.image_path}", file=sys.stderr)
        return 2

    selected = set(args.test or ["all"])
    if "all" in selected:
        selected = set(ALL_TESTS)
    normal_tests = selected & set(TESTS_REQUIRING_DEFAULT_HARNESS)
    custom_tests = selected & set(TESTS_REQUIRING_CUSTOM_HARNESS)

    log_dir = REPO_ROOT / "logs" / "cpu_ec_multinode" / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] log dir: {log_dir}")
    print(f"[setup] namespace={args.namespace}  image={args.image}")
    print(f"[setup] tests: {sorted(selected)}")

    make_harness = _make_harness(args)
    failures: list[tuple[str, BaseException]] = []

    # Tests 1-4 share one Harness (one cold-start pair of deployments).
    if normal_tests:
        producer, consumer = make_k8s_specs(args, log_dir)
        with make_harness(
            producer, consumer, args.model, keep_on_exit=args.keep_servers
        ) as h:
            for test_name in TESTS_REQUIRING_DEFAULT_HARNESS:
                if test_name not in normal_tests:
                    continue
                try:
                    if test_name == "baseline":
                        test_baseline(h, args.image_path, args.prompt)
                    elif test_name == "cache-reuse":
                        test_cache_reuse(h, args.prompt)
                    elif test_name == "multi-image":
                        test_multi_image(h, args.prompt)
                    elif test_name == "concurrent":
                        test_concurrent_ec(h, args.prompt)
                except AssertionError as e:
                    failures.append((test_name, e))
                    print(f"  ✗ {test_name} FAILED: {e}", file=sys.stderr)

    # Tests 5-6 each create their own Harness (different configs or restart logic).
    for test_name in TESTS_REQUIRING_CUSTOM_HARNESS:
        if test_name not in custom_tests:
            continue
        try:
            if test_name == "pool-exhaustion":
                test_pool_exhaustion(
                    log_dir,
                    args.model,
                    args.prompt,
                    make_specs_fn=lambda ld, **kw: make_k8s_specs(args, ld, **kw),
                    make_harness=make_harness,
                )
            elif test_name == "producer-restart":
                test_producer_restart(
                    log_dir,
                    args.model,
                    args.prompt,
                    make_specs_fn=lambda ld, **kw: make_k8s_specs(args, ld, **kw),
                    make_harness=make_harness,
                )
        except AssertionError as e:
            failures.append((test_name, e))
            print(f"  ✗ {test_name} FAILED: {e}", file=sys.stderr)

    print(f"\n[teardown] logs preserved at {log_dir}")
    if failures:
        print(f"\n[done] {len(failures)} of {len(selected)} tests FAILED:")
        for name, e in failures:
            print(f"  - {name}: {e}")
        return 1
    print(f"\n[done] all {len(selected)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
