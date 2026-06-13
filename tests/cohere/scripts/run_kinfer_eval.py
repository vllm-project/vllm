# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deploy a model via kinfer and run a bee eval suite against it.

Invoked by .github/workflows/kinfer-pipeline.yaml on a CPU runner.
GPU work happens on a Cohere K8s cluster via the kinfer SDK.

Flow:
1. Load the per-model deploy spec from tests/cohere/configs/kinfer_deploy/.
   The YAML is fully authoritative (hardware, weights, cluster, queue, etc.).
   Only cluster.image is injected at runtime (nightly-built SHA).
2. Deploy via client.deployments.deploy_full_pipeline() and wait for health.
3. For each eval YAML mapped to the model, inject a unique W&B run name, then
   call client.evals.launch() and block until the job finishes.
   bee logs artifact tables directly to W&B from inside the cluster.
4. After each suite: call log_kinfer_eval_to_wandb.log_eval() to download the
   artifact tables, extract scalar scores, and append one step to the persistent
   per-(model, device) W&B trend run — giving a clean time-series chart.
5. Always tear down the deployment.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger("run_kinfer_eval")

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR.parent / "configs"
DEPLOY_DIR = CONFIGS_DIR / "kinfer_deploy"
EVAL_DIR = CONFIGS_DIR / "kinfer_eval"

# Prefix for stable W&B persistent run IDs (one per model+device).
# W&B tombstones deleted run IDs permanently — bump this prefix if a clean
# slate is ever needed (e.g. "vllm-ci-5").
_WANDB_RUN_ID_PREFIX = "vllm-ci-4"


def load_deploy_spec(model: str) -> tuple[dict, dict]:
    """Return (ci_config, kinfer_spec) for the given model key.

    Deploy spec is looked up by convention: ``{model}.yaml`` in the deploy
    config directory. The YAML may contain a top-level ``ci`` section with
    fields used by this script (e.g. ``cluster_context``) that are not part
    of the kinfer VLLMSpec. That section is popped before the spec is passed
    to kinfer.
    """
    spec_path = DEPLOY_DIR / f"{model}.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"No kinfer deploy spec found for model '{model}': "
            f"{spec_path}. Create it before routing this model to kinfer."
        )
    with open(spec_path) as f:
        raw = yaml.safe_load(f)
    ci_config: dict = raw.pop("ci", {})
    return ci_config, raw


def _export_trainjob_to_gha(trainjob: str, cluster_context: str) -> None:
    """Write the TrainJob name and cluster context to GITHUB_ENV so the GHA
    cleanup step can target the right cluster and delete the job if the runner
    is cancelled or hits timeout-minutes before the Python finally block runs."""
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as fh:
            fh.write(f"KINFER_TRAINJOB={trainjob}\n")
            fh.write(f"KINFER_CLUSTER_CONTEXT={cluster_context}\n")


def set_image(spec: dict, image: str) -> None:
    """Inject the nightly-built image SHA — the only runtime override.

    Validates that total_gpus and tensor-parallel-size are consistent with the
    replicas declared in the YAML, but does not overwrite any of those values —
    the deploy YAML is fully authoritative for all fields except cluster.image.
    """
    tp = spec["model"]["vllm_args"]["tensor-parallel-size"]
    total_gpus = spec["cluster"]["total_gpus"]
    replicas = spec.get("standard", {}).get("replicas")
    if total_gpus % tp != 0:
        raise ValueError(
            f"tensor-parallel-size={tp} must divide "
            f"total_gpus={total_gpus} from the deploy spec."
        )
    if replicas is not None and replicas * tp != total_gpus:
        raise ValueError(
            f"standard.replicas={replicas} × tensor-parallel-size={tp} "
            f"does not equal total_gpus={total_gpus} in the deploy spec."
        )
    spec["cluster"]["image"] = image


def resolve_eval_suites(model: str) -> list[Path]:
    import json

    suite_map_path = EVAL_DIR / "eval_suite_map.json"
    with open(suite_map_path) as f:
        suite_map = json.load(f)
    suites = suite_map.get(model)
    if not suites:
        raise KeyError(
            f"No eval suites mapped for model '{model}' in {suite_map_path}."
        )
    return [EVAL_DIR / s for s in suites]


class _EarlyExportReporter:
    """Wraps ConsoleReporter to export the TrainJob name to GITHUB_ENV as soon
    as kinfer creates it — before wait_for_pods_ready — so the GHA cleanup
    step has the name even if the runner is killed during pod startup."""

    def __init__(self, delegate: object, *, cluster_context: str, keep: bool) -> None:
        self._delegate = delegate
        self._cluster_context = cluster_context
        self._keep = keep

    def step(self, message: str) -> None:
        self._delegate.step(message)  # type: ignore[attr-defined]
        # kinfer calls reporter.step(f"Created TrainJob: {name}") before waiting.
        if not self._keep and message.startswith("Created TrainJob: "):
            trainjob = message.removeprefix("Created TrainJob: ").strip()
            _export_trainjob_to_gha(trainjob, self._cluster_context)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def _log_to_wandb(
    bee_run_name: str,
    model: str,
    deploy_spec: dict,
    ci_config: dict,
) -> None:
    """Best-effort: extract scalar scores from the finished Bee run's artifact
    tables and append one step to the persistent per-(model, device) W&B run.

    Bee only logs artifact tables (no scalars). This function downloads those
    tables, pulls the primary score from each, and logs a single scalar step to
    the persistent trend run — giving a clean nightly time-series chart.
    """
    try:
        import sys

        sys.path.insert(0, str(SCRIPT_DIR))
        from log_kinfer_eval_to_wandb import log_eval  # type: ignore[import]
    except ImportError as exc:
        logger.warning(
            "log_kinfer_eval_to_wandb not available — skipping W&B logging: %s", exc
        )
        return

    device = deploy_spec.get("cluster", {}).get("hardware", "").lower() or None
    cluster = ci_config.get("cluster_context") or None
    tp = deploy_spec.get("model", {}).get("vllm_args", {}).get("tensor-parallel-size")
    try:
        log_eval(
            bee_run_name=bee_run_name,
            model=model,
            device=device,
            cluster=cluster,
            tp=tp,
        )
    except Exception as exc:
        logger.warning("W&B trend logging failed (non-fatal): %s", exc)


def run_eval(args: argparse.Namespace) -> None:
    from kinfer.sdk import (
        ConsoleReporter,
        Kinfer,
        KinferSpec,
        KubeClient,
        switch_kube_context,
    )

    ci_config, deploy_spec = load_deploy_spec(args.model)
    set_image(deploy_spec, args.docker_image)

    cluster_context = ci_config.get("cluster_context")
    if not cluster_context:
        raise ValueError(
            f"Deploy spec for '{args.model}' is missing ci.cluster_context."
        )
    queue = ci_config.get("queue")
    priority = ci_config.get("priority")
    if not queue:
        raise ValueError(f"Deploy spec for '{args.model}' is missing ci.queue.")
    if not priority:
        raise ValueError(f"Deploy spec for '{args.model}' is missing ci.priority.")

    reporter = _EarlyExportReporter(
        ConsoleReporter(),
        cluster_context=cluster_context,
        keep=args.keep_deployment,
    )
    switch_kube_context(cluster_context)
    client = Kinfer(k8s=KubeClient())

    trainjob: str | None = None
    try:
        if args.existing_trainjob:
            trainjob = args.existing_trainjob
            logger.info("Using existing TrainJob %s", trainjob)
        else:
            spec = KinferSpec.from_object(deploy_spec, queue=queue, priority=priority)
            deploy = client.deployments.deploy_full_pipeline(
                spec,
                wait_for_health=True,
                follow_logs=False,
                reporter=reporter,
            )
            trainjob = deploy.trainjob_name
            logger.info("Deployed TrainJob %s (%s)", trainjob, deploy.serving_url)
            # Guarantee GITHUB_ENV export even if the reporter's log-line heuristic
            # never fired (e.g. kinfer changes the "Created TrainJob:" message).
            # Skip when --keep-deployment is set so the GHA safety-net cleanup
            # step does not delete a TrainJob the caller explicitly wants to keep.
            if not args.keep_deployment:
                _export_trainjob_to_gha(trainjob, cluster_context)

        run_timestamp = args.run_timestamp or datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M"
        )
        for suite_path in resolve_eval_suites(args.model):
            logger.info("Launching eval suite %s", suite_path.name)
            with open(suite_path) as f:
                eval_config: dict = yaml.safe_load(f)

            bee_run_name = f"{args.model}-{suite_path.stem}-{run_timestamp}"
            eval_config["log_wandb_run_name"] = bee_run_name

            launch = client.evals.launch(trainjob, eval_config, reporter=reporter)
            logger.info(
                "Eval job %s started (bee: %s, wandb run: %s, reports: %s)",
                launch.job_id,
                launch.bee,
                bee_run_name,
                launch.reports_path,
            )

            logs = client.evals.wait_for_logs(launch, timeout_s=args.eval_timeout)
            print(logs)

            _log_to_wandb(
                bee_run_name=bee_run_name,
                model=args.model,
                deploy_spec=deploy_spec,
                ci_config=ci_config,
            )
    finally:
        # trainjob is None if deploy_full_pipeline raised before returning a name;
        # kinfer is expected to clean up its own partial TrainJob in that case.
        # Wrap delete in try/except so a teardown failure never masks the original
        # eval exception — the original propagates, delete failure is logged only.
        if trainjob and not args.keep_deployment and not args.existing_trainjob:
            logger.info("Deleting TrainJob %s", trainjob)
            try:
                client.deployments.delete(trainjob)
            except Exception as exc:
                logger.warning("TrainJob deletion failed (non-fatal): %s", exc)


def cleanup_trainjob() -> None:
    """Safety-net teardown for the GHA cleanup step.

    Reads KINFER_TRAINJOB and KINFER_CLUSTER_CONTEXT from the environment
    (written by run_eval after a successful deploy) and deletes the TrainJob
    on the correct cluster.  Called when the runner is cancelled or hits
    timeout-minutes before the Python finally block runs.
    """
    import sys

    from kinfer.sdk import (
        Kinfer,
        KubeClient,
        TrainJobNotFoundError,
        switch_kube_context,
    )

    trainjob = os.environ.get("KINFER_TRAINJOB", "")
    if not trainjob:
        logger.info("No TrainJob to clean up.")
        return

    cluster_context = os.environ.get("KINFER_CLUSTER_CONTEXT", "")
    if not cluster_context:
        logger.error(
            "KINFER_CLUSTER_CONTEXT is not set — cannot safely target the "
            "right cluster to delete TrainJob %s. Aborting cleanup to avoid "
            "acting on the wrong cluster.",
            trainjob,
        )
        sys.exit(1)

    try:
        switch_kube_context(cluster_context)
        client = Kinfer(k8s=KubeClient())
        client.deployments.delete(trainjob)
        logger.info("Deleted TrainJob %s on %s", trainjob, cluster_context)
    except TrainJobNotFoundError:
        logger.info("TrainJob %s already deleted — nothing to clean up.", trainjob)
    except Exception as e:
        logger.error("Failed to delete TrainJob %s: %s", trainjob, e)
        sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    # --- eval subcommand ---
    eval_parser = subparsers.add_parser("eval", help="Deploy and run evals")
    eval_parser.add_argument(
        "--model",
        required=True,
        help="Model key (must match deploy_map.json and eval_suite_map.json)",
    )
    eval_parser.add_argument(
        "--docker-image",
        required=True,
        help="Full vLLM image ref (e.g. .../vllm-nvidia:<sha>)",
    )
    eval_parser.add_argument(
        "--eval-timeout",
        type=float,
        default=14400.0,
        help="Seconds to wait for a single eval suite (default: 14400 = 4h)",
    )
    eval_parser.add_argument(
        "--keep-deployment",
        action="store_true",
        help="Do not delete the TrainJob after evaluation (for debugging)",
    )
    eval_parser.add_argument(
        "--existing-trainjob",
        default="",
        help="Skip deploy and run evals against this already-running TrainJob",
    )
    eval_parser.add_argument(
        "--run-timestamp",
        default="",
        help=(
            "Shared UTC timestamp (YYYYMMDD-HHMM) injected into bee run names "
            "so parallel GPU jobs land at the same x position in W&B charts. "
            "Defaults to the current UTC time when not provided."
        ),
    )

    # --- cleanup subcommand ---
    subparsers.add_parser(
        "cleanup",
        help="Delete the TrainJob named in $KINFER_TRAINJOB (GHA safety-net teardown)",
    )

    args = parser.parse_args()
    if args.command == "cleanup":
        cleanup_trainjob()
    elif args.command == "eval":
        run_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
