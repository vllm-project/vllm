# SPDX-License-Identifier: Apache-2.0
"""``lmcache coordinator`` — launch the LMCache mp coordinator (HTTP).

The coordinator tracks mp server instances via a registry and evicts those
whose heartbeats lapse. Configuration falls back to ``LMCACHE_MP_COORDINATOR_*``
environment variables; CLI flags override them.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class CoordinatorCommand(BaseCommand):
    """CLI command that launches the LMCache mp coordinator (HTTP)."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"coordinator"``.
        """
        return "coordinator"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Launch the LMCache mp coordinator (HTTP)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add coordinator-specific arguments to the parser.

        Each flag defaults to ``None`` so that unset flags fall back to the
        ``LMCACHE_MP_COORDINATOR_*`` environment variables (and then the
        config defaults) in :meth:`execute`.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Host the coordinator's HTTP server binds to (default: 0.0.0.0).",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Port the coordinator's HTTP server binds to (default: 9300).",
        )
        parser.add_argument(
            "--instance-timeout",
            type=float,
            default=None,
            help=(
                "Seconds without a heartbeat after which an instance is evicted "
                "(default: 30)."
            ),
        )
        parser.add_argument(
            "--health-check-interval",
            type=float,
            default=None,
            help=(
                "Seconds between health-check sweeps; 0 disables the loop "
                "(default: 10)."
            ),
        )
        parser.add_argument(
            "--eviction-check-interval",
            type=float,
            default=None,
            help=(
                "Seconds between L2 eviction sweeps; 0 disables the loop (default: 5)."
            ),
        )
        parser.add_argument(
            "--eviction-ratio",
            type=float,
            default=None,
            help=(
                "Fraction of tracked keys (by count) to evict per cycle, "
                "0.0 to 1.0 (default: 0.2)."
            ),
        )
        parser.add_argument(
            "--trigger-watermark",
            type=float,
            default=None,
            help=(
                "Eviction fires when usage reaches this fraction of the "
                "quota, 0.0 (exclusive) to 1.0 (default: 1.0)."
            ),
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=None,
            help=(
                "Tokens per KV chunk: the CacheBlend match unit and the unit used "
                "to resolve pin token_ids to keys. Must equal the MP servers' "
                "--chunk-size (default: 256)."
            ),
        )
        parser.add_argument(
            "--hash-algorithm",
            type=str,
            default=None,
            help=(
                "Token hash algorithm for pin key resolution; must equal the MP "
                "servers' --hash-algorithm. 'blake3' (default) is self-contained; "
                "other algorithms require vLLM importable in the coordinator."
            ),
        )
        parser.add_argument(
            "--blend-probe-stride",
            type=int,
            default=None,
            help=(
                "Positions between CacheBlend match probes; 1 probes every "
                "offset for full recall (default: 1)."
            ),
        )
        parser.add_argument(
            "--timeout-keep-alive",
            type=int,
            default=None,
            help=(
                "Seconds the HTTP server keeps idle connections open "
                "before closing them (default: 10)."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Build the coordinator config and serve the app with uvicorn.

        Resolves config from the environment, then overrides any field whose
        corresponding CLI flag was supplied.

        Args:
            args: Parsed CLI arguments.

        Raises:
            SystemExit: When coordinator dependencies are not installed.
        """
        # Standard
        import dataclasses
        import sys

        try:
            # Third Party
            import uvicorn

            # First Party
            from lmcache.v1.mp_coordinator.app import create_app
            from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
        except ImportError:
            print(
                "The 'lmcache coordinator' command requires the full lmcache "
                "installation.\nInstall with: pip install lmcache",
                file=sys.stderr,
            )
            sys.exit(1)

        config = MPCoordinatorConfig.from_env()

        overrides = {
            field: value
            for field, value in (
                ("host", args.host),
                ("port", args.port),
                ("instance_timeout", args.instance_timeout),
                ("health_check_interval", args.health_check_interval),
                ("eviction_check_interval", args.eviction_check_interval),
                ("eviction_ratio", args.eviction_ratio),
                ("trigger_watermark", args.trigger_watermark),
                ("chunk_size", args.chunk_size),
                ("hash_algorithm", args.hash_algorithm),
                ("blend_probe_stride", args.blend_probe_stride),
                ("timeout_keep_alive", args.timeout_keep_alive),
            )
            if value is not None
        }
        if overrides:
            config = dataclasses.replace(config, **overrides)

        app = create_app(config)
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
            timeout_keep_alive=config.timeout_keep_alive,
        )
