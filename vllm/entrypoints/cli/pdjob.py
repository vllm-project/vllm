# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid

import ray

from vllm.entrypoints.cli.pd.config import read_config
from vllm.entrypoints.cli.pd.multiple_prefills import MultiplePrefillsPDJob
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.cli.pd.base import RankIDGenerator
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger(__name__)


class PDDisaggregatedJobCommand(CLISubcommand):
    """The `pdjob` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "pdjob"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:

        config = read_config(args.config)

        # connect to an existed ray cluster
        runtime_env = {"env_vars": config.envs, "working_dir": config.working_dir}
        ray.init(log_to_driver=not args.disable_log_to_driver,
                 runtime_env=runtime_env)

        job_id = str(uuid.uuid4())
        
        # reset the global rank ID generator
        RankIDGenerator.reset()
        # Supported KV Connectors (registered in vllm/distributed/kv_transfer/kv_connector/factory.py):
        # - NixlConnector: NIXL-based KV cache transfer
        # - P2pNcclConnector: Point-to-point NCCL transfer
        # - MultiConnector: Multi-backend connector
        # - SharedStorageConnector: Shared storage based connector
        # - LMCacheConnectorV1: LMCache integration
        # - LMCacheMPConnector: LMCache multi-process connector
        # - OffloadingConnector: KV cache offloading connector
        # - DecodeBenchConnector: Decode benchmarking connector
        supported_connectors = (
            "nixlconnector",
            "p2pncclconnector",
            "multiconnector",
            "sharedstorageconnector",
            "lmcacheconnectorv1",
            "lmcachempconnector",
            "offloadingconnector",
            "decodebenchconnector",
        )
        
        if config.kv_connector in supported_connectors:
            job = MultiplePrefillsPDJob(job_id, config)
        else:
            raise ValueError(f"Unrecognized kv_connector: {config.kv_connector}. "
                             f"Supported connectors: {', '.join(supported_connectors)}")

        # job.start() contains internal while loop and handles shutdown
        job.start()

    def validate(self, args: argparse.Namespace) -> None:
        return

    def subparser_init(
        self,
        subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        pdjob_parser = subparsers.add_parser(
            "pdjob",
            help="Start a prefill & decode disaggregated job on ray",
            usage="vllm pdjob --config={PATH_TO_CONFIG}")
        pdjob_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=True,
            help="Read CLI options from a config file."
                 "See the example under examples/pdjob/config.yaml"
        )

        pdjob_parser.add_argument(
            "--disable-log-to-driver",
            action='store_true',
            default=False,
            required=False,
            help="Whether enable ray log-to-driver"
        )

        return pdjob_parser


def cmd_init() -> list[CLISubcommand]:
    return [PDDisaggregatedJobCommand()]
