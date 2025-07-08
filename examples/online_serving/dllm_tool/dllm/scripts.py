from typing import List, Optional
import click
import ray
import logging
import shlex

from dllm.controller import Controller
from dllm.endpoint import deploy_endpoint_to_cluster
from dllm.logging import setup_logging
from dllm.constants import ENDPOINT_APPLICATION_NAME, DLLM_NAMESPACE, CONTROLLER_ACTOR_NAME
from dllm.entities import SchedulerPolicy
from dllm.config import ControllerConfig, InferenceInstanceConfig

setup_logging()
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """DLLM Cluster Management"""
    pass


@cli.command(name="deploy", context_settings={"show_default": True})
@click.option("--head-ip", type=str, help='IP of Ray head node (e.g. "10.2.3.4")', default="auto")
@click.option("--prefill-instances-num", type=int, help="the num of Prefill instances", default=0)
@click.option(
    "--prefill-startup-params",
    type=str,
    help="the Prefill instance start up command",
    default="vllm serve /workspace/models/qwen2.5_7B",
    callback=lambda ctx, param, value: shlex.split(value),
)
@click.option(
    "--prefill-startup-env",
    type=str,
    help="the Prefill instance start up env",
    default=None,
)
@click.option("--prefill-data-parallel-size", "-pdp", type=int, help="the dp of Prefill instances", default=1)
@click.option("--prefill-tensor-parallel-size", "-ptp", type=int, help="the tp of Prefill instances", default=1)
@click.option(
    "--prefill-expert-parallel-size",
    "-pep",
    type=int,
    help="the ep of Prefill instances, should be equal to dp*tp, 0 means disable expert parallelism",
    default=0,
)
@click.option("--decode-instances-num", type=int, help="the num of Decode instances", default=0)
@click.option(
    "--decode-startup-params",
    type=str,
    help="the Decode instance start up command",
    default="vllm serve /workspace/models/qwen2.5_7B",
    callback=lambda ctx, param, value: shlex.split(value),
)
@click.option(
    "--decode-startup-env",
    type=str,
    help="the decode instance start up env",
    default=None,
)
@click.option("--decode-data-parallel-size", "-ddp", type=int, help="the dp of Decode instances", default=1)
@click.option("--decode-tensor-parallel-size", "-dtp", type=int, help="the tp of Decode instances", default=1)
@click.option(
    "--decode-expert-parallel-size",
    "-dep",
    type=int,
    help="the ep of Decode instances, should be equal to dp*tp, 0 means disable expert parallelism",
    default=0,
)
@click.option(
    "--scheduler-policy",
    type=click.Choice([e.name for e in SchedulerPolicy], case_sensitive=False),
    help="the scheduling policy, default to RoundRobin",
    default=SchedulerPolicy.ROUND_ROBIN.name,
    callback=lambda ctx, param, value: SchedulerPolicy[value.upper()],
)
@click.option("--proxy-host", type=str, help="the dllm service listening host", default="0.0.0.0")
@click.option("--proxy-port", type=int, help="the dllm service listening port", default=8000)
def deploy(
    head_ip: str,
    prefill_instances_num: int,
    prefill_startup_params: List[str],
    prefill_startup_env: Optional[str],
    prefill_data_parallel_size: int,
    prefill_tensor_parallel_size: int,
    prefill_expert_parallel_size: int,
    decode_instances_num: int,
    decode_startup_params: List[str],
    decode_startup_env: Optional[str],
    decode_data_parallel_size: int,
    decode_tensor_parallel_size: int,
    decode_expert_parallel_size: int,
    scheduler_policy: SchedulerPolicy,
    proxy_host: str,
    proxy_port: int,
):
    _inner_deploy(
        head_ip,
        prefill_instances_num,
        prefill_startup_params,
        prefill_startup_env,
        prefill_data_parallel_size,
        prefill_tensor_parallel_size,
        prefill_expert_parallel_size,
        decode_instances_num,
        decode_startup_params,
        decode_startup_env,
        decode_data_parallel_size,
        decode_tensor_parallel_size,
        decode_expert_parallel_size,
        scheduler_policy,
        proxy_host,
        proxy_port,
    )


def _inner_deploy(
    head_ip: str,
    prefill_instances_num: int,
    prefill_startup_params: List[str],
    prefill_startup_env: Optional[str],
    prefill_data_parallel_size: int,
    prefill_tensor_parallel_size: int,
    prefill_expert_parallel_size: int,
    decode_instances_num: int,
    decode_startup_params: List[str],
    decode_startup_env: Optional[str],
    decode_data_parallel_size: int,
    decode_tensor_parallel_size: int,
    decode_expert_parallel_size: int,
    scheduler_policy: SchedulerPolicy,
    proxy_host: str,
    proxy_port: int,
):
    config = ControllerConfig(
        scheduler_policy=scheduler_policy,
        num_prefill_instances=prefill_instances_num,
        prefill_instance_config=InferenceInstanceConfig(
            startup_params=prefill_startup_params,
            startup_env=prefill_startup_env,
            dp=prefill_data_parallel_size,
            tp=prefill_tensor_parallel_size,
            ep=prefill_expert_parallel_size,
        ),
        num_decode_instances=decode_instances_num,
        decode_instance_config=InferenceInstanceConfig(
            startup_params=decode_startup_params,
            startup_env=decode_startup_env,
            dp=decode_data_parallel_size,
            tp=decode_tensor_parallel_size,
            ep=decode_expert_parallel_size,
        ),
    )

    """Deploy to Ray cluster"""
    try:
        logger.info("Connecting to existing Ray cluster at: %s", head_ip)
        ray.init(address=head_ip, namespace=DLLM_NAMESPACE,
                 runtime_env={"worker_process_setup_hook": setup_logging})
    except Exception as e:
        logger.exception("Failed to connect ray cluster: %s", str(e))
        return

    logger.info("Ray cluster resources: %s", ray.cluster_resources())

    should_start_controller = False
    try:
        controller = ray.get_actor(CONTROLLER_ACTOR_NAME)
        logger.exception(
            "There is already an dllm controller running in the cluster, please clean dllm before " "deploy again"
        )
    except ValueError:
        should_start_controller = True

    if not should_start_controller:
        return

    logger.info("No existing Controller found, creating new instance")
    controller = ray.remote(Controller).options(
        name=CONTROLLER_ACTOR_NAME,
        lifetime="detached",
    ).remote(config)
    ray.get(controller.initialize.remote())
    logger.info("Controller actor created.")

    try:
        ray.serve.shutdown()
        deploy_endpoint_to_cluster(proxy_host, proxy_port)
        logger.info("Deployment completed successfully")
    except Exception as e:
        logger.exception("Deployment failed: %s", str(e))


@cli.command("clean", context_settings={"show_default": True})
@click.option("--head-ip", type=str, help='IP of Ray head node (e.g. "10.2.3.4")', default="auto")
@click.option("--shutdown-ray-serve/--no-shutdown-ray-serve", type=bool, is_flag=True,
              help="whether or not to shutdown Ray serve proxy", default=True)
def clean(head_ip, shutdown_ray_serve):
    """Clean up deployment from Ray cluster"""
    _inner_clean(head_ip, shutdown_ray_serve)


def _inner_clean(head_ip, shutdown_ray_serve):
    try:
        logger.info("Connecting to existing Ray cluster at: %s", head_ip)
        ray.init(address=head_ip, namespace=DLLM_NAMESPACE, log_to_driver=False,
                 runtime_env={"worker_process_setup_hook": setup_logging})
    except Exception as e:
        logger.exception("Failed to connect ray cluster: %s", str(e))
        return

    if shutdown_ray_serve:
        ray.serve.shutdown()
    else:
        try:
            ray.serve.delete(ENDPOINT_APPLICATION_NAME)
        except Exception as e:
            logger.warning("Cleanup endpoint failed: %s", str(e))

    controller = None
    try:
        controller = ray.get_actor(CONTROLLER_ACTOR_NAME)
        logger.info("Found existing Controller actor, attempting to kill it")
        ray.get(controller.terminate.remote())
    except ValueError:
        logger.info("No existing Controller actor found, nothing to clean")
    except Exception as e:
        logger.info(f"Failed to clean up controller {e}")
    finally:
        if controller:
            ray.kill(controller)

if __name__ == "__main__":
    cli()