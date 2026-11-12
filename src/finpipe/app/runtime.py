from contextlib import AbstractContextManager, nullcontext
from logging import Logger

import docker
import mlflow
from docker.errors import APIError

from finpipe.app.settings import ManualPipelineConfig, PipelineConfig


def check_docker_available(logger: Logger) -> None:
    client = docker.from_env()
    try:
        client.ping()
        logger.info("Docker server available")
    except APIError as e:
        logger.error(f"Docker server connection required to run pipeline: {e}")
    finally:
        client.close()


def init_context(
    cfg: PipelineConfig | ManualPipelineConfig,
    logger: Logger,
) -> AbstractContextManager:

    if "mlflow" not in cfg:
        return nullcontext()

    tracking = cfg.mlflow
    logger.info("Starting MLFLow")
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.langchain.autolog()
    mlflow.set_experiment(tracking.experiment_name)
    return mlflow.start_run()
