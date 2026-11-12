"""Run the same task-building stages for a search batch or a single paper."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from omegaconf import DictConfig
from sqlalchemy import Engine, create_engine

from finpipe.app.runtime import check_docker_available, init_context
from finpipe.app.settings import Scope
from finpipe.discovery.service import PaperScrapingPipeline
from finpipe.documents.service import PaperRetrieval
from finpipe.evaluation.service import Judge
from finpipe.export.service import TaskExporter
from finpipe.extraction.service import TaskExtractor
from finpipe.generation.service import TaskGenerator
from finpipe.review.service import TaskCritic
from finpipe.storage.models import Base, PaperStatus
from finpipe.storage.status import set_paper_status

logger = logging.getLogger(__name__)
RunMode = Literal["batch", "paper"]


def _scopes(cfg: DictConfig, mode: RunMode) -> list[Scope]:
    if mode == "paper":
        assert len(cfg.scopes) == 1, "A single scope should be selected"
        scope = Scope(**next(iter(cfg.scopes.values())))
        assert cfg.arxiv_id is not None, "Field `arxiv_id` should be set"
        return [scope]
    return [Scope(**value) for value in cfg.scopes.values() if value.enabled]


def _execute(
    engine: Engine,
    cfg: DictConfig,
    output: Path,
    scopes: list[Scope],
    mode: RunMode,
) -> None:
    discovery = PaperScrapingPipeline(engine, cfg.scraping)
    paper = None
    if mode == "paper":
        paper = discovery.scrape_paper(output, scopes[0], cfg.arxiv_id)
    else:
        discovery.run(
            output,
            scopes,
            datetime.strptime(cfg.scraping.since, "%Y-%m-%d"),
            datetime.strptime(cfg.scraping.until, "%Y-%m-%d"),
            cfg.scraping.max_papers_per_scope,
        )

    PaperRetrieval(engine, cfg.retrieval).download_and_chunk_papers(output)
    if mode == "batch" or cfg.enable_judge:
        Judge(engine, cfg.judge).run(output, scopes)
    else:
        logger.info("Skipping paper judgement stage and auto accepting paper")
        set_paper_status(engine, paper, PaperStatus.ACCEPTED)

    TaskExtractor(engine, cfg.task_extractor).run(output, scopes)
    critic = TaskCritic(engine, cfg.task_critic)
    if cfg.enable_task_critic:
        critic.run(output, scopes)
    else:
        critic.run(output, scopes, auto_approve=True)

    TaskGenerator(engine, cfg.task_generator).run(output, scopes)
    TaskExporter(engine, cfg.task_exporter).run(output, scopes)


def run_workflow(cfg: DictConfig, output_dir: Path, mode: RunMode) -> None:
    """Initialize a run, execute its stages, and release its database pool."""
    if mode not in ("batch", "paper"):
        raise ValueError(f"Unknown run mode: {mode}")

    # Preserve each command's environment/bootstrap ordering.
    if mode == "paper":
        check_docker_available(logger)
        load_dotenv()
        context = init_context(cfg, logger)
        scopes = _scopes(cfg, mode)
        engine = create_engine("sqlite:///:memory:")
    else:
        load_dotenv()
        check_docker_available(logger)
        engine = create_engine(cfg.db_engine)
        context = None

    try:
        Base.metadata.create_all(engine)
        if mode == "batch":
            scopes = _scopes(cfg, mode)
            context = init_context(cfg, logger)
        with context:
            logger.info("Starting %s pipeline", mode)
            _execute(engine, cfg, Path(output_dir), scopes, mode)
    finally:
        engine.dispose()
