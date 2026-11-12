import asyncio
import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.mcp import MCPAdapter
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.evaluation.scoring import filter_chunks
from finpipe.extraction.prompts import (
    PaperTaskSpecification,
    build_paper_summary_prompt,
)
from finpipe.extraction.settings import TaskExtractionConfig
from finpipe.extraction.specifications import unpack_datasets, unpack_metrics
from finpipe.integrations.tools import get_tools, init_mcp_servers
from finpipe.storage.models import (
    Chunk,
    Paper,
    PaperStatus,
    TaskCandidate,
    TaskCandidateStatus,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TestTargetDatasetCandidate,
    TrainInputDatasetCandidate,
    TrainTargetDatasetCandidate,
)
from finpipe.storage.status import set_paper_status

logger = logging.getLogger(__name__)


class TaskExtractor:
    def __init__(self, db: Engine, cfg: TaskExtractionConfig) -> None:
        """
        Extract candidate task metadata from accepted paper

        Args:
            db: SQLAlchemy db engine
            cfg: Task extraction configuration
        """
        self.db: Engine = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Run task extraction across provided scopes

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_extraction/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            asyncio.run(self.extract_scope(scope, output_path))

    async def extract_scope(self, scope: Scope, output_path: Path) -> None:
        """
        Extract task from accepted papers for a given scope

        Args:
            scope: Task scope
            output_path: Hydra run output path
        """

        with Session(self.db) as session:
            stmt = select(Paper).where(
                Paper.scope_id == scope.id, Paper.status == PaperStatus.ACCEPTED
            )
            papers: list[Paper] = session.execute(stmt).scalars().all()

        logger.info(f"Extracting tasks from {len(papers)} papers for scope {scope.id}")

        task_summaries = []
        task_candidates = []
        dataset_candidates = []
        metric_candidates = []

        system_prompt = build_paper_summary_prompt(scope)

        async with MCPAdapter(init_mcp_servers()) as mcp_adapter:
            tools = await get_tools(mcp_adapter)
            agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=system_prompt,
                response_format=ToolStrategy(
                    PaperTaskSpecification, handle_errors=False
                ),
            )

            for paper in papers:
                task_summary = await self._extract_paper(
                    agent, scope, paper, task_summaries
                )
                if not task_summary:
                    continue
                task_candidate, datasets, metrics = self._build_candidate(
                    scope, paper, task_summary
                )
                task_candidates.append(task_candidate)
                dataset_candidates.extend(datasets)
                metric_candidates.extend(metrics)

        with Session(self.db) as session:
            session.add_all(task_candidates)
            session.add_all(dataset_candidates)
            session.add_all(metric_candidates)
            session.commit()

        with open(output_path / f"{scope.id}.json", "w") as f:  # noqa: ASYNC230 - synchronous artifact boundary
            json.dump(
                [x.model_dump(mode="json") for x in task_summaries],
                f,
                indent=4,
            )

    async def _extract_paper(
        self, agent, scope: Scope, paper: Paper, summaries: list
    ) -> PaperTaskSpecification | None:
        with Session(self.db) as session:
            chunks = list(
                session.scalars(select(Chunk).where(Chunk.paper_id == paper.paper_id))
            )
        extract = "\n\n".join(x.text for x in filter_chunks(chunks))
        prompt = f"Title: {paper.title}\n\n{extract}"
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            task_summary = response["structured_response"]
            summaries.append(task_summary)
            set_paper_status(self.db, paper, PaperStatus.TASK_EXTRACTED)
            return task_summary
        except Exception as e:  # noqa: BLE001 - isolate failure to this pipeline item
            logger.error(
                f"Task extraction failed from paper {paper.paper_id} from scope {scope.id} due to {e}"
            )
            set_paper_status(self.db, paper, PaperStatus.TASK_EXTRACTION_FAILED)
            return None

    @staticmethod
    def _build_candidate(
        scope: Scope, paper: Paper, summary: PaperTaskSpecification
    ) -> tuple:
        candidate = TaskCandidate(
            scope_id=scope.id,
            paper_id=paper.paper_id,
            task_name=summary.task_name,
            task_type=scope.task_type,
            plan=summary.task_implementation_plan,
            instructions=summary.task_instructions,
            status=TaskCandidateStatus.NEW,
        )
        datasets = []
        for field, model in (
            ("training_inputs", TrainInputDatasetCandidate),
            ("training_targets", TrainTargetDatasetCandidate),
            ("test_inputs", TestInputDatasetCandidate),
            ("test_outputs", TestOutputDatasetCandidate),
            ("test_targets", TestTargetDatasetCandidate),
        ):
            records = unpack_datasets(model, candidate, getattr(summary, field))
            getattr(candidate, field).extend(records)
            datasets.extend(records)
        metrics = unpack_metrics(candidate, summary.assessment_metrics)
        candidate.assessment_metrics.extend(metrics)
        return candidate, datasets, metrics
