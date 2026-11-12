import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.extraction.specifications import (
    TaskSpecification,
    build_task_specification,
)
from finpipe.review.models import CritiqueResult
from finpipe.review.prompts import TaskCritique, build_critique_prompt
from finpipe.review.rules import structural_issues
from finpipe.review.settings import TaskCriticConfig
from finpipe.storage.models import JudgeLabel, TaskCandidate, TaskCandidateStatus
from finpipe.storage.status import set_task_candidate_status

logger = logging.getLogger(__name__)


class TaskCritic:
    def __init__(self, db: Engine, cfg: TaskCriticConfig) -> None:
        self.db = db
        self.cfg = cfg
        self.llm: BaseChatModel = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(
        self, output_path: Path, scopes: list[Scope], auto_approve: bool = False
    ) -> None:
        output_path = output_path / "task_critic/"
        output_path.mkdir(exist_ok=True)
        for scope in scopes:
            if auto_approve:
                self.approve_scope(scope)
            else:
                self.critique_scope(output_path, scope)

    def approve_scope(self, scope: Scope) -> None:
        with Session(self.db) as session:
            candidates = list(session.scalars(self._new_candidates_query(scope)))
        for candidate in candidates:
            set_task_candidate_status(
                self.db, candidate.task_id, TaskCandidateStatus.APPROVED
            )

    @staticmethod
    def _new_candidates_query(scope: Scope):
        return select(TaskCandidate).where(
            TaskCandidate.scope_id == scope.id,
            TaskCandidate.status == TaskCandidateStatus.NEW,
        )

    def critique_scope(self, output_path: Path, scope: Scope) -> None:
        with Session(self.db) as session:
            candidates = session.scalars(self._new_candidates_query(scope))
            specs = [build_task_specification(candidate) for candidate in candidates]

        logger.info(f"Critiquing {len(specs)} task candidates for scope {scope.id}")

        results = [self.critique_task(scope, spec) for spec in specs]

        with open(output_path / f"{scope.id}.json", "w") as f:
            json.dump([x.model_dump(mode="json") for x in results], f, indent=4)

    def critique_task(self, scope: Scope, spec: TaskSpecification) -> CritiqueResult:
        issues = structural_issues(spec, scope.task_params)
        if issues:
            self.set_status(spec.id, TaskCandidateStatus.REJECTED)
            return CritiqueResult(
                task_id=spec.id,
                scope_id=scope.id,
                approved=False,
                score=None,
                confidence=None,
                reasoning=f"Rejected on structural checks: {'; '.join(issues)}",
            )

        result = self._llm_critique(scope, spec)
        self.set_status(
            spec.id,
            TaskCandidateStatus.APPROVED
            if result.approved
            else TaskCandidateStatus.REJECTED,
        )
        return result

    def _llm_critique(self, scope: Scope, spec: TaskSpecification) -> CritiqueResult:
        prompt = build_critique_prompt(scope, spec)
        llm = self.llm.with_structured_output(TaskCritique)

        try:
            response: TaskCritique = llm.invoke(prompt)
            approved = (
                response.label == JudgeLabel.ACCEPTED
                and response.score >= self.cfg.threshold_default
            )
            reasoning = (
                f"{response.issues}\n\n"
                f"Consistency: {response.consistency_assessment}\n"
                f"Completeness: {response.completeness_assessment}\n"
                f"Data Availability: {response.data_availability_assessment}"
            )
            score = response.score
            confidence = response.confidence
        except Exception as e:  # noqa: BLE001 - isolate failure to this pipeline item
            logger.error(
                f"Task critique failed for task {spec.id} scope {scope.id}: {e}"
            )
            approved = False
            reasoning = f"LLM error during critique: {e}"
            score = None
            confidence = None

        return CritiqueResult(
            task_id=spec.id,
            scope_id=scope.id,
            approved=approved,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )

    def set_status(self, task_id: int, status: TaskCandidateStatus) -> None:
        set_task_candidate_status(self.db, task_id, status)
