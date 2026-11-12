import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.evaluation.models import JudgeResult
from finpipe.evaluation.prompts import (
    Evidence,
    PrefilterDecision,
    SiftJudgement,
    build_prefilter_prompt,
    build_sift_judge_prompt,
)
from finpipe.evaluation.scoring import filter_chunks, rank_papers_for_llm
from finpipe.evaluation.settings import JudgeConfig
from finpipe.storage.models import (
    Chunk,
    JudgeLabel,
    Paper,
    PaperStatus,
    RejectionReason,
)
from finpipe.storage.status import set_paper_status

logger = logging.getLogger(__name__)


class Judge:
    def __init__(
        self,
        db: Engine,
        cfg: JudgeConfig,
    ) -> None:
        """
        Judge papers for acceptance to the task creation stage

        Args:
            db: SqlAlchemy database engine
            cfg: Judge configuration
        """
        self.db = db
        self.cfg = cfg
        self.llm = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(
        self,
        output_path: Path,
        scopes: list[Scope],
    ) -> None:
        """
        Run the judgement process across the provided scopes and update db records

        Args:
            output_path: Hydra run output path
            scopes: List of task scopes
        """
        output_path = output_path / "judgments/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            self.judge_scope(output_path, scope)

    def judge_scope(
        self,
        output_path: Path,
        scope: Scope,
    ) -> None:
        """
        Run judgement across papers for a given scope and update db records

        Args:
            output_path: Hydra run output path
            scope: Task scope
        """
        with Session(self.db) as session:
            stmt = select(Paper).where(
                Paper.scope_id == scope.id, Paper.status == PaperStatus.EXTRACTED
            )
            scope_papers = session.execute(stmt).scalars().all()

        logger.info(f"Judging {len(scope_papers)} papers for scope {scope.id}")
        candidates, judgements = self._prefilter_papers(scope, scope_papers)
        ranked = rank_papers_for_llm(
            candidates,
            citation_boost=self.cfg.ranking_citation_boost,
            recency_weight=self.cfg.ranking_recency_weight,
            recency_half_life_days=self.cfg.ranking_recency_half_life_days,
        )
        selected = ranked[: self.cfg.sift_budget]
        for paper in ranked[len(selected) :]:
            self.set_paper_status(
                paper,
                PaperStatus.REJECTED,
                rejection_reason=RejectionReason.JudgeCutoff,
            )
            judgements.append(
                JudgeResult(
                    paper_id=paper.paper_id,
                    scope_id=scope.id,
                    accepted=False,
                    rejection_reason=RejectionReason.JudgeCutoff,
                    reasoning="Did not make judge paper cutoff",
                )
            )
        judgements.extend(self._judge_paper(scope, paper) for paper in selected)
        with open(output_path / f"{scope.id}.json", "w") as f:
            json.dump([x.model_dump(mode="json") for x in judgements], f, indent=4)

    def _prefilter_papers(
        self, scope: Scope, papers: list[Paper]
    ) -> tuple[list[Paper], list[JudgeResult]]:
        candidates, judgements = [], []
        for paper in papers:
            passed, score, result = (
                self.llm_prefilter(scope, paper)
                if self.cfg.prefilter_enabled
                else (True, 0.0, None)
            )
            self.set_paper_status(
                paper,
                PaperStatus.EXTRACTED if passed else PaperStatus.REJECTED,
                prefilter_passed=passed,
                prefilter_score=score,
                **(
                    {}
                    if passed
                    else {"rejection_reason": RejectionReason.JudgeRejected}
                ),
            )
            if passed:
                candidates.append(paper)
            else:
                judgements.append(
                    JudgeResult(
                        paper_id=paper.paper_id,
                        scope_id=scope.id,
                        accepted=False,
                        rejection_reason=RejectionReason.JudgeRejected,
                        reasoning=result.reasons,
                    )
                )
        return candidates, judgements

    def _judge_paper(self, scope: Scope, paper: Paper) -> JudgeResult:
        with Session(self.db) as session:
            chunks = list(
                session.scalars(select(Chunk).where(Chunk.paper_id == paper.paper_id))
            )
        excerpt = "\n\n".join(x.text for x in filter_chunks(chunks))
        prompt = build_sift_judge_prompt(scope, paper, excerpt=excerpt)
        llm = self.llm.with_structured_output(SiftJudgement)
        rejection_reason = None
        try:
            response: SiftJudgement = llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001 - isolate failure to this pipeline item
            logger.error(
                f"LLM judgement failed for paper {paper.paper_id} from scope {paper.scope_id}: {e}"
            )
            response = SiftJudgement(
                label=JudgeLabel.REJECTED,
                score=0,
                confidence=0.0,
                reasons=f"LLM error for this paper {e}",
                evidence=Evidence(experiments="", datasets="", metrics=""),
                data_publicly_available=False,
                data_availability_reasoning=f"LLM error for this paper {e}",
            )
            rejection_reason = RejectionReason.LLMError
        accepted = (
            response.label == JudgeLabel.ACCEPTED
            and response.score >= self.cfg.threshold_default
        )
        if (
            not accepted
            and rejection_reason is None
            and not response.data_publicly_available
        ):
            rejection_reason = RejectionReason.DataNotPublic
        self.set_paper_status(
            paper,
            PaperStatus.ACCEPTED if accepted else PaperStatus.REJECTED,
            rejection_reason=rejection_reason,
        )
        return JudgeResult(
            paper_id=paper.paper_id,
            scope_id=scope.id,
            accepted=accepted,
            rejection_reason=rejection_reason,
            reasoning=(
                f"{response.reasons}\n\n"
                f"Data availability: {response.data_availability_reasoning}"
            ),
        )

    def llm_prefilter(
        self, scope: Scope, paper: Paper
    ) -> tuple[bool, float, PrefilterDecision]:
        """
        Pre-filter papers using an LLM judge

        Args:
            scope: Task scope
            paper: Paper record

        Returns:
            Judgement outcome
        """
        llm = self.llm.with_structured_output(PrefilterDecision)
        prompt = build_prefilter_prompt(scope, paper)

        try:
            result: PrefilterDecision = llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001 - isolate failure to this pipeline item
            logger.error(
                f"LLM prefilter failed for paper {paper.paper_id} from scope {paper.scope_id}: {e}"
            )
            result: PrefilterDecision = PrefilterDecision(
                label=JudgeLabel.REJECTED,
                relevance_score=0.0,
                confidence=0.0,
                reasons=f"prefilter llm error: {e}",
            )

        score = result.relevance_score
        conf = result.confidence
        keep = result.label == JudgeLabel.ACCEPTED
        min_score = self.cfg.prefilter_llm_score_min
        min_conf = self.cfg.prefilter_llm_confidence_min
        passed = keep and score >= min_score and conf >= min_conf
        return passed, score, result

    def set_paper_status(self, paper: Paper, status: PaperStatus, **kwargs) -> None:
        """
        Update paper status in the DB

        Args:
            paper: Current paper instance
            status: Status value
        """
        set_paper_status(self.db, paper, status, **kwargs)
