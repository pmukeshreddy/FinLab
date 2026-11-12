"""Stable state values and decision payloads shared across pipeline features."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    FORECASTING = "forecasting"
    GENERATION = "generation"


class JudgeLabel(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class JudgeDecision(BaseModel):
    paper_id: str
    scope_id: str
    label: JudgeLabel
    score_0_10: float = Field(ge=0.0, le=10.0)
    reasons: str = ""
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    model: str
    raw_response: dict[str, Any] = Field(default_factory=dict)


class PaperStatus(StrEnum):
    SCRAPED = "scraped"
    EXTRACTED = "extracted"
    ERRORED = "errored"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TASK_EXTRACTED = "task_extracted"
    TASK_EXTRACTION_FAILED = "task_extraction_failed"


class SourceName(StrEnum):
    ARXIV = "arxiv"
    MANUAL = "manual"


class SourceFormat(StrEnum):
    HTML = "html"
    PDF = "pdf"


class RejectionReason(StrEnum):
    NoPaperURL = "no_paper_url"
    RetrievalError = "retrieval_error"
    PreFiltered = "pre_filtered"
    JudgeRejected = "judge_rejected"
    JudgeCutoff = "judge_cutoff"
    LLMError = "llm_error"
    DataNotPublic = "data_not_public"


class TaskCandidateStatus(StrEnum):
    NEW = "new"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSED = "processed"
    FAILED = "failed"


class TaskStatus(StrEnum):
    NEW = "new"
    EXPORT_FAILED = "export_failed"
    EXPORTED = "exported"
