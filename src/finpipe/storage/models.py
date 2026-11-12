from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite.json import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from finpipe.contracts.states import (
    JudgeDecision,
    JudgeLabel,
    PaperStatus,
    RejectionReason,
    SourceFormat,
    SourceName,
    TaskCandidateStatus,
    TaskStatus,
    TaskType,
)


class Base(DeclarativeBase):
    """Declarative registry for the persisted pipeline state."""


class Paper(Base):
    __tablename__ = "papers"
    paper_id = mapped_column(String, primary_key=True)
    scope_id = mapped_column(String, primary_key=True)
    title = mapped_column(String)
    abstract = mapped_column(String)
    source = mapped_column(Enum(SourceName))
    authors = mapped_column(JSON)
    categories = mapped_column(JSON)
    published_at = mapped_column(DateTime, nullable=True)
    updated_at = mapped_column(DateTime, nullable=True)
    arxiv_url = mapped_column(String, nullable=True)
    pdf_url = mapped_column(String, nullable=True)
    primary_category = mapped_column(String, nullable=True)
    doi = mapped_column(String, nullable=True)
    citation_count = mapped_column(Integer)
    semantic_scholar_id = mapped_column(String, nullable=True)
    influential_citation_count = mapped_column(Integer)
    venue = mapped_column(String, nullable=True)
    journal_name = mapped_column(String, nullable=True)
    publication_types = mapped_column(JSON)
    peer_reviewed = mapped_column(Boolean)
    prefilter_score = mapped_column(Float)
    prefilter_passed = mapped_column(Boolean)
    status = mapped_column(Enum(PaperStatus), index=True)
    rejection_reason = mapped_column(Enum(RejectionReason), nullable=True)
    __table_args__ = (
        UniqueConstraint("paper_id", "scope_id", name="paper_constraint"),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id = mapped_column(String, index=True)
    chunk_index = mapped_column(Integer, index=True)
    header = mapped_column(String)
    text = mapped_column(String)
    source_format = mapped_column(Enum(SourceFormat))


class TaskCandidate(Base):
    __tablename__ = "task_candidates"
    task_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_id = mapped_column(String, index=True)
    paper_id = mapped_column(String, index=True)
    new = mapped_column(Boolean, default=True)
    status = mapped_column(Enum(TaskCandidateStatus), index=True)
    task_type = mapped_column(Enum(TaskType))
    task_name = mapped_column(String)
    plan = mapped_column(String)
    instructions = mapped_column(String)
    training_inputs: Mapped[list["TrainInputDatasetCandidate"]] = relationship(
        "TrainInputDatasetCandidate", backref="task_candidate"
    )
    training_targets: Mapped[list["TrainTargetDatasetCandidate"]] = relationship(
        "TrainTargetDatasetCandidate", backref="task_candidate"
    )
    test_inputs: Mapped[list["TestInputDatasetCandidate"]] = relationship(
        "TestInputDatasetCandidate", backref="task_candidate"
    )
    test_outputs: Mapped[list["TestOutputDatasetCandidate"]] = relationship(
        "TestOutputDatasetCandidate", backref="task_candidate"
    )
    test_targets: Mapped[list["TestTargetDatasetCandidate"]] = relationship(
        "TestTargetDatasetCandidate", backref="task_candidate"
    )
    assessment_metrics: Mapped[list["MetricCandidate"]] = relationship(
        "MetricCandidate", backref="task_candidate"
    )


class BaseDatasetCandidate(Base):
    __abstract__ = True
    dataset_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_candidates.task_id"))
    name = mapped_column(String)
    filename = mapped_column(String)
    description = mapped_column(String)
    source = mapped_column(String)
    relevant_urls = mapped_column(JSON)
    download_link = mapped_column(String, nullable=True)


class TrainInputDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "train_input_dataset_candidates"


class TrainTargetDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "train_target_dataset_candidates"


class TestInputDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "test_input_dataset_candidates"


class TestOutputDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "test_output_dataset_candidates"


class TestTargetDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "test_target_dataset_candidates"


class MetricCandidate(Base):
    __tablename__ = "metric_candidates"
    metric_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_candidates.task_id"))
    name = mapped_column(String)
    description = mapped_column(String)
    input_datasets = mapped_column(JSON)


class Task(Base):
    __tablename__ = "task"
    task_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    name = mapped_column(String)
    task_candidate_id = mapped_column(Integer, index=True)
    status = mapped_column(Enum(TaskStatus))
    train_script = mapped_column(String)
    test_script = mapped_column(String)
    assessment_script = mapped_column(String)
    requirements = mapped_column(JSON)
    instructions = mapped_column(String)
    short_description = mapped_column(String)
    difficulty_explanation = mapped_column(String)


__all__ = [
    "Base",
    "BaseDatasetCandidate",
    "Chunk",
    "JudgeDecision",
    "JudgeLabel",
    "MetricCandidate",
    "Paper",
    "PaperStatus",
    "RejectionReason",
    "SourceFormat",
    "SourceName",
    "Task",
    "TaskCandidate",
    "TaskCandidateStatus",
    "TaskStatus",
    "TaskType",
    "TestInputDatasetCandidate",
    "TestOutputDatasetCandidate",
    "TestTargetDatasetCandidate",
    "TrainInputDatasetCandidate",
    "TrainTargetDatasetCandidate",
]
