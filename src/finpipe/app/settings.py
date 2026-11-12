"""
Pipeline configuration classes
"""

from dataclasses import dataclass, field

from finpipe.contracts.task_types import (
    ForecastingParams,
    GenerationParams,
    TaskTypeParams,
)
from finpipe.discovery.settings import ScrapingConfig
from finpipe.documents.settings import RetrievalConfig
from finpipe.evaluation.settings import JudgeConfig
from finpipe.export.settings import TaskExportConfig
from finpipe.extraction.settings import TaskExtractionConfig
from finpipe.generation.settings import TaskGenerationConfig
from finpipe.review.settings import TaskCriticConfig
from finpipe.storage.models import TaskType


@dataclass
class Scope:
    id: str
    name: str
    task_type: TaskType
    task_params: TaskTypeParams = field(init=False)
    description: str
    enabled: bool = True
    queries: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.task_type = TaskType(self.task_type)
        self.task_params = {
            TaskType.FORECASTING: ForecastingParams,
            TaskType.GENERATION: GenerationParams,
        }[self.task_type]


def scope_context(scope: Scope) -> str:
    query_block = "\n".join(f"- {q}" for q in scope.queries[:10]) or "- (none)"
    categories = ", ".join(scope.categories) or "(none)"
    details = (
        ("name", scope.name),
        ("task type", scope.task_type),
        ("description", scope.description),
        ("categories", categories),
    )
    return (
        "\n".join(f"Scope {label}: {value}" for label, value in details)
        + f"\nScope queries:\n{query_block}"
    )


@dataclass
class PipelineConfig:
    db_engine: str
    enable_task_critic: bool
    scraping: ScrapingConfig
    retrieval: RetrievalConfig
    judge: JudgeConfig
    task_extractor: TaskExtractionConfig
    task_critic: TaskCriticConfig
    task_generator: TaskGenerationConfig
    task_exporter: TaskExportConfig
    scopes: dict[str, Scope]


@dataclass
class ManualPipelineConfig:
    arxiv_id: str | None
    scopes: dict[str, Scope]
    scraping: ScrapingConfig
    retrieval: RetrievalConfig
    enable_judge: bool
    enable_task_critic: bool
    judge: JudgeConfig
    task_extractor: TaskExtractionConfig
    task_critic: TaskCriticConfig
    task_generator: TaskGenerationConfig
    task_exporter: TaskExportConfig
