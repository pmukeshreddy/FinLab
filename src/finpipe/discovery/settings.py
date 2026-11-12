from dataclasses import dataclass

from finpipe.discovery.sources.settings import (
    ArxivConfig,
    CrossrefConfig,
    SemanticScholarConfig,
)


@dataclass
class ScrapingConfig:
    arxiv: ArxivConfig
    semantic_scholar: SemanticScholarConfig
    crossref: CrossrefConfig
    max_papers_per_scope: int
    since: str
    until: str
