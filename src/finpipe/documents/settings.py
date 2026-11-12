from dataclasses import dataclass

from finpipe.storage.models import SourceFormat


@dataclass
class RetrievalConfig:
    source_preference: list[SourceFormat]
    arxiv_html_url: str
    request_interval_sec: float
    timeout_sec: float
    user_agent: str
