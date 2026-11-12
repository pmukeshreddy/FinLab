from contextlib import nullcontext

import pytest
from omegaconf import OmegaConf
from sqlalchemy.orm import Session

from finpipe.storage.models import Paper, PaperStatus


@pytest.fixture
def workflow_harness(monkeypatch, tmp_path):
    # The doubles replace network/model stages, while workflow routing and its
    # database mutations remain real. A dropped/reordered stage fails this test.
    from finpipe.app import workflow

    events = []
    settings = OmegaConf.create(
        {
            "db_engine": f"sqlite:///{tmp_path / 'state.db'}",
            "enable_task_critic": False,
            "enable_judge": False,
            "arxiv_id": "1706.03762v7",
            "scraping": {
                "since": "2026-01-01",
                "until": "2026-02-01",
                "max_papers_per_scope": 2,
            },
            "retrieval": {},
            "judge": {},
            "task_extractor": {},
            "task_critic": {},
            "task_generator": {},
            "task_exporter": {},
            "scopes": {
                "example": {
                    "id": "example",
                    "name": "Example",
                    "description": "Forecast returns",
                    "task_type": "forecasting",
                    "enabled": True,
                }
            },
        }
    )
    monkeypatch.setattr(workflow, "check_docker_available", lambda logger: None)
    monkeypatch.setattr(workflow, "init_context", lambda cfg, logger: nullcontext())
    monkeypatch.setattr(workflow, "load_dotenv", lambda: None)

    def stage(name):
        class FakeStage:
            def __init__(self, engine, cfg):
                self.engine = engine

            def run(self, output, scopes, *args, **kwargs):
                assert output == tmp_path
                events.append((name, [scope.id for scope in scopes], kwargs))
                if name == "extraction":
                    with Session(self.engine) as session:
                        paper = session.get(Paper, ("1706.03762v7", "example"))
                        if paper is not None:
                            events.append(("paper_status", paper.status))

            def scrape_paper(self, output, scope, paper_id):
                assert output == tmp_path
                events.append(("single_paper", paper_id, scope.id))
                paper = Paper(
                    paper_id=paper_id, scope_id=scope.id, status=PaperStatus.EXTRACTED
                )
                with Session(self.engine, expire_on_commit=False) as session:
                    session.add(paper)
                    session.commit()
                return paper

            def download_and_chunk_papers(self, output):
                assert output == tmp_path
                events.append(("documents",))

        return FakeStage

    for symbol, name in {
        "PaperScrapingPipeline": "discovery",
        "PaperRetrieval": "documents",
        "Judge": "evaluation",
        "TaskExtractor": "extraction",
        "TaskCritic": "review",
        "TaskGenerator": "generation",
        "TaskExporter": "export",
    }.items():
        monkeypatch.setattr(workflow, symbol, stage(name))
    return workflow, settings, events, tmp_path


def test_batch_preserves_stage_order_and_automatic_review(workflow_harness):
    workflow, settings, events, output = workflow_harness
    workflow.run_workflow(settings, output, "batch")
    assert events == [
        ("discovery", ["example"], {}),
        ("documents",),
        ("evaluation", ["example"], {}),
        ("extraction", ["example"], {}),
        ("review", ["example"], {"auto_approve": True}),
        ("generation", ["example"], {}),
        ("export", ["example"], {}),
    ]


def test_single_paper_skip_judge_accepts_paper_before_extraction(workflow_harness):
    workflow, settings, events, output = workflow_harness
    workflow.run_workflow(settings, output, "paper")
    assert events == [
        ("single_paper", "1706.03762v7", "example"),
        ("documents",),
        ("extraction", ["example"], {}),
        ("paper_status", PaperStatus.ACCEPTED),
        ("review", ["example"], {"auto_approve": True}),
        ("generation", ["example"], {}),
        ("export", ["example"], {}),
    ]


def test_optional_gates_are_used_when_enabled(workflow_harness):
    workflow, settings, events, output = workflow_harness
    settings.enable_judge = settings.enable_task_critic = True
    workflow.run_workflow(settings, output, "paper")
    assert ("evaluation", ["example"], {}) in events
    assert ("review", ["example"], {}) in events
    assert ("review", ["example"], {"auto_approve": True}) not in events


def test_batch_filters_disabled_scopes(workflow_harness):
    workflow, settings, events, output = workflow_harness
    settings.scopes.example.enabled = False
    workflow.run_workflow(settings, output, "batch")
    assert all(event[1] == [] for event in events if len(event) == 3)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("scopes", {}, "A single scope should be selected"),
        ("arxiv_id", None, "Field `arxiv_id` should be set"),
    ],
)
def test_single_paper_requires_scope_and_paper_id(
    workflow_harness, field, value, message
):
    workflow, settings, events, output = workflow_harness
    settings[field] = value
    with pytest.raises(AssertionError, match=message):
        workflow.run_workflow(settings, output, "paper")
    assert events == []
