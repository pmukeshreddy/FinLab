"""Offline characterization of stage boundaries and partial-failure behavior."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.documents.service import PaperRetrieval
from finpipe.documents.settings import RetrievalConfig
from finpipe.export.service import TaskExporter
from finpipe.export.settings import TaskConfig, TaskExportConfig
from finpipe.extraction import service as extraction
from finpipe.extraction.prompts import PaperTaskSpecification
from finpipe.generation import service as generation
from finpipe.generation.prompts import Assessment, DatasetRetrieval
from finpipe.generation.settings import TaskGenerationConfig
from finpipe.storage.models import (
    Chunk,
    Paper,
    PaperStatus,
    RejectionReason,
    SourceFormat,
    Task,
    TaskCandidate,
    TaskCandidateStatus,
    TaskStatus,
)


@pytest.fixture
def scope():
    return Scope(id="scope", name="Scope", task_type="forecasting", description="")


def paper(db, paper_id, **kwargs):
    with Session(db) as session:
        session.add(Paper(paper_id=paper_id, scope_id="scope", title="Title", **kwargs))
        session.commit()


def get_status(db, cls, identity):
    with Session(db) as session:
        return session.get(cls, identity).status


def fake_mcp(monkeypatch, module, agent):
    adapter = MagicMock()
    adapter.__aenter__ = AsyncMock(return_value=adapter)
    adapter.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module, "MCPAdapter", lambda _: adapter)
    monkeypatch.setattr(module, "init_mcp_servers", dict)
    monkeypatch.setattr(module, "get_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(module, "create_agent", lambda **_: agent)


def test_retrieval_shares_chunks_across_scopes_and_rejects_missing_url(db, tmp_path):
    paper(db, "cached", status=PaperStatus.SCRAPED)
    paper(db, "missing", status=PaperStatus.SCRAPED)
    with Session(db) as session:
        session.add(
            Chunk(paper_id="cached", chunk_index=0, header="data", text="cached")
        )
        session.commit()
    retrieval = PaperRetrieval(
        db,
        RetrievalConfig(["html", "pdf"], "https://arxiv.org/html/{id}", 0, 1, "test"),
    )
    retrieval.download_and_chunk_papers(tmp_path)
    with Session(db) as session:
        cached, missing = session.execute(
            select(Paper).order_by(Paper.paper_id)
        ).scalars()
        assert cached.status == PaperStatus.EXTRACTED
        assert missing.status == PaperStatus.REJECTED
        assert missing.rejection_reason == RejectionReason.NoPaperURL


def test_retrieval_failure_continues_to_next_paper(db, tmp_path, monkeypatch):
    paper(db, "bad", status=PaperStatus.SCRAPED, pdf_url="bad")
    paper(db, "good", status=PaperStatus.SCRAPED, pdf_url="good")
    retrieval = PaperRetrieval(
        db,
        RetrievalConfig(["html", "pdf"], "https://arxiv.org/html/{id}", 0, 1, "test"),
    )

    def retrieve(p):
        if p.paper_id == "bad":
            raise ValueError("conversion failed")
        return "# Data\ncontent", SourceFormat.HTML

    monkeypatch.setattr(retrieval, "retrieve_markdown", retrieve)
    retrieval.download_and_chunk_papers(tmp_path)
    with Session(db) as session:
        records = {p.paper_id: p for p in session.execute(select(Paper)).scalars()}
        assert records["bad"].status == PaperStatus.ERRORED
        assert records["bad"].rejection_reason == RejectionReason.RetrievalError
        assert records["good"].status == PaperStatus.EXTRACTED
        assert session.execute(select(Chunk)).scalar_one().paper_id == "good"


def test_retrieval_unavailable_format_falls_back_but_conversion_error_propagates(
    db, monkeypatch
):
    retrieval = PaperRetrieval(
        db,
        RetrievalConfig(["html", "pdf"], "https://arxiv.org/html/{id}", 0, 1, "test"),
    )
    p = Paper(paper_id="p", scope_id="scope", pdf_url="pdf")
    monkeypatch.setattr(retrieval, "source_url", lambda _, fmt: fmt.value)
    monkeypatch.setattr(retrieval, "get", lambda url: None if url == "html" else b"pdf")
    monkeypatch.setattr(
        "finpipe.documents.service.pdf_to_markdown", lambda _: "converted"
    )
    assert retrieval.retrieve_markdown(p) == ("converted", SourceFormat.PDF)
    monkeypatch.setattr(retrieval, "get", lambda _: b"invalid")
    monkeypatch.setattr(
        "finpipe.documents.service.arxiv_html_to_markdown",
        MagicMock(side_effect=ValueError("bad html")),
    )
    with pytest.raises(ValueError, match="bad html"):
        retrieval.retrieve_markdown(p)


def summary():
    data = {
        "name": " Input Data ",
        "filename": "data.csv",
        "description": "Data",
        "source": "public",
        "relevant_urls": [],
        "download_link": None,
    }
    return PaperTaskSpecification(
        reasoning="why",
        task_name="Task",
        task_implementation_plan="plan",
        task_instructions="instructions",
        training_inputs=[data],
        training_targets=[],
        test_inputs=[],
        test_outputs=[data],
        test_targets=[data],
        assessment_metrics=[
            {"name": "mse", "description": "metric", "input_datasets": ["data.csv"]}
        ],
    )


def test_extraction_persists_relationships_and_continues_after_agent_error(
    db, scope, tmp_path, monkeypatch
):
    paper(db, "bad", status=PaperStatus.ACCEPTED)
    paper(db, "good", status=PaperStatus.ACCEPTED)
    extractor = extraction.TaskExtractor.__new__(extraction.TaskExtractor)
    extractor.db, extractor.llm = db, MagicMock()
    agent = SimpleNamespace(
        ainvoke=AsyncMock(
            side_effect=[
                RuntimeError("offline error"),
                {"structured_response": summary()},
            ]
        )
    )
    fake_mcp(monkeypatch, extraction, agent)
    asyncio.run(extractor.extract_scope(scope, tmp_path))
    with Session(db) as session:
        records = {
            p.paper_id: p.status for p in session.execute(select(Paper)).scalars()
        }
        assert records == {
            "bad": PaperStatus.TASK_EXTRACTION_FAILED,
            "good": PaperStatus.TASK_EXTRACTED,
        }
        candidate = session.execute(select(TaskCandidate)).scalar_one()
        assert candidate.status == TaskCandidateStatus.NEW
        assert candidate.paper_id == "good"
        assert [d.name for d in candidate.training_inputs] == ["input_data"]
        assert (
            len(candidate.test_targets)
            == len(candidate.test_outputs)
            == len(candidate.assessment_metrics)
            == 1
        )
        assert candidate.training_targets == candidate.test_inputs == []
    assert json.loads((tmp_path / "scope.json").read_text()) == [
        summary().model_dump(mode="json")
    ]
    assert agent.ainvoke.call_args.args[0] == {
        "messages": [{"role": "user", "content": "Title: Title\n\n"}]
    }


def candidate(db):
    with Session(db) as session:
        obj = TaskCandidate(
            scope_id="scope",
            paper_id="p",
            task_name="Task",
            task_type="forecasting",
            plan="plan",
            instructions="instructions",
            status=TaskCandidateStatus.APPROVED,
        )
        session.add(obj)
        session.commit()
        return obj.task_id


@pytest.mark.parametrize("failure", [None, "pylint", "train", "test", "agent"])
def test_generation_artifacts_status_and_validation_failures(
    db, scope, tmp_path, monkeypatch, failure
):
    task_id = candidate(db)
    monkeypatch.setattr(
        generation, "instantiate", lambda _: MagicMock(spec=generation.BaseChatModel)
    )
    generator = generation.TaskGenerator(
        db, TaskGenerationConfig(llm=None, templates_path=None)
    )
    dataset = DatasetRetrieval(
        training_script="train = 1", testing_script="test = 1", requirements=["numpy"]
    )
    agent = SimpleNamespace(
        ainvoke=AsyncMock(side_effect=RuntimeError("agent failed"))
        if failure == "agent"
        else AsyncMock(return_value={"structured_response": dataset})
    )
    fake_mcp(monkeypatch, generation, agent)
    generator.llm.with_structured_output.return_value.invoke.return_value = Assessment(
        assessment_script="verify = 1", requirements=["numpy", "pandas"]
    )
    generator.llm.invoke.side_effect = [
        SimpleNamespace(content="description"),
        SimpleNamespace(content="difficulty"),
    ]
    monkeypatch.setattr(
        generation, "run_pylint", lambda _: (failure != "pylint", "lint logs")
    )
    monkeypatch.setattr(
        generation,
        "test_train_download_script",
        lambda *_: (failure != "train", "train logs"),
    )
    monkeypatch.setattr(
        generation,
        "test_test_download_script",
        lambda *_: (failure != "test", "test logs"),
    )
    asyncio.run(generator.extract_scope(scope, tmp_path))
    assert get_status(db, TaskCandidate, task_id) == (
        TaskCandidateStatus.PROCESSED if failure is None else TaskCandidateStatus.FAILED
    )
    with Session(db) as session:
        tasks = session.execute(select(Task)).scalars().all()
        assert len(tasks) == (1 if failure is None else 0)
        if tasks:
            task = tasks[0]
            assert task.status == TaskStatus.NEW
            assert task.short_description == "description"
            assert task.difficulty_explanation == "difficulty"
            assert set(task.requirements) == {"numpy", "pandas"}
            assert task.train_script == (tmp_path / "scripts/Task/train.py").read_text()
            assert (
                task.instructions
                == (tmp_path / "scripts/Task/instruction.md").read_text()
            )


def test_export_writes_all_artifacts_and_preserves_existing_directory_failure(
    db, scope, tmp_path
):
    task_id = candidate(db)
    with Session(db) as session:
        obj = Task(
            task_candidate_id=task_id,
            name="A / Task",
            status=TaskStatus.NEW,
            train_script="train",
            test_script="test",
            assessment_script="verify",
            requirements=["numpy"],
            instructions="instructions",
            short_description='quoted "description"',
            difficulty_explanation="difficulty",
        )
        session.add(obj)
        session.commit()
        export_id = obj.task_id
    exporter = TaskExporter(
        db, TaskExportConfig(str(tmp_path / "export"), None, TaskConfig("org"))
    )
    exporter.extract_scope(scope, tmp_path)
    task_dir = tmp_path / "export/a_task"
    assert get_status(db, Task, export_id) == TaskStatus.EXPORTED
    assert {
        str(p.relative_to(task_dir)) for p in task_dir.rglob("*") if p.is_file()
    } == {
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/docker-compose.yaml",
        "environment/data.py",
        "tests/Dockerfile",
        "tests/docker-compose.yaml",
        "tests/data.py",
        "tests/verifier.py",
        "tests/test.sh",
    }
    assert (task_dir / "environment/data.py").read_text() == "train"
    assert (task_dir / "tests/data.py").read_text() == "test"
    assert (task_dir / "tests/verifier.py").read_text() == "verify"
    with Session(db) as session:
        session.get(Task, export_id).status = TaskStatus.NEW
        session.commit()
    exporter.extract_scope(scope, tmp_path)
    assert get_status(db, Task, export_id) == TaskStatus.EXPORT_FAILED


def test_feature_services_use_explicit_template_directory(db, tmp_path, monkeypatch):
    import shutil

    from finpipe.resources import template_directory

    custom = tmp_path / "custom"
    shutil.copytree(template_directory(), custom)
    (custom / "instructions.md.j2").write_text("custom instructions")
    (custom / "task.toml.j2").write_text("custom metadata")
    monkeypatch.setattr(
        generation, "instantiate", lambda _: MagicMock(spec=generation.BaseChatModel)
    )
    generator = generation.TaskGenerator(
        db, TaskGenerationConfig(llm=None, templates_path=str(custom))
    )
    exporter = TaskExporter(
        db, TaskExportConfig(str(tmp_path / "export"), str(custom), TaskConfig("org"))
    )
    assert generator.instructions_template.render() == "custom instructions"
    assert exporter.task_meta_template.render() == "custom metadata"
