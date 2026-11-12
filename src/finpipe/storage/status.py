"""Transactional updates for each independently keyed pipeline record."""

from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from finpipe.storage.models import (
    Paper,
    PaperStatus,
    Task,
    TaskCandidate,
    TaskCandidateStatus,
    TaskStatus,
)


def _update(db: Engine, model, predicates: tuple, changes: dict) -> None:
    statement = update(model).where(*predicates).values(changes)
    with Session(db) as session, session.begin():
        session.execute(statement)


def set_paper_status(db: Engine, paper: Paper, status: PaperStatus, **kwargs) -> None:
    _update(
        db,
        Paper,
        (Paper.scope_id == paper.scope_id, Paper.paper_id == paper.paper_id),
        {"status": status, **kwargs},
    )


def set_task_candidate_status(
    db: Engine, task_id: int, status: TaskCandidateStatus, **kwargs
) -> None:
    _update(
        db,
        TaskCandidate,
        (TaskCandidate.task_id == task_id,),
        {"status": status, **kwargs},
    )


def set_task_status(db: Engine, task_id: int, status: TaskStatus, **kwargs) -> None:
    _update(db, Task, (Task.task_id == task_id,), {"status": status, **kwargs})
