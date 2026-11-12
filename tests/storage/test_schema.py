import json
from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from finpipe.storage.models import Paper, PaperStatus, RejectionReason
from finpipe.storage.status import set_paper_status


def test_database_schema_matches_existing_databases(db):
    inspector = inspect(db)
    schema = {}
    for table in sorted(inspector.get_table_names()):
        schema[table] = {
            "columns": [
                {**column, "type": str(column["type"])}
                for column in inspector.get_columns(table)
            ],
            "pk": inspector.get_pk_constraint(table),
            "foreign_keys": inspector.get_foreign_keys(table),
            "unique": inspector.get_unique_constraints(table),
            "indexes": inspector.get_indexes(table),
        }
    baseline = Path(__file__).parents[1] / "fixtures" / "sqlite_schema.json"
    assert schema == json.loads(baseline.read_text())


def test_paper_updates_are_scoped_and_preserve_other_rows(db):
    with Session(db) as session:
        session.add_all(
            [
                Paper(
                    paper_id="same-paper", scope_id="one", status=PaperStatus.SCRAPED
                ),
                Paper(
                    paper_id="same-paper", scope_id="two", status=PaperStatus.SCRAPED
                ),
            ]
        )
        session.commit()
    set_paper_status(
        db,
        Paper(paper_id="same-paper", scope_id="one"),
        PaperStatus.REJECTED,
        rejection_reason=RejectionReason.DataNotPublic,
    )
    with Session(db) as session:
        rejected = session.get(Paper, ("same-paper", "one"))
        assert rejected.status == PaperStatus.REJECTED
        assert rejected.rejection_reason == RejectionReason.DataNotPublic
        assert session.get(Paper, ("same-paper", "two")).status == PaperStatus.SCRAPED
