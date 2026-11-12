import tomllib

import pytest
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.export.naming import slugify
from finpipe.export.service import TaskExporter
from finpipe.export.settings import TaskConfig, TaskExportConfig
from finpipe.storage.models import Task, TaskStatus


@pytest.mark.parametrize(
    "description",
    [
        "Predicts equity returns from a 30-day window.",
        'Predicts the "alpha" signal from order flow.',
        "Predicts returns.\n\nUses an LSTM.",
        r"Uses a \tau-scaled loss.",
        "Forecast returns 📈 and compare café prices in 東京.",
        "Uses <alpha> & 'beta' signals.",
        "Includes DEL: \x7f and controls: \x00\b\f\t\r\x1f.",
    ],
)
def test_exported_task_config_survives_llm_prose(
    db, tmp_path, description: str
) -> None:
    difficulty = f"Difficulty: {description}"
    with Session(db) as session:
        task = Task(
            name="Task",
            status=TaskStatus.NEW,
            train_script="train",
            test_script="test",
            assessment_script="verify",
            requirements=[],
            instructions="instructions",
            short_description=description,
            difficulty_explanation=difficulty,
        )
        session.add(task)
        session.commit()
        task_id = task.task_id
    exporter = TaskExporter(
        db, TaskExportConfig(str(tmp_path / "export"), None, TaskConfig("org"))
    )
    exporter.extract_scope(
        Scope(id="scope", name="Scope", task_type="forecasting", description=""),
        tmp_path,
    )

    config_text = (tmp_path / "export/task/task.toml").read_text()
    config = tomllib.loads(config_text)
    assert config["task"]["description"] == description
    assert config["metadata"]["difficulty_explanation"] == difficulty
    assert all(char not in config_text for char in "<>&'")
    with Session(db) as session:
        assert session.get(Task, task_id).status == TaskStatus.EXPORTED


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Stock Return Forecasting", "stock_return_forecasting"),
        ("LOB/mid-price prediction", "lob_mid_price_prediction"),
        ("VaR (95%) backtest", "var_95_backtest"),
        ("../escape", "escape"),
        ("!!!", "task"),
    ],
)
def test_slugify_keeps_the_name_inside_the_export_directory(
    name: str, expected: str
) -> None:
    assert slugify(name) == expected
