import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.export.naming import slugify
from finpipe.export.settings import TaskExportConfig
from finpipe.resources import template_directory
from finpipe.storage.models import Task, TaskStatus
from finpipe.storage.status import set_task_status

logger = logging.getLogger(__name__)


def _toml_compatible_json_dumps(value, **kwargs) -> str:
    # TOML forbids JSON surrogate escapes and requires DEL to be escaped.
    kwargs["ensure_ascii"] = False
    return json.dumps(value, **kwargs).replace("\x7f", "\\u007f")


class TaskExporter:
    def __init__(self, db: Engine, cfg: TaskExportConfig) -> None:
        self.db = db
        self.export_path = Path(cfg.export_path)
        self.export_path.mkdir(exist_ok=True)
        self.template_env = Environment(
            loader=FileSystemLoader(template_directory(cfg.templates_path))
        )
        self.template_env.policies["json.dumps_function"] = _toml_compatible_json_dumps

        def strformat(v, fmt):
            return fmt % v

        self.template_env.filters["strformat"] = strformat

        self.task_meta_template = self.template_env.get_template("task.toml.j2")
        self.train_docker_template = self.template_env.get_template(
            "train.Dockerfile.j2"
        )
        self.test_docker_template = self.template_env.get_template("test.Dockerfile.j2")
        self.docker_compose_template = self.template_env.get_template(
            "docker-compose.yaml.j2"
        )
        self.test_script_template = self.template_env.get_template("test.sh.j2")
        self.task_meta = cfg.task_config

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Run task export across provided scopes

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_generation/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            self.extract_scope(scope, output_path)

    def extract_scope(self, scope: Scope, output_path: Path) -> None:

        with Session(self.db) as session:
            stmt = select(Task).where(Task.status == TaskStatus.NEW)
            tasks: list[Task] = session.execute(stmt).scalars().all()

        logger.info(f"Exporting {len(tasks)} tasks for scope {scope.name}")

        for task in tasks:
            try:
                task_dir = self._write_task(task)
                logger.info(f"Exported task {task.name} to {task_dir}")
                set_task_status(self.db, task.task_id, TaskStatus.EXPORTED)

            except Exception as e:  # noqa: BLE001 - isolate failure to this pipeline item
                logger.error(f"Export of task {task.name} failed - {e}")
                set_task_status(self.db, task.task_id, TaskStatus.EXPORT_FAILED)

    def _write_task(self, task: Task) -> Path:
        task_dir = self.export_path / slugify(task.name)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_config = self.task_meta_template.render(
            org_name=self.task_meta.org_name,
            task_name=task_dir.name,
            description=task.short_description,
            keywords=[],
            difficulty_explanation=task.difficulty_explanation,
        )
        (task_dir / "task.toml").write_text(task_config)
        (task_dir / "instruction.md").write_text(task.instructions)

        environment = task_dir / "environment"
        environment.mkdir()
        compose = self.docker_compose_template.render()
        self._write_environment(
            environment,
            self.train_docker_template,
            task.train_script,
            task.requirements,
            compose,
        )

        tests = task_dir / "tests"
        tests.mkdir()
        self._write_environment(
            tests,
            self.test_docker_template,
            task.test_script,
            task.requirements,
            compose,
        )
        (tests / "verifier.py").write_text(task.assessment_script)
        (tests / "test.sh").write_text(
            self.test_script_template.render(requirements=task.requirements)
        )
        return task_dir

    @staticmethod
    def _write_environment(
        directory, template, data_script, requirements, compose
    ) -> None:
        (directory / "Dockerfile").write_text(
            template.render(requirements=requirements, test=False)
        )
        (directory / "docker-compose.yaml").write_text(compose)
        (directory / "data.py").write_text(data_script)
