import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

from hydra.utils import instantiate
from jinja2 import Environment, FileSystemLoader
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.mcp import MCPAdapter
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from finpipe.app.settings import Scope
from finpipe.extraction.specifications import (
    TaskSpecification,
    build_task_specification,
)
from finpipe.generation.prompts import (
    Assessment,
    DatasetRetrieval,
    build_dataset_download_prompt,
    build_description_summary_prompt,
    build_difficulty_explanation_prompt,
    build_metric_prompt,
)
from finpipe.generation.settings import TaskGenerationConfig
from finpipe.generation.validation import (
    run_pylint,
    test_test_download_script,
    test_train_download_script,
)
from finpipe.integrations.tools import get_tools, init_mcp_servers
from finpipe.resources import template_directory
from finpipe.storage.models import Task, TaskCandidate, TaskCandidateStatus, TaskStatus
from finpipe.storage.status import set_task_candidate_status

logger = logging.getLogger(__name__)


class TaskGenerator:
    def __init__(self, db: Engine, cfg: TaskGenerationConfig) -> None:
        self.db = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
        self.template_env = Environment(
            loader=FileSystemLoader(template_directory(cfg.templates_path))
        )

        def strformat(v, fmt):
            return fmt % v

        self.template_env.filters["strformat"] = strformat

        self.instructions_template = self.template_env.get_template(
            "instructions.md.j2"
        )
        self.train_docker_template = self.template_env.get_template(
            "train.Dockerfile.j2"
        )
        self.test_docker_template = self.template_env.get_template("test.Dockerfile.j2")
        self.test_sh_template = self.template_env.get_template("test.sh.j2")
        assert isinstance(self.llm, BaseChatModel)

    def get_code_check_tools(
        self,
        expected_train_files: list[str],
        expected_test_files: list[str],
    ) -> list[Callable]:

        def run_pylint_checks(code: str) -> tuple[bool, str]:
            """
            Run PyLint checks on Python code to detect code errors

            Args:
                code: Python code string

            Returns:
            Tuple containing boolean flag indicating success/failure of
            checks, and string containing any error messages
            """

            with NamedTemporaryFile(suffix=".py") as f:
                with open(f.name, "w") as wf:
                    wf.write(code)
                return run_pylint(Path(f.name))

        def check_training_data_script(
            download_script: str,
            requirements: list[str],
        ) -> tuple[bool, str]:
            """
            Run training data download script and check it generates
            the expected files

            Args:
                download_script: Python train/test data download script
                requirements: List of Python requirements used by the script

            Returns:
            Tuple containing success flag and run logs
            """
            return test_train_download_script(
                self.train_docker_template,
                download_script,
                requirements,
                expected_train_files,
            )

        def check_test_data_script(
            download_script: str,
            verification_script: str,
            requirements: list[str],
        ) -> tuple[bool, str]:
            """
            Run testing data download script and check it generates
            the expected files

            Args:
                download_script: Python test data download script
                verification_script: Python verifier script
                requirements: List of Python requirements used by the script

            Returns:
            Tuple containing success flag and run logs
            """
            return test_test_download_script(
                self.test_docker_template,
                self.test_sh_template,
                download_script,
                verification_script,
                requirements,
                expected_test_files,
            )

        return [run_pylint_checks, check_training_data_script, check_test_data_script]

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Run task generation across provided scopes

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_generation/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            asyncio.run(self.extract_scope(scope, output_path))

    async def extract_scope(self, scope: Scope, output_path: Path) -> None:

        scripts_path = output_path / "scripts/"
        scripts_path.mkdir(exist_ok=True)

        with Session(self.db) as session:
            stmt = select(TaskCandidate).where(
                TaskCandidate.scope_id == scope.id,
                TaskCandidate.status == TaskCandidateStatus.APPROVED,
            )
            tasks: list[TaskCandidate] = session.execute(stmt).scalars().all()
            task_specs = [build_task_specification(task) for task in tasks]

        logger.info(f"Generating {len(task_specs)} tasks for scope {scope.name}")

        async with MCPAdapter(init_mcp_servers()) as mcp_adapter:
            mcp_tools = await get_tools(mcp_adapter)

            for task_spec in task_specs:
                expected_train_files = (
                    [x.filename for x in task_spec.training_inputs]
                    + [x.filename for x in task_spec.training_targets]
                    + [x.filename for x in task_spec.test_inputs]
                )
                expected_test_files = [x.filename for x in task_spec.test_targets]

                code_tools = [
                    tool(t)
                    for t in self.get_code_check_tools(
                        expected_train_files,
                        expected_test_files,
                    )
                ]
                dataset_agent = create_agent(
                    model=self.llm,
                    response_format=ToolStrategy(DatasetRetrieval, handle_errors=False),
                    tools=mcp_tools + code_tools,
                )

                dataset_prompt = build_dataset_download_prompt(task_spec)
                metric_prompt = build_metric_prompt(task_spec, scope.task_params)
                description_prompt = build_description_summary_prompt(
                    task_spec.task_instructions
                )
                difficulty_prompt = build_difficulty_explanation_prompt(
                    task_spec.task_instructions
                )

                try:
                    new_task = await self._generate_task(
                        task_spec,
                        dataset_agent,
                        dataset_prompt=dataset_prompt,
                        metric_prompt=metric_prompt,
                        description_prompt=description_prompt,
                        difficulty_prompt=difficulty_prompt,
                        scripts_path=scripts_path,
                        expected_train_files=expected_train_files,
                        expected_test_files=expected_test_files,
                    )
                    with Session(self.db) as session:
                        session.add(new_task)
                        session.commit()
                    set_task_candidate_status(
                        self.db, task_spec.id, TaskCandidateStatus.PROCESSED
                    )
                except Exception as e:  # noqa: BLE001 - isolate failure to this pipeline item
                    set_task_candidate_status(
                        self.db, task_spec.id, TaskCandidateStatus.FAILED
                    )
                    logger.error(
                        f"Task generation failed for task {task_spec.task_name} - {e}"
                    )

    async def _generate_task(
        self,
        spec: TaskSpecification,
        dataset_agent,
        *,
        dataset_prompt: str,
        metric_prompt: str,
        description_prompt: str,
        difficulty_prompt: str,
        scripts_path: Path,
        expected_train_files: list[str],
        expected_test_files: list[str],
    ) -> Task:
        response = await dataset_agent.ainvoke(
            {"messages": [{"role": "user", "content": dataset_prompt}]}
        )
        scripts = response["structured_response"]
        assessment: Assessment = self.llm.with_structured_output(Assessment).invoke(
            metric_prompt
        )
        description = self.llm.invoke(description_prompt).content
        difficulty = self.llm.invoke(difficulty_prompt).content
        requirements = list(
            set(scripts.requirements).union(set(assessment.requirements))
        )
        instructions = self.instructions_template.render(
            task_description=spec.task_instructions,
            training_inputs=spec.training_inputs,
            training_targets=spec.training_targets,
            test_inputs=spec.test_inputs,
            test_outputs=spec.test_outputs,
            metrics=spec.metrics,
        )
        task_path = scripts_path / spec.task_name
        task_path.mkdir(exist_ok=True)
        self._write_checked_script(
            task_path / "train.py", scripts.training_script, "train data"
        )
        self._write_checked_script(
            task_path / "test.py", scripts.testing_script, "test data"
        )
        self._write_checked_script(
            task_path / "verifier.py", assessment.assessment_script, "verifier"
        )
        (task_path / "requirements.txt").write_text("\n".join(requirements))
        (task_path / "instruction.md").write_text(instructions)
        self._validate_downloads(
            scripts, assessment, requirements, expected_train_files, expected_test_files
        )
        return Task(
            task_candidate_id=spec.id,
            name=spec.task_name,
            status=TaskStatus.NEW,
            train_script=scripts.training_script,
            test_script=scripts.testing_script,
            assessment_script=assessment.assessment_script,
            requirements=requirements,
            instructions=instructions,
            short_description=description,
            difficulty_explanation=difficulty,
        )

    @staticmethod
    def _write_checked_script(path: Path, code: str, label: str) -> None:
        path.write_text(code)
        passed, messages = run_pylint(path)
        if not passed:
            raise RuntimeError(
                f"PyLint detected the following errors in the {label} script \n\n{messages}"
            )

    def _validate_downloads(
        self,
        scripts: DatasetRetrieval,
        assessment: Assessment,
        requirements: list[str],
        train_files: list[str],
        test_files: list[str],
    ) -> None:
        passed, message = test_train_download_script(
            self.train_docker_template,
            scripts.training_script,
            requirements,
            train_files,
        )
        if not passed:
            raise RuntimeError(f"Train DockerFile build failed - {message}")
        passed, message = test_test_download_script(
            self.test_docker_template,
            self.test_sh_template,
            scripts.testing_script,
            assessment.assessment_script,
            requirements,
            test_files,
        )
        if not passed:
            raise RuntimeError(f"Test DockerFile build failed - {message}")
