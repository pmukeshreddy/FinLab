from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import Engine

from finpipe.generation.service import TaskGenerator
from finpipe.generation.settings import TaskGenerationConfig
from finpipe.generation.validation import run_pylint


@pytest.mark.parametrize(
    "code, expected", [("print('Hello world')", True), ("x 10", False)]
)
def test_pylint_checks(code: str, expected: bool) -> None:

    with NamedTemporaryFile() as tmpfile:
        with open(tmpfile.name, "w") as f:
            f.write(code)

        path = Path(tmpfile.name)
        result, _messages = run_pylint(path)

    assert expected == result


@pytest.fixture
def llm_config() -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": "langchain_core.language_models.fake_chat_models.FakeChatModel",
            "responses": [],
        }
    )


def test_tools_code_check_tools_pylint(db: Engine, llm_config: DictConfig) -> None:

    config = TaskGenerationConfig(
        llm_config, templates_path="src/finpipe/resources/templates"
    )
    generator = TaskGenerator(db, config)

    run_pylint_checks, _, _ = generator.get_code_check_tools(["foo.txt"], ["bar.txt"])

    good_code_string = """def foo():\n\treturn 10"""
    result = run_pylint_checks(good_code_string)

    assert result[0]

    bad_code_string = "x 10"
    result = run_pylint_checks(bad_code_string)

    assert not result[0]


@pytest.mark.parametrize(
    "script, expected",
    [
        ("with open('foo.txt', 'w') as f:\n\tf.write('Hello World')", True),
        ("with open('bar.txt', 'w') as f:\n\tf.write('Hello World')", False),
        ("x 10", False),
    ],
)
@pytest.mark.usefixtures("docker_daemon")
def test_tools_code_check_tools_train_data(
    db: Engine, llm_config: DictConfig, script: str, expected: bool
) -> None:

    config = TaskGenerationConfig(
        llm_config, templates_path="src/finpipe/resources/templates"
    )
    generator = TaskGenerator(db, config)

    _, check_training_data_script, _ = generator.get_code_check_tools(
        ["foo.txt"], ["bar.txt"]
    )

    result = check_training_data_script(script, [])
    assert result[0] == expected


@pytest.mark.parametrize("stage", ["train", "test"])
@pytest.mark.parametrize("outcome", ["success", "missing", "build_error", "run_error"])
def test_download_validation_artifacts_and_failures(monkeypatch, stage, outcome):
    """Exercise Docker boundaries offline, including the exact staged build files."""
    from pathlib import Path
    from types import SimpleNamespace

    from docker.errors import BuildError
    from jinja2 import Template

    from finpipe.generation import validation

    captured = {}

    def build(**kwargs):
        folder = Path(kwargs["path"])
        captured.update({p.name: p.read_text() for p in folder.iterdir()})
        assert kwargs["buildargs"].keys() == {
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
            "KAGGLE_API_TOKEN",
        }
        if outcome == "build_error":
            raise BuildError("bad build", [{"error": "details"}])
        return "image", []

    def run(image, command, remove):
        assert (image, command, remove) == ("image", "ls -a", True)
        if outcome == "run_error":
            raise RuntimeError("container failed")
        return b".\n..\ndata.csv\n" if outcome == "success" else b".\n"

    monkeypatch.setattr(
        validation.docker,
        "from_env",
        lambda: SimpleNamespace(
            images=SimpleNamespace(build=build), containers=SimpleNamespace(run=run)
        ),
    )
    template = Template("requirements={{ requirements | join(',') }} test={{ test }}")
    if stage == "train":
        passed, message = validation.test_train_download_script(
            template, "data code", ["numpy"], ["data.csv"]
        )
        assert captured == {
            "data.py": "data code",
            "Dockerfile": "requirements=numpy test=True",
        }
    else:
        passed, message = validation.test_test_download_script(
            template,
            Template("{{ requirements | join(',') }}"),
            "data code",
            "verifier code",
            ["numpy"],
            ["data.csv"],
        )
        assert captured == {
            "data.py": "data code",
            "verifier.py": "verifier code",
            "Dockerfile": "requirements=numpy test=True",
            "test.sh": "numpy",
        }
    assert passed is (outcome == "success")
    expected = {
        "success": f"{stage.capitalize()} data retrieval script successful",
        "missing": f"Expected {stage} data file data.csv not found in test docker image",
        "build_error": "Error message:bad build\nBuild Logs: {'error': 'details'}",
        "run_error": "container failed",
    }
    assert message == expected[outcome]
