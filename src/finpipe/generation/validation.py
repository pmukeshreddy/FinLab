import json
import os
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import docker
from docker.errors import BuildError
from jinja2 import Template
from pylint.lint import Run
from pylint.reporters.json_reporter import JSON2Reporter


def run_pylint(script_path: Path) -> tuple[bool, str]:
    """
    Run pylint and report any code errors or fatal issues

    Args:
        script_path: Path to Python script/module

    Returns:
    Tuple containing flag indicating failure, and list of any error messages
    """
    pylint_output = StringIO()
    reporter = JSON2Reporter(pylint_output)
    Run(
        [str(script_path), "--disable=imports", "--disable=maybe-no-member"],
        reporter=reporter,
        exit=False,
    )
    output = pylint_output.getvalue()
    results = json.loads(output)
    counts = results["statistics"]["messageTypeCount"]

    if counts["fatal"] == 0 and counts["error"] == 0:
        return True, []

    error_messages = [
        m["message"] for m in results["messages"] if m["type"] in {"fatal", "error"}
    ]
    error_messages = "\n".join([f"- {x}" for x in error_messages])

    return False, error_messages


def test_train_download_script(
    docker_template: Template,
    download_script: str,
    requirements: list[str],
    expected_files: list[str],
) -> tuple[bool, str]:
    """
    Build train docker image and check expected data files are created

    Args:
        docker_template: Train docker image template
        download_script: Train data download Python script
        requirements: List of python requirements
        expected_files: list of expected data files

    Returns:
    Tuple containing success flag, and reasoning string
    """

    client = docker.from_env()
    files = {
        "data.py": download_script,
        "Dockerfile": docker_template.render(requirements=requirements, test=True),
    }
    return _validate_image(client, files, expected_files, "train")


def test_test_download_script(
    docker_template: Template,
    test_template: Template,
    data_download_script: str,
    verification_script: str,
    requirements: list[str],
    expected_files: list[str],
) -> tuple[bool, str]:
    """
    Build test docker image and check expected data files are created

    Args:
        docker_template: Test docker image template
        test_template: tesh.sh bash script template
        data_download_script: Test data download Python script
        verification_script: Python results verification script
        requirements: List of python requirements
        expected_files: list of expected data files

    Returns:
    Tuple containing success flag, and reasoning string
    """

    client = docker.from_env()
    files = {
        "data.py": data_download_script,
        "verifier.py": verification_script,
        "Dockerfile": docker_template.render(requirements=requirements, test=True),
        "test.sh": test_template.render(requirements=requirements),
    }
    return _validate_image(client, files, expected_files, "test")


def _validate_image(
    client, files: dict[str, str], expected_files: list[str], stage: str
) -> tuple[bool, str]:
    with TemporaryDirectory() as folder:
        folder_path = Path(folder)
        for filename, content in files.items():
            (folder_path / filename).write_text(content)
        try:
            image, _ = client.images.build(
                path=str(folder_path),
                buildargs={
                    name: os.environ.get(name)
                    for name in (
                        "ALPACA_API_KEY",
                        "ALPACA_SECRET_KEY",
                        "KAGGLE_API_TOKEN",
                    )
                },
            )
        except BuildError as error:
            logs = "\n".join(str(item) for item in error.build_log)
            return False, f"Error message:{error.msg}\nBuild Logs: {logs}"
    try:
        output = client.containers.run(image, command="ls -a", remove=True)
    except Exception as error:  # noqa: BLE001 - preserve container failure diagnostics
        return False, str(error)
    actual_files = set(output.decode("utf-8").split("\n"))
    for filename in expected_files:
        if filename not in actual_files:
            return (
                False,
                f"Expected {stage} data file {filename} not found in test docker image",
            )
    return True, f"{stage.capitalize()} data retrieval script successful"
