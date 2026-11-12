import pytest
from sqlalchemy import Engine, create_engine

from finpipe.storage.models import Base


def pytest_addoption(parser):
    parser.addoption(
        "--require-docker",
        action="store_true",
        help="Fail instead of skipping Docker integration tests when Docker is unavailable.",
    )


@pytest.fixture
def docker_daemon(request):
    import docker

    client = None
    try:
        client = docker.from_env()
        client.ping()
    except docker.errors.DockerException as error:
        if request.config.getoption("--require-docker"):
            pytest.fail(f"Docker integration tests require a running daemon: {error}")
        pytest.skip("Docker integration test requires a running daemon")
    finally:
        if client is not None:
            client.close()


@pytest.fixture(scope="function")
def db() -> Engine:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine
