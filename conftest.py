import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dir_path",
        action="store",
        default=None,
        help="Directory containing harmonized tables to validate",
    )


@pytest.fixture
def dir_path(request):
    path = request.config.getoption("--dir_path")
    assert path is not None, "Please provide --dir_path"
    assert os.path.isdir(path), f"Directory does not exist: {path}"
    return path
