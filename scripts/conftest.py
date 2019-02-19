import pytest
import tempfile


@pytest.fixture(scope="function")
def file_factory():
    def make(mode="w+b"):
        return tempfile.NamedTemporaryFile(mode)

    return make
