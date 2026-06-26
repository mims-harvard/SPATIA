import shutil
import sys

import lamindb_setup as ln_setup
import pytest


def pytest_sessionstart():
    ln_setup.init(storage="./testdb", name="test", schema="bionty")


def pytest_sessionfinish(session):
    shutil.rmtree("./testdb")
    ln_setup.delete("test", force=True)


@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    tmpdir = request.getfixturevalue("tmpdir")
    sys.path.insert(0, str(tmpdir))
    with tmpdir.as_cwd():
        yield
