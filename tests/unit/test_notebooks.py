"""Execute the tutorial notebooks (slow; run with ``pytest -m notebooks``)."""
import os

import pytest

from tests.helpers import PKG_DIR, REPO_DIR

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK_DIR = PKG_DIR / "learning_notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
# notebook 2 contains coding challenges that are intentionally incomplete
EXERCISE_NOTEBOOKS = {"2.0.0_maenvs4vrp_exploration_and_challenges.ipynb"}
EXERCISE_ERRORS = {"NameError", "KeyError"}


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs(path, monkeypatch):
    # the kernel is a separate process: make it import this repository
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO_DIR), os.environ.get("PYTHONPATH", "")]))
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=1200, allow_errors=True,
                                     resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
    client.execute()
    errors = [(i, o["ename"], o["evalue"][:120]) for i, cell in enumerate(notebook.cells) if cell.cell_type == "code"
              for o in cell.get("outputs", []) if o.get("output_type") == "error"]
    if path.name in EXERCISE_NOTEBOOKS:
        errors = [e for e in errors if e[1] not in EXERCISE_ERRORS]
    assert not errors, errors
