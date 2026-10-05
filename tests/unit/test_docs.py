"""Sphinx sources are consistent with the code (a full build is run with ``sphinx -W`` in CI)."""
import importlib
import importlib.util
import re

import pytest

from tests.helpers import REPO_DIR

DOCS = REPO_DIR / "docs" / "source"
DIRECTIVE = re.compile(r"^\s*\.\. (automodule|autoclass|autofunction|automethod|currentmodule|py:currentmodule)::\s*(\S+)")


def resolve(dotted):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part)
        return obj
    raise ModuleNotFoundError(dotted)


def autodoc_targets():
    for rst in sorted(DOCS.rglob("*.rst")):
        if "notebooks" in rst.parts:
            continue
        module, klass = None, None
        for line in rst.read_text().splitlines():
            m = DIRECTIVE.match(line)
            if not m:
                continue
            kind, target = m.groups()
            if kind in ("automodule", "currentmodule", "py:currentmodule"):
                module = klass = target
                yield rst, target
                continue
            if not target.startswith("maenvs4vrp"):
                target = f"{klass if kind == 'automethod' and '.' not in target else module}.{target}"
            if kind == "autoclass":
                klass = target
            yield rst, target


@pytest.mark.parametrize("rst, target", [pytest.param(r, t, id=f"{r.relative_to(DOCS)}:{t.split('.')[-1]}")
                                          for r, t in autodoc_targets()])
def test_autodoc_target_exists(rst, target):
    resolve(target)


def test_observation_pages_are_up_to_date():
    spec = importlib.util.spec_from_file_location("gen_obs_docs", REPO_DIR / "docs" / "tools" / "generate_observations_docs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    stale = []
    for group in gen.GROUPS:
        for env_dir in sorted((REPO_DIR / "maenvs4vrp" / group).iterdir()):
            if not (env_dir / "observations.py").exists():
                continue
            rst = DOCS / group / env_dir.name / "observations" / "observations.rst"
            label = rst.read_text().splitlines()[0].strip()[4:-1] if rst.exists() else None
            cls = importlib.import_module(f"maenvs4vrp.{group}.{env_dir.name}.observations").Observations
            text, missing = gen.page(group, env_dir.name, cls, label)
            assert not missing, missing
            if not rst.exists() or rst.read_text() != text:
                stale.append(str(rst.relative_to(REPO_DIR)))
    assert not stale, f"run `python docs/tools/generate_observations_docs.py`; stale pages: {stale}"


def test_citations_are_defined_once():
    references = (DOCS / "content" / "references.rst").read_text()
    defined = re.findall(r"^\.\. \[(\w+)\]", references, re.M)
    assert len(defined) == len(set(defined)), "duplicate bibliography keys"
    used = set()
    for rst in DOCS.rglob("*.rst"):
        text = rst.read_text()
        if rst.name != "references.rst":
            assert not re.search(r"^\.\. \[\w+\]", text, re.M), f"{rst}: define citations in content/references.rst"
        used |= set(re.findall(r"\[(\w+)\]_", text))
    assert used <= set(defined), f"undefined citations: {sorted(used - set(defined))}"


def test_tutorial_notebooks_exist():
    index = (DOCS / "index.rst").read_text()
    for name in re.findall(r"^\s+notebooks/(\S+)$", index, re.M):
        assert (REPO_DIR / "maenvs4vrp" / "learning_notebooks" / f"{name}.ipynb").exists(), name
