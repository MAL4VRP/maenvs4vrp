"""Docstrings follow the Google-style convention described in AGENTS.md."""
import ast
import re

import pytest

from tests.helpers import PKG_DIR, REPO_DIR

FILES = sorted(p for p in PKG_DIR.rglob("*.py") if "learning_notebooks" not in p.parts)
ENTRY_RE = re.compile(r"^(\*{0,2}\w+)(?: \(([^)]*(?:\([^)]*\))?[^)]*)\))?: \S")
BAD_HEADER_RE = re.compile(r"^\s*(Args|Returns|Raises|Yields)\s*$", re.M)


def args_section(doc: str) -> list[str] | None:
    """Return the top-level entries of the Args section, or None when there is no Args section."""
    m = re.search(r"^Args:\n(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    if not m:
        return None
    return [line[4:] for line in m.group(1).splitlines() if line.startswith("    ") and not line.startswith("     ")]


def check_function(fn, qualname):
    problems = []
    public = not fn.name.startswith("_") or fn.name == "__init__"
    doc = ast.get_docstring(fn)
    positional = [a.arg for a in fn.args.posonlyargs + fn.args.args if a.arg not in ("self", "cls")]
    params = positional + [a.arg for a in fn.args.kwonlyargs]
    if fn.args.vararg:
        params.insert(len(positional), "*" + fn.args.vararg.arg)
    if fn.args.kwarg:
        params.append("**" + fn.args.kwarg.arg)
    if doc is None:
        return [f"{qualname}: missing docstring"] if public else []
    if BAD_HEADER_RE.search(doc):
        problems.append(f"{qualname}: section header without ':'")
    if re.search(r"\bN/a\b|^\s*n/a\.?\s*$", doc, re.M):
        problems.append(f"{qualname}: uses N/a")
    if fn.name == "__init__" and re.search(r"^Returns:", doc, re.M):
        problems.append(f"{qualname}: __init__ has a Returns section")
    entries = args_section(doc)
    if entries is None:
        if params and public:
            problems.append(f"{qualname}: parameters {params} are not documented")
        return problems
    names = []
    for entry in entries:
        m = ENTRY_RE.match(entry)
        if not m:
            problems.append(f"{qualname}: malformed Args entry {entry!r}")
            continue
        names.append(m.group(1))
    if [n.lstrip("*") for n in names] != [p.lstrip("*") for p in params]:
        problems.append(f"{qualname}: Args {names} do not match the signature {params}")
    return problems


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(PKG_DIR)) for p in FILES])
def test_docstrings(path):
    tree = ast.parse(path.read_text())
    problems = [] if ast.get_docstring(tree) else ["module: missing docstring"]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            problems += check_function(node, node.name)
        elif isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                problems.append(f"{node.name}: missing class docstring")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    problems += check_function(item, f"{node.name}.{item.name}")
    assert not problems, f"{path.relative_to(REPO_DIR)}:\n  " + "\n  ".join(problems)


BENCHMARK_FILES = sorted(PKG_DIR.glob("*/*/benchmark_instances_generator.py"))


def accepted_set_names(cls: ast.ClassDef) -> set[str]:
    """Set names accepted by a benchmark generator: its `assert instance_* in [...]`, else the keys it returns."""
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    for node in ast.walk(methods["__init__"]):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id in ("instance_name", "instance_type")
                and isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], ast.List)):
            return {e.value for e in node.comparators[0].elts}
    return {k.value for node in ast.walk(methods["get_list_of_instances"])
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            for k in node.value.keys if isinstance(k, ast.Constant)}


@pytest.mark.parametrize("path", BENCHMARK_FILES, ids=[p.parent.parent.name[:3] + "-" + p.parent.name for p in BENCHMARK_FILES])
def test_benchmark_set_names_documented(path):
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "BenchmarkInstanceGenerator")
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    accepted = accepted_set_names(cls)
    assert accepted
    keys = re.search(r"Keys (.*?);", ast.get_docstring(methods["get_list_of_instances"]))
    assert keys and set(re.findall(r"'([^']+)'", keys.group(1))) == accepted
    init_doc = ast.get_docstring(methods["__init__"])
    documented = set()
    for line in re.findall(r"^\s+instance_(?:name|type) \(str, optional\): (.*)$", init_doc, re.M):
        documented |= set(re.findall(r'"([^"]+)"', line.split("Defaults to")[0]))
    assert documented == accepted
