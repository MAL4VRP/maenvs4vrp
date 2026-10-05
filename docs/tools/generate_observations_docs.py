"""
Regenerate the ``observations.rst`` page of every environment from its ``Observations`` class.

Each page gets the same structure, and every feature listed in the class ``POSSIBLE_*_FEATURES``
lists is documented with the method that computes it. Features without a matching method are
reported, since they would fail at runtime.

Usage (from the repository root)::

    python docs/tools/generate_observations_docs.py
"""
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "source"
sys.path.insert(0, str(ROOT))

GROUPS = {"environments": "", "parallel_environments": " parallel environment"}

# (section title, POSSIBLE_* list, method prefix)
SECTIONS = [
    ("Nodes static features", "POSSIBLE_NODES_STATIC_FEATURES", "get_feat_"),
    ("Edges static features", "POSSIBLE_EDGES_STATIC_FEATURES", "get_edges_feat_"),
    ("Nodes dynamic features", "POSSIBLE_NODES_DYNAMIC_FEATURES", "get_feat_"),
    ("Current agent features", "POSSIBLE_AGENT_FEATURES", "get_feat_agent_"),
    ("Other agents features", "POSSIBLE_OTHER_AGENTS_FEATURES", "get_feat_other_agents_"),
    ("All agents features", "POSSIBLE_ALL_AGENTS_FEATURES", "get_feat_all_agents_"),
    ("Global features", "POSSIBLE_GLOBAL_FEATURES", "get_feat_global_"),
]
COMPUTE = ["compute_static_features", "compute_edges_static_features", "compute_dynamic_features",
           "compute_agent_features", "compute_other_agents_features", "compute_all_agents_features",
           "compute_global_features", "get_observations"]
INTERNAL = ["_concat_features", "_normalize_feature", "_min_max_normalization",
            "_min_max_normalization2d", "_standardize"]


def heading(text, char):
    return f"{text}\n{char * len(text)}\n"


def page(group, env, cls, label):
    qual = f"maenvs4vrp.{group}.{env}.observations.Observations"
    lines = [f".. _{label}:\n" if label else "",
             "=" * 15, "Observations", "=" * 15, "",
             f"{env.upper()}{GROUPS[group]} observations.", "",
             "Observations settings are defined in file ``observations.py``.", "",
             heading("Observations", "-"),
             f".. autoclass:: {qual}",
             "    :members: __init__, set_env", ""]
    missing = []
    for title, attr, prefix in SECTIONS:
        feats = getattr(cls, attr, [])
        if not feats:
            continue
        lines.append(heading(title, "^"))
        for f in dict.fromkeys(feats):  # keep order, drop duplicates
            method = f"{prefix}{f}"
            if hasattr(cls, method):
                lines += [f".. automethod:: {qual}.{method}", ""]
            else:
                missing.append(f"{attr}: '{f}' has no method {method}()")
    lines.append(heading("Computing features", "^"))
    lines += sum(([f".. automethod:: {qual}.{m}", ""] for m in COMPUTE if hasattr(cls, m)), [])
    lines.append(heading("Internal methods", "^"))
    lines += sum(([f".. automethod:: {qual}.{m}", ""] for m in INTERNAL if hasattr(cls, m)), [])
    return "\n".join(lines).rstrip() + "\n", missing


def main():
    problems = 0
    for group in GROUPS:
        for env_dir in sorted((ROOT / "maenvs4vrp" / group).iterdir()):
            if not (env_dir / "observations.py").exists():
                continue
            env = env_dir.name
            rst = DOCS / group / env / "observations" / "observations.rst"
            label = None
            if rst.exists():
                first = rst.read_text().splitlines()[0].strip()
                if first.startswith(".. _") and first.endswith(":"):
                    label = first[4:-1]
            cls = importlib.import_module(f"maenvs4vrp.{group}.{env}.observations").Observations
            text, missing = page(group, env, cls, label)
            rst.parent.mkdir(parents=True, exist_ok=True)
            rst.write_text(text)
            for m in missing:
                problems += 1
                print(f"[{group}/{env}] {m}")
            print(f"wrote {rst.relative_to(ROOT)}")
    if problems:
        print(f"{problems} declared features without a method")


if __name__ == "__main__":
    main()
