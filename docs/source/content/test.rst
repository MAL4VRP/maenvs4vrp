=====================
Unit Testing
=====================

The library includes a test suite that checks every environment, generator and observation feature,
and that the code, the docstrings and this documentation stay consistent. Run it from the repository root:

.. code-block:: bash

    pip install -e ".[dev]"
    pytest

The default run takes a couple of minutes on a CPU. Two groups of tests are excluded by default:

.. code-block:: bash

    pytest -m slow        # large instances and every multi-task variant with every selector
    pytest -m notebooks   # execute the tutorial notebooks (needs the [notebooks] extra)

Random instances never need network access; benchmark tests only use the benchmark instance sets that are
already available locally and are skipped otherwise.

Solution validity tests
=======================

``tests/unit/environments/check_solution_test.py`` (single-task environments),
``check_solution_mt_test.py`` (multi-task environments, including every variant preset) and
``check_solution_parallel_env_test.py`` (parallel environments) run random episodes with every agent
selection mode and check the final solution with ``check_solution_validity``.

The problem sizes can be set from the command line:

* ``--device`` can be ``cpu`` or ``cuda``.
* ``--batch`` sets the batch size (default 2).
* ``--num_agents`` / ``--agents`` sets the number of agents. Repeat the option to test several values.
* ``--num_nodes`` / ``--nodes`` sets the number of nodes. Repeat the option to test several values.

.. code-block:: bash

    pytest tests/unit/environments/check_solution_test.py --batch 4 --num_agents 20 --num_nodes 51 --num_nodes 101

Environment tests
=================

* ``reset_test.py``: every reset method and every agent selection mode (agent selectors, sequential
  agent-node, simultaneous and sequential node-agent selection) of every AEC environment.
* ``reset_bench_test.py`` and ``seed_bench_test.py``: episodes and seeding on benchmark instances.
* ``seed_test.py``: identical seeds give identical instances and different seeds give different ones.
* ``tests/unit/test_episodes.py``: identical seeds give identical episodes, the dense and sparse rewards agree
  over an episode, and the solution validators reject corrupted solutions.
* ``tests/unit/test_generators.py``: unknown ``sample_type`` values raise ``ValueError``, random generation
  works without network access, and every toy instance can be solved.
* ``tests/unit/test_observations.py``: every feature listed in the ``POSSIBLE_*_FEATURES`` lists works and has
  the expected shape, alone and combined, and the default features are declared.

Consistency tests
=================

* ``tests/unit/test_structure.py``: every environment package has the same files, classes, base classes and
  public API (see ``AGENTS.md``), and its documentation pages.
* ``tests/unit/test_docstrings.py``: docstrings follow the Google-style convention and document exactly the
  parameters of each signature.
* ``tests/unit/test_docs.py``: every ``autodoc`` target exists, the observation pages are up to date
  (regenerate them with ``python docs/tools/generate_observations_docs.py``) and every citation is defined.
