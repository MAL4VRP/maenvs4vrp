=====================
Training Performance
=====================

In reinforcement learning the environment step runs thousands of times per training iteration, so its fixed
cost matters. This page explains how to get the most speed from the environments.

Recommended setup
=================

.. code-block:: python

    env = Environment(
        instance_generator_object=gen,
        obs_builder_object=obs,
        agent_selector_object=sel,
        reward_evaluator=rew,
        device="cuda",          # same device as the policy
        check_actions=False,    # the policy only picks feasible actions
    )

    td = env.reset_agent_select_observe(num_agents=10, num_nodes=51, obs_list=reset_obs)
    while not td["done"].all():
        td["next_action"] = policy(td["observations"])
        td = env.step_agent_select_observe(td, obs_list=step_obs)

* **Device.** ``reset(device=None)`` keeps the device the environment was built with. Passing a device to
  ``reset`` moves the environment to it. Instances are generated on, or moved to, that device.
* **check_actions.** With the default ``True``, every step asserts that the selected agent is active and the
  action is allowed by its mask. On a GPU this makes the CPU wait for the device at every step. Leave it on
  while developing a policy, and turn it off for training when the actions come from the action mask.
* **Static observations.** Request the static node features (``nodes_static``) only at reset. They do not
  change during an episode, so the training scripts use a ``step_obs`` list without them.

No GPU synchronization in the step
==================================

The step avoids operations that make the CPU wait for the GPU: boolean-mask indexing or assignment,
``nonzero``, ``.item()`` and ``.any()``/``.all()`` in Python conditions. It uses ``torch.where``,
``scatter_``/``gather`` and logical operations instead. The CPU can therefore queue the environment work and
keep running ahead of the GPU. In 15 of the 17 sequential environments, a step with ``check_actions=False``
does not synchronize. The two exceptions are DSVRPTW, which draws late customers on the CPU, and PDPTW, which
loops over pending deliveries.

If you extend an environment, keep this property and check with ``tests/unit/test_golden_rollouts.py`` that
episodes are unchanged.

Measured speed
==============

Measured on a GTX 1080 with CVRPTW, batch 512, random feasible actions:

.. list-table::
   :header-rows: 1

   * - Measure
     - v0.3
     - v0.3.1
   * - Step on GPU (50–100 customers)
     - 2.9–3.2 ms
     - 2.1–2.5 ms
   * - GPU synchronizations per step
     - 11
     - 0
   * - Attention model REINFORCE iteration (N=50, B=512)
     - 0.41–0.44 s
     - 0.37–0.41 s

The environment now takes about 30% of a training iteration; the remaining time is spent in the policy. The
full study, with a comparison against RL4CO and the next possible optimizations, is in
`docs/performance.md <https://github.com/MAEnvs4VRP/maenvs4vrp/blob/master/docs/performance.md>`_. Reproduce
the step timings with ``python docs/tools/benchmark_step.py``.
