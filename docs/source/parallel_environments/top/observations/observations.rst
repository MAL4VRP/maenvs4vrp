.. _TOP_PAR-observations:

===============
Observations
===============

TOP parallel environment observations.

Observations settings are defined in file ``observations.py``.

Observations
------------

.. autoclass:: maenvs4vrp.parallel_environments.top.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_profits

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_service_time

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_is_depot

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_reachable_frac_agents

Other agents features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_x_coordinate

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_y_coordinate

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_frac_current_time

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_frac_current_profit

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_frac_feasible_nodes

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_other_agents_frac_time_left

All agents features
^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_all_agents_x_coordinate

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_all_agents_y_coordinate

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_all_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_all_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_all_agents_frac_current_time

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_all_agents_frac_current_profit

Global features
^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_global_frac_profits

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_global_frac_colect_profits

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_feat_global_frac_done_agents

Computing features
^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_static_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_edges_static_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_dynamic_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_agent_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_other_agents_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_all_agents_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.compute_global_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations.get_observations

Internal methods
^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations._concat_features

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations._normalize_feature

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations._min_max_normalization

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations._min_max_normalization2d

.. automethod:: maenvs4vrp.parallel_environments.top.observations.Observations._standardize
