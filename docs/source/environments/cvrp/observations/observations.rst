.. _CVRP-observations:

===============
Observations
===============

CVRP observations.

Observations settings are defined in file ``observations.py``.

Observations
------------

.. autoclass:: maenvs4vrp.environments.cvrp.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_demand

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_is_depot

Edges static features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_edges_feat_distance_matrix

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_reachable_frac_agents

Current agent features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_remaining_capacity

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_frac_feasible_nodes

Other agents features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_remaining_capacity

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_was_last

All agents features
^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_remaining_capacity

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_cur_time

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_was_last

Global features
^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_global_frac_fleet_load_capacity

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_global_frac_done_agents

Computing features
^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_static_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_edges_static_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_dynamic_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_agent_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_other_agents_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_all_agents_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.compute_global_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_observations

Internal methods
^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations._concat_features

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations._normalize_feature

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations._min_max_normalization

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations._min_max_normalization2d

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations._standardize
