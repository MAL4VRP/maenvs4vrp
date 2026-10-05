.. _PCVRP-observations:

===============
Observations
===============

PCVRP observations.

Observations settings are defined in file ``observations.py``.

Observations
------------

.. autoclass:: maenvs4vrp.environments.pcvrp.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_demand

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_service_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_is_depot

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_profits

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_arrive2node_div_end_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_time2end_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_fract_time_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_reachable_frac_agents

Current agent features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_x_coordinate

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_y_coordinate

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_frac_current_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_frac_current_load

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_arrivedepot_div_end_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_agent_frac_feasible_nodes

Other agents features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_frac_current_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_frac_current_load

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_dist2depot_div_end_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_dist2agent_div_end_time

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_frac_feasible_nodes

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_time_delta2agent_div_max_dur

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_other_agents_was_last

Global features
^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_global_frac_profits

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_global_frac_demands

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_global_frac_fleet_load_capacity

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_feat_global_frac_done_agents

Computing features
^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_static_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_edges_static_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_dynamic_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_agent_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_other_agents_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_all_agents_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.compute_global_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations.get_observations

Internal methods
^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations._concat_features

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations._normalize_feature

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations._min_max_normalization

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations._min_max_normalization2d

.. automethod:: maenvs4vrp.environments.pcvrp.observations.Observations._standardize
