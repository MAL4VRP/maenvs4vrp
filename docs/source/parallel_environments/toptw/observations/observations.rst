.. _TOPTW_PAR-observations:

===============
Observations
===============

TOPTW parallel environment observations.

Observations settings are defined in file ``observations.py``.

Observations
------------

.. autoclass:: maenvs4vrp.parallel_environments.toptw.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_tw_low

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_tw_high

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_profits

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_service_time

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_tw_high_minus_tw_low_div_max_dur

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_is_depot

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_reachable_frac_agents

Other agents features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_x_coordinate

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_y_coordinate

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_frac_current_time

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_frac_current_profit

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_other_agents_frac_feasible_nodes

Global features
^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_global_frac_profits

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_global_frac_colect_profits

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_feat_global_frac_done_agents

Computing features
^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_static_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_edges_static_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_dynamic_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_agent_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_other_agents_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_all_agents_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.compute_global_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations.get_observations

Internal methods
^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations._concat_features

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations._normalize_feature

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations._min_max_normalization

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations._min_max_normalization2d

.. automethod:: maenvs4vrp.parallel_environments.toptw.observations.Observations._standardize
