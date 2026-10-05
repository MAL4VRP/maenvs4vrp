.. _PDPTW-generation-instance-generation:

======================
Instance Generation 
======================

Instance generation adapts the paper [Li21]_ to pickup and delivery pairs:

#. The depot and services’ $(x, y)$ locations are sampled uniformly from $[0, 1]^2$;
#. Each service’s demand $d_i$ is sampled uniformly from ${1, 2, ..., 10}$, and each vehicle has capacity $C = 50$;
#. For the time window constraint, we set the time window for the depot as $[b_0, e_0] = [0, 3]$, and the service time at each $i$ to be $s_i = 0.2$. For a pickup $i$ with delivery $j$ we set the time windows by: (a) sampling the time window center $c_i ∼ U([b_0 + t_{0,i}, e_0 − t_{0,i} − t_{i,j} − t_{j,0} − 2 s_i])$, where $t_{i,j}$ is the travel time, equaling the Euclidean distance, between nodes $i$ and $j$, so that a vehicle serving the pair alone can return to the depot in time (the upper bound is raised to the lower bound when the pair is too far from the depot to be served within $[b_0, e_0]$); (b) sampling the time window half-width $h_i$ uniformly at random from $[s_i, e_0/3] = [0.2, 1]$, so that the delivery window is still open after the pickup service; (c) setting the time window for $i$ as $[max(b_0, c_i − h_i), min(e_0, c_i + h_i)]$ and for $j$ as $[max(b_0, c_i + t_{i,j} − h_i), min(e_0, c_i + t_{i,j} + h_i)]$.

Instances generation settings are defined in file ``instances_generator.py``.

InstanceGenerator 
--------------------

.. autoclass:: maenvs4vrp.environments.pdptw.instances_generator.InstanceGenerator
    :members:
    :special-members: __init__
