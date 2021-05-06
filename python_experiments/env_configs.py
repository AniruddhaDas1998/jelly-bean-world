import jbw
import gym
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from math import pi

# slightly updated default framework configuration
# this allows the agent to pick up every item when it first goes to it
# the default couldn't pick up bananas and onions initially
def default_config():
    # specify the item types
    items = []
    # items.append(jbw.Item("banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
    items.append(jbw.Item("banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
                       intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
                       interaction_fns=[
                           [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 0.0, -6.0],      # parameters for interaction between item 0 and item 0
                           [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -6.0, -6.0],      # parameters for interaction between item 0 and item 1
                           [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 0 and item 2
                           [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 0 and item 3
                        ]))
    # items.append(jbw.Item("onion",     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 0, 0], False, 0.0,
    items.append(jbw.Item("onion",     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
                       intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.0],
                       interaction_fns=[
                           [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -6.0, -6.0],      # parameters for interaction between item 1 and item 0
                           [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 1 and item 1
                           [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -100.0, -100.0],  # parameters for interaction between item 1 and item 2
                           [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 1 and item 3
                        ]))
    items.append(jbw.Item("jellybean", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
                       intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
                       interaction_fns=[
                           [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 2 and item 0
                           [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -100.0, -100.0],  # parameters for interaction between item 2 and item 1
                           [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 0.0, -6.0],      # parameters for interaction between item 2 and item 2
                           [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 2 and item 3
                        ]))
    items.append(jbw.Item("wall",      [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0, 0, 0, 1], [0, 0, 0, 0], True, 1.0,
                       intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[0.0],
                       interaction_fns=[
                           [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 0
                           [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 1
                           [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 2
                           [jbw.InteractionFunction.CROSS, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0] # parameters for interaction between item 3 and item 3
                        ]))

    # construct the simulator configuration
    return jbw.SimulatorConfig(max_steps_per_movement=1, vision_range=8,
        allowed_movement_directions=[jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED],
        allowed_turn_directions=[jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.ALLOWED],
        no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
        collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
        decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000, seed=1234567890)

def uniform_config():
    # specify the item types
    items = []
    # items.append(jbw.Item("banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
    items.append(jbw.Item("banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
               intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
               interaction_fns=[
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],      # parameters for interaction between item 0 and item 0
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],      # parameters for interaction between item 0 and item 1
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 0 and item 2
                 [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 0 and item 3
              ]))
    # items.append(jbw.Item("onion",     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 0, 0], False, 0.0,
    items.append(jbw.Item("onion",     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
               intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.0],
               interaction_fns=[
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],      # parameters for interaction between item 1 and item 0
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],                                       # parameters for interaction between item 1 and item 1
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],  # parameters for interaction between item 1 and item 2
                 [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 1 and item 3
              ]))
    items.append(jbw.Item("jellybean", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
               intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
               interaction_fns=[
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 2 and item 0
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],  # parameters for interaction between item 2 and item 1
                 [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],      # parameters for interaction between item 2 and item 2
                 [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 2 and item 3
              ]))
    items.append(jbw.Item("wall",      [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0, 0, 0, 1], [0, 0, 0, 0], True, 1.0,
               intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[0.0],
               interaction_fns=[
                 [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 0
                 [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 1
                 [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 2
                 [jbw.InteractionFunction.CROSS, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0] # parameters for interaction between item 3 and item 3
              ]))

    # construct the simulator configuration
    return jbw.SimulatorConfig(max_steps_per_movement=1, vision_range=8,
      allowed_movement_directions=[jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED],
      allowed_turn_directions=[jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.ALLOWED],
      no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
      collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
      decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000, seed=1234567890)

def make_config(config_type = 'uniform'):
    if config_type == 'uniform':
        return uniform_config()
    else:
        return default_config()

