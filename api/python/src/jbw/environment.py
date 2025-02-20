# Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""OpenAI gym environment implementation for the JBW
simulator."""

from __future__ import absolute_import, division, print_function

try:
  import gym
  from gym import spaces, logger
  modules_loaded = True
except:
  modules_loaded = False

import numpy as np

from .agent import Agent
from .direction import RelativeDirection
from .simulator import Simulator
from .visualizer import MapVisualizer


if not modules_loaded:
  __all__ = []
else:
  __all__ = ['JBWEnv']

  class JBWEnv(gym.Env):
      """JBW environment for OpenAI gym.

      The action space consists of three actions:
        - `0`: Move forward.
        - `1`: Turn left.
        - `2`: Turn right.

      The observation space consists of a dictionary:
        - `scent`: Vector with shape `[S]`, where `S` is the
          scent dimensionality.
        - `vision`: Matrix with shape `[2R+1, 2R+1, V]`,
          where `R` is the vision range and `V` is the
          vision/color dimensionality.
        - `moved`: Binary value indicating whether the last
          action resulted in the agent moving.

      After following the instructions provided in the main
      `README` file to install the `jbw` framework, and
      installing `gym` using `pip install gym`, this
      environment can be used as follows:

      ```
      import gym
      import jbw

      # Use 'JBW-render-v0' to include rendering support.
      # Otherwise, use 'JBW-v0', which should be much faster.
      env = gym.make('JBW-render-v0')

      # The created environment can then be used as any other
      # OpenAI gym environment. For example:
      for t in range(10000):
        # Render the current environment.
        env.render()
        # Sample a random action.
        action = env.action_space.sample()
        # Run a simulation step using the sampled action.
        observation, reward, _, _ = env.step(action)
      ```
      """

      def __init__(
          self, sim_config, reward_fn, render=False):
        """Creates a new JBW environment for OpenAI gym.

        Arguments:
          sim_config(SimulatorConfig) Simulator configuration
                                      to use.
          reward_fn(callable)         Function that takes the
                                      previously collected
                                      items and the current
                                      collected items as inputs
                                      and returns a reward
                                      value.
          render(bool)                Boolean value indicating
                                      whether or not to support
                                      rendering the
                                      environment.
        """
        self.sim_config = sim_config
        self._sim = None
        self._painter = None
        self._reward_fn = reward_fn
        self._render = render

        self.reset()

        # Computing shapes for the observation space.
        scent_shape = [len(self.sim_config.items[0].scent)]
        vision_dim = len(self.sim_config.items[0].color)
        vision_range = self.sim_config.vision_range
        vision_shape = [
          2 * vision_range + 1,
          2 * vision_range + 1,
          vision_dim]

        min_float = np.finfo(np.float32).min
        max_float = np.finfo(np.float32).max
        min_scent = min_float * np.ones(scent_shape)
        max_scent = max_float * np.ones(scent_shape)
        min_vision = min_float * np.ones(vision_shape)
        max_vision = max_float * np.ones(vision_shape)

        # Observations in this environment consist of a scent
        # vector, a vision matrix, and a binary value
        # indicating whether the last action resulted in the
        # agent moving.
        self.observation_space = spaces.Dict({
          'scent': spaces.Box(low=min_scent, high=max_scent),
          'vision': spaces.Box(low=min_vision, high=max_vision),
          'moved': spaces.Discrete(2)})

        # There are three possible actions:
        #   1. Move forward,
        #   2. Turn left,
        #   3. Turn right.
        self.action_space = spaces.Discrete(3)

      def step(self, action):
        """Runs a simulation step.

        Arguments:
          action(int) Action to take, which can be one of:
                        - `0`: Move forward.
                        - `1`: Turn left.
                        - `2`: Turn right.

        Returns:
          observation (dictionary): Contains:
              - `scent`: Vector with shape `[S]`, where `S`
                is the scent dimensionality.
              - `vision`: Matrix with shape
                `[2R+1, 2R+1, V]`, where `R` is the vision
                range and `V` is the vision/color
                dimensionality.
              - `moved`: Binary value indicating whether the
                last action resulted in the agent moving.
          reward (float): Amount of reward obtained from the
              last action.
          done (bool): Whether or not the episode has ended
              which is always `False` for this environment.
          info (dict): Empty dictionary.
        """
        prev_position = self._agent.position()
        prev_items = self._agent.collected_items()

        self._agent._next_action = action
        self._agent.do_next_action()

        position = self._agent.position()
        items = self._agent.collected_items()
        reward = self._reward_fn(prev_items, items)
        done = False

        self.state = {
          'scent': self._agent.scent(),
          'vision': self._agent.vision(),
          'moved': np.any(prev_position != position)}

        return self.state, reward, done, {}

      def reset(self):
        """Resets this environment to its initial state."""
        del self._sim
        self._sim = Simulator(sim_config=self.sim_config)
        self._agent = _JBWEnvAgent(self._sim)
        if self._render:
          del self._painter
          self._painter = MapVisualizer(
            self._sim, self.sim_config,
            bottom_left=(-70, -70), top_right=(70, 70))
        self.state = {
          'scent': self._agent.scent(),
          'vision': self._agent.vision(),
          'moved': False}
        return self.state

      def render(self, mode='matplotlib'):
        """Renders this environment in its current state.
         Note that, in order to support rendering,
        `render=True` must be passed to the environment
        constructor.

        Arguments:
          mode(str) Rendering mode. Currently, only
                    `"matplotlib"` is supported.
        """
        if mode == 'matplotlib' and self._render:
          self._painter.draw()
        elif not self._render:
          logger.warn(
            'Need to pass `render=True` to support '
            'rendering.')
        else:
          logger.warn(
            'Invalid rendering mode "%s". '
            'Only "matplotlib" is supported.')

      def update_render(self, render_val):

        if render_val:
          # implies the environment will start rendering now
          del self._painter
          self._painter = MapVisualizer(
            self._sim, self.sim_config,
            bottom_left=(-70, -70), top_right=(70, 70)
          )
        else:
          del self._painter
          self._painter = None

        self._render = render_val

      def close(self):
        """Deletes the underlying simulator and deallocates
        all associated memory. This environment cannot be used
        again after it's been closed."""
        del self._sim
        return

      def seed(self, seed=None):
        self.sim_config.seed = seed
        self.reset()
        return


class _JBWEnvAgent(Agent):
  """Helper class for the JBW environment, that represents
  a JBW agent living in the simulator.
  """

  def __init__(self, simulator):
    """Creates a new JBW environment agent.

    Arguments:
      simulator(Simulator)  The simulator the agent lives in.
    """
    super(_JBWEnvAgent, self).__init__(
      simulator, load_filepath=None)
    self._next_action = None

  def do_next_action(self):
    if self._next_action == 0:
      self.move(RelativeDirection.FORWARD)
    elif self._next_action == 1:
      self.turn(RelativeDirection.LEFT)
    elif self._next_action == 2:
      self.turn(RelativeDirection.RIGHT)
    else:
      logger.warn(
        'Ignoring invalid action %d.'
        % self._next_action)

  # There is no need for saving and loading an agent's
  # state, as that can be done outside the gym environment.

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass
