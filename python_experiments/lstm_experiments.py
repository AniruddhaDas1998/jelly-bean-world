import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from env_configs import make_config
from reward_fns import avoid_onion
from lstm_model import Transition, ReplayMemory, DQN_LSTM
from utils import window_plot, create_gif

# RL-Training code is adapted from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_env(env_type='JBW-v0', config=None, reward_fn=lambda prev_items, items: sum(items) - sum(prev_items)):
  # default reward gives +1 reward for every item collected
  if config is None:
    # create a local simulator configuration
    config = make_config()
  if env_type=='JBW-v0':
    return gym.make(env_type, sim_config=config, reward_fn=reward_fn, render=False)
  else:
    return gym.make(env_type, sim_config=config, reward_fn=reward_fn, render=True)

REWARD_FN = avoid_onion()
config = make_config('uniform')

env = get_env(reward_fn=REWARD_FN, config=config)

################################################################################
########################### Hyperparameters and Setup ##########################
################################################################################


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym.
screen_height, screen_width, channels = env._agent.vision().shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN_LSTM(screen_height, screen_width, n_actions).to(device)
target_net = DQN_LSTM(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


# these parameters are taken from the github implementation at:
# https://github.com/eaplatanios/jelly-bean-world/blob/master/api/swift/Sources/JellyBeanWorldExperiments/Experiment.swift
optimizer = optim.Adam(policy_net.parameters(), lr=5e-5, amsgrad=True)
# replay memory is much smaller than the usual 10,000 to avoid stale hidden states
memory = ReplayMemory(500)

steps_done = 0

def select_action(state, lstm_inputs, inference=False):
    # this should basically just have a single state passed in
    # lstm_inputs should be tuple with shapes (1 x hidden, 1 x hidden)

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or inference:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            net_output = policy_net(state, lstm_inputs)
            (action_logits, action_values, (h_out, c_out)) = net_output

            # additionally returning updated hidden+cell state
            return action_logits.max(1)[1].view(1, 1), (h_out, c_out)

            # return policy_net(state).max(1)[1].view(1, 1)
    else:
        # additionally returning updated hidden+cell state
        (action_logits, action_values, (h_out, c_out)) = policy_net(state, lstm_inputs)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), (h_out, c_out)

def optimize_model():

    # we will use the lstm_inputs that are obtained from the Memory Buffer

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # The next line basically just makes all states, actions etc more easily
    # accessible i.e. all Transition.state has all the states in the batch
    # as opposed to a single Transition containing a single state

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # the mask computation is probably useless due to the fact this is a single
    # never ending episode

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)


    # TODO pass in actual moved parameters if required
    # TODO vectorize construction of batch
    non_final_next_states = (
        torch.cat([s[0] for s in batch.next_state if s is not None]),
        torch.cat([s[1] for s in batch.next_state if s is not None]),
        False
    )

    state_batch = (
        torch.cat([s[0] for s in batch.state if s is not None]),
        torch.cat([s[1] for s in batch.state if s is not None]),
        False
    )

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    h_batch = torch.cat([hs[0] for hs in batch.hidden_state], dim=0)
    c_batch = torch.cat([hs[1] for hs in batch.hidden_state], dim=0)
    # not detaching seems to break computational graph
    # not sure why but I think this is a necessary step to only backpropagate
    # one timestep back as opposed to all the way back
    batch_lstm_inputs = (h_batch.detach(), c_batch.detach())

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    (action_logits, action_values, (h_out, c_out)) = policy_net(state_batch, batch_lstm_inputs)

    action_logits = action_logits.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # next_state_values[non_final_mask] = target_net(non_final_next_states, lstm_inputs).max(1)[0].detach()
    t_action_logits, _, _ = target_net(non_final_next_states, batch_lstm_inputs)
    next_state_values[non_final_mask] = t_action_logits.max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(action_logits.float(), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return (h_out, c_out), loss.item()

################################################################################
################################# TRAINING LOOP ################################
################################################################################

NUM_STEPS = 1000000

LSTM_INIT = torch.zeros((1, 256), device=device), torch.zeros((1, 256), device=device)

# Initialize the environment
env = get_env(reward_fn=REWARD_FN)

# uncommenting this allows you to see what the environment initially looks like
# env.update_render(True)
# env.render()
# env.update_render(False)

# we will use this to track the rewards across the training
rewards = []
action_dist = np.zeros(env.action_space.n)
losses = []

R_MEAN = 0.0
DONE, DONE_IDX = False, -1
L_IDX = 0
WINDOW_SIZE = 100000

LSTM_INPUTS = LSTM_INIT

for i_step in range(NUM_STEPS):

    if DONE:


        # uncomment this to do nothing when DONE is true
        pass

        # uncomment this to break when the DONE flag is true
        # break

        # uncomment this to begin rendering during the training run

        # if env._render:

        #   env.render()

        #   if (i_step+1)%100 == 0:
        #     print('On render step: {}'.format(i_step-DONE_IDX))
        #   if (i_step+1)%500 == 0:
        #     print('Creating GIF')
        #     create_gif(i_step-DONE_IDX)
        #     print('Done\n')

        # else:
        #   print('Beginning rendering of training')
        #   env.update_render(True)
        #   env.render()

    # print(i_step)
    if (i_step+1) % 5000 == 0:
      print('Completed step {}/{} with R_MEAN: {}'.format(i_step+1, NUM_STEPS, R_MEAN))

    # TODO fix and pass in actual moved parameter -- if required
    # Moved is not being used right now
    state = (
        torch.tensor(env._agent.vision(), device=device).view((1, screen_height, screen_width, channels)),
        torch.tensor(env._agent.scent(), device=device).view((1, -1)),
        False
      )

    # Select and perform an action
    action, LSTM_OUTPUTS = select_action(state, LSTM_INPUTS)
    observation, reward, done, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)

    rewards.append(reward.item())
    action_dist[action.item()]+=1

    # update running mean to track termination
    if i_step < WINDOW_SIZE:
      R_MEAN = ((R_MEAN * i_step)+reward.item())/(i_step+1)
    else:
      R_MEAN = ((R_MEAN * WINDOW_SIZE)+reward.item()-rewards[L_IDX])/(WINDOW_SIZE)
      L_IDX+=1

    if R_MEAN >= 0.15 and i_step>=(WINDOW_SIZE*2):
      # when this threshold is crossed, we've done atleast WINDOW_SIZE steps,
      # we're done
      DONE = True
      DONE_IDX = i_step
      print('DONE. With R_MEAN={}'.format(R_MEAN))


    next_state = (
        torch.tensor(observation['vision'], device=device).view((1, screen_height, screen_width, channels)),
        torch.tensor(observation['scent'], device=device).view((1, -1)),
        observation['moved']
    )

    # Store the transition in memory
    memory.push(state, action, next_state, reward, LSTM_INPUTS)

    # Move to the next state
    state = next_state
    LSTM_INPUTS = LSTM_OUTPUTS[0].detach(), LSTM_OUTPUTS[1].detach()

    # Perform one step of the optimization (on the target network)
    optim_out = optimize_model()

    if optim_out is not None:
      _, loss_huber = optim_out
      losses.append(loss_huber)

    # Update the target network, copying all weights and biases in DQN
    if i_step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Completed Training')

# plot losses
plt.plot(losses)
plt.show()

# see distribution over actions to ensure model isn't only taking a single
# action all the time
plt.bar(np.arange(env.action_space.n), action_dist)
plt.show()

# total/average reward for run
sum(rewards), sum(rewards)*1.0/NUM_STEPS

# see items collected in the run
env._agent.collected_items()

# plot reward rate for entire run
window_plot(rewards, WINDOW_SIZE=100000, INC=1)

################################################################################
############################ UNROLL MODEL (no gradient) ########################
################################################################################

# here we continue running the model on the environment to see how it behaves
# very similar to training just without gradient updates

UNROLL_STEPS = 1000000

rewards = []
action_dist = np.zeros(env.action_space.n)

# don't reset to init since we're continuing from earlier environment state
# LSTM_INPUTS = LSTM_INIT

for i_step in range(UNROLL_STEPS):

    if (i_step+1)%50000 == 0:
        print('On step: {}'.format(i_step+1))

    # TODO fix and pass in actual moved parameter
    # TODO construct state and take actions dictated by model
    # Moved is not being used right now
    state = (
        torch.tensor(env._agent.vision(), device=device).view((1, screen_height, screen_width, channels)),
        torch.tensor(env._agent.scent(), device=device).view((1, -1)),
        False
      )

    # Select and perform an action
    # NOTE: even though inference is technically True and we don't want the
    # model to go off policy, we empirically see that the model often gets stuck
    # rotating in an area and needs the epsilon-greedy to help it get unstuck
    action, LSTM_OUTPUTS = select_action(state, LSTM_INPUTS, inference=False)

    # currently sampling actions at random
    # action = env.action_space.sample()
    observation, reward, done, _ = env.step(action.item())

    rewards.append(reward)
    action_dist[action]+=1

    LSTM_INPUTS = LSTM_OUTPUTS

print('Completed unrolling')

# see distribution over actions to ensure model isn't only taking a single
# action all the time
plt.bar(np.arange(env.action_space.n), action_dist)
plt.show()

# total/average reward for run
sum(rewards), sum(rewards)*1.0/NUM_STEPS

# see items collected in the run -- this includes items collected while training
env._agent.collected_items()

# plot reward rate for entire run
window_plot(rewards, WINDOW_SIZE=100000, INC=1)

################################################################################
######################### UNROLL MODEL (with gradient + viz.) ##################
################################################################################

# don't reset to init since we're continuing from earlier environment state
# LSTM_INPUTS = LSTM_INIT

SIM_STEPS = 10000

rewards = []
action_dist = np.zeros((env.action_space.n))

# this turns allows the environment to begin rendering
# we need to still call env.render() for every rendering
# NOTE: env.render() is a very slow function
env.update_render(True)

for i_step in range(SIM_STEPS):

    if (i_step+1) % 100 == 0:
      print('Completed step {}/{} with R_MEAN: {}'.format(i_step+1, NUM_STEPS, R_MEAN))

    # TODO fix and pass in actual moved parameter -- if required
    # Moved is not being used right now
    state = (
        torch.tensor(env._agent.vision(), device=device).view((1, screen_height, screen_width, channels)),
        torch.tensor(env._agent.scent(), device=device).view((1, -1)),
        False
      )

    # Select and perform an action
    action, LSTM_OUTPUTS = select_action(state, LSTM_INPUTS)
    observation, reward, done, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)

    rewards.append(reward.item())
    action_dist[action.item()]+=1

    # update running mean to track termination
    if i_step < WINDOW_SIZE:
      R_MEAN = ((R_MEAN * i_step)+reward.item())/(i_step+1)
    else:
      R_MEAN = ((R_MEAN * WINDOW_SIZE)+reward.item()-rewards[L_IDX])/(WINDOW_SIZE)
      L_IDX+=1

    next_state = (
        torch.tensor(observation['vision'], device=device).view((1, screen_height, screen_width, channels)),
        torch.tensor(observation['scent'], device=device).view((1, -1)),
        observation['moved']
    )

    # Store the transition in memory
    memory.push(state, action, next_state, reward, LSTM_INPUTS)

    # Move to the next state
    state = next_state
    LSTM_INPUTS = LSTM_OUTPUTS[0].detach(), LSTM_OUTPUTS[1].detach()

    # Perform one step of the optimization (on the target network)
    optim_out = optimize_model()

    if optim_out is not None:
      _, loss_huber = optim_out
      losses.append(loss_huber)

    # Update the target network, copying all weights and biases in DQN
    if i_step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # actually does the rendering
    # will save the rendering png in this function call
    env.render()

    if (i_step+1)%500 == 0:
        print('Creating GIF')
        create_gif(i_step+1)
        print('Done\n')

print('Completed unrolling with gradient updates and rendering')
