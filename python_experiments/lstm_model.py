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

# code adapted from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hidden_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# model inspired from original JBW paper: https://arxiv.org/pdf/2002.06306.pdf
class DQN_LSTM(nn.Module):

    def __init__(self, h, w, n_actions):
        super(DQN_LSTM, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, stride=1)
        # self.bn2 = nn.BatchNorm2d(16)

        OUT_DIM = 512

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = (conv2d_size_out(conv2d_size_out(w, 3, 2), 2, 1))
        convh = (conv2d_size_out(conv2d_size_out(h, 3, 2), 2, 1))
        conv_output_size = convw * convh * 16
        self.conv_head = nn.Linear(conv_output_size, OUT_DIM)

        linear_input_size = 3
        self.sl1 = nn.Linear(linear_input_size, 32)
        self.scent_head = nn.Linear(32, OUT_DIM)

        self.lstm = nn.LSTMCell(
            input_size=OUT_DIM+OUT_DIM,
            hidden_size=(OUT_DIM+OUT_DIM)//4,
        )

        self.lstm_dense_action = nn.Linear((OUT_DIM+OUT_DIM)//4, n_actions)
        self.lstm_dense_value = nn.Linear((OUT_DIM+OUT_DIM)//4, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, lstm_inputs):
        h_n, c_n = lstm_inputs
        vision_input, scent_input, moved = x

        v1 = F.gelu(self.conv1(vision_input.permute(0, 3, 1, 2).clone()))
        v2 = F.gelu(self.conv2(v1))
        v_flat = v2.view((v2.size(0), -1))
        vision_out = F.gelu(self.conv_head(v_flat))

        s1 = F.gelu(self.sl1(scent_input))
        scent_out = F.gelu(self.scent_head(s1))

        concat_vs_out = torch.cat((vision_out, scent_out), dim=-1)

        lstm_out = self.lstm(concat_vs_out, (h_n, c_n))
        h_out, c_out = lstm_out

        action_logits = self.lstm_dense_action(h_out)
        action_values = self.lstm_dense_value(h_out)

        return (action_logits, action_values, (h_out, c_out))
