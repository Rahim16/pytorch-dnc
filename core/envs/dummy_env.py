from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from random import randint
import torch
import torch.utils.data as data_utils
import sys

from core.env import Env
from utils.helpers import read_data_file

class DummyEnv(Env):
    def __init__(self, args, env_ind=0):
        super(DummyEnv, self).__init__(args, env_ind)

        # state space setup
        self.batch_size = args.batch_size

        self.data = read_data_file(args.input_file)

        self.train_x, self.train_y, self.dev_x, self.dev_y = self.data["train_x"], self.data["train_y"], self.data["dev_x"], self.data["dev_y"]
        self.train_dataset = data_utils.TensorDataset(self.train_x, self.train_y)
        self.dev_dataset = data_utils.TensorDataset(self.dev_x, self.dev_y)
        self.train_data_loader = data_utils.DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
        self.dev_data_loader = data_utils.DataLoader(self.dev_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def _preprocessState(self, state):
        # NOTE: state input in size: batch_size x num_words  x len_word
        # NOTE: we return as:        num_words  x batch_size x len_word
        # NOTE: to ease feeding in one row from all batches per forward pass
        for i in range(len(state)):
            state[i] = state[i].permute(1, 0, 2)
        return state

    # this is just size of the input
    @property
    def state_shape(self):
        # NOTE: we use this as the input_dim to be consistent with the sl & rl tasks
        return self.train_x.shape[-1]

    # output_size
    @property
    def action_dim(self):
        # NOTE: we use this as the output_dim to be consistent with the sl & rl tasks
        return self.train_y.shape[-1]

    def render(self):
        pass

    def _readable(self, datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

    def visual(self, input_ts, target_ts, mask_ts, output_ts=None):
        """
        input_ts:  [(num_wordsx2+2) x batch_size x (len_word+2)]
        target_ts: [(num_wordsx2+2) x batch_size x (len_word)]
        mask_ts:   [(num_wordsx2+2) x batch_size x (len_word)]
        output_ts: [(num_wordsx2+2) x batch_size x (len_word)]
        """
        output_ts = torch.round(output_ts * mask_ts) if output_ts is not None else None
        input_strings  = [self._readable(input_ts[:, 0, i])  for i in range(input_ts.size(2))]
        target_strings = [self._readable(target_ts[:, 0, i]) for i in range(target_ts.size(2))]
        mask_strings   = [self._readable(mask_ts[:, 0, 0])]
        output_strings = [self._readable(output_ts[:, 0, i]) for i in range(output_ts.size(2))] if output_ts is not None else None
        input_strings  = 'Input:\n'  + '\n'.join(input_strings)
        target_strings = 'Target:\n' + '\n'.join(target_strings)
        mask_strings   = 'Mask:\n'   + '\n'.join(mask_strings)
        output_strings = 'Output:\n' + '\n'.join(output_strings) if output_ts is not None else None
        # strings = [input_strings, target_strings, mask_strings, output_strings]
        # self.logger.warning(input_strings)
        # self.logger.warning(target_strings)
        # self.logger.warning(mask_strings)
        # self.logger.warning(output_strings)
        print(input_strings)
        print(target_strings)
        print(mask_strings)
        print(output_strings) if output_ts is not None else None

    def sample_random_action(self):
        pass

    def _generate_sequence(self, train=True):
        self.exp_state1 = []

        data_loader = self.train_data_loader if train else self.dev_data_loader
        for sample_batched in data_loader:
            x, y = sample_batched[0].to(self.device), sample_batched[1].to(self.device)
            self.exp_state1.append(x)
            self.exp_state1.append(y)
            self.exp_state1.append(torch.ones_like(y))
            return

    def reset(self, train=True):
        self._reset_experience()
        self._generate_sequence(train)
        return self._get_experience()

    def step(self, action_index, train=True):
        self.exp_action = action_index
        self._generate_sequence(train)
        return self._get_experience()

    def reset_for_eval(self):
        return self.reset(train=False)

    def step_eval(self, action_index):
        return self.step(action_index, train=False)
