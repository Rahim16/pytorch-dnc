from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.controller import Controller

class LSTMController(Controller):
    def __init__(self, args):
        super(LSTMController, self).__init__(args)

        # build model

        self.in_2_hid = nn.LSTMCell(self.input_dim + self.read_vec_dim, self.hidden_dim, 1)
        self.cells = [self.in_2_hid] # i had to do it this ugly way, because magic
        for l in range(self.depth - 1):
            self.in_2_hid1 = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)
            self.cells += [self.in_2_hid1]


        self._reset()

    def _init_weights(self):
        pass

    def forward(self, input_vb, read_vec_vb):
        # print(input_vb.type())
        # print(read_vec_vb.type())

        for l in range(self.depth):
            if l == 0:
                # print(input_vb.contiguous().view(-1, self.input_dim).shape)
                # print(read_vec_vb.contiguous().view(-1, self.read_vec_dim).shape)
                # sys.exit(0)
                input = torch.cat((input_vb.contiguous().view(-1, self.input_dim), \
                                read_vec_vb.contiguous().view(-1, self.read_vec_dim)), 1)
            else:
                input = self.lstm_hidden_vb[l - 1][0]
            self.lstm_hidden_vb[l] = [x for x in self.cells[l](input, tuple(self.lstm_hidden_vb[l]))]
            # we clip the controller hidden states here
            self.lstm_hidden_vb[l] = [self.lstm_hidden_vb[l][0].clamp(min=-self.clip_value, max=self.clip_value),
                                   self.lstm_hidden_vb[l][1]]

        return self.lstm_hidden_vb[-1][0]
