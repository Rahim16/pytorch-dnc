from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.init_weights import init_weights, normalized_columns_initializer

class Controller(nn.Module):
    def __init__(self, args):
        super(Controller, self).__init__()
        # logging
        self.logger = args.logger
        # general params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.read_vec_dim = args.read_vec_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value
        self.depth = args.depth

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<--------------------------------===> Controller:")
        self.logger.warning(self)

    def _reset_states(self):
        # we reset controller's hidden state
        self.lstm_hidden_vb = [[Variable(self.lstm_hidden_ts[l][0]).type(self.dtype), \
                                Variable(self.lstm_hidden_ts[l][1]).type(self.dtype)] for l in range(self.depth)]


    def _reset(self):           # NOTE: should be called at each child's __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        self.print_model()
        # reset internal states
        self.lstm_hidden_ts = []
        for l in range(self.depth):
            ts_l = []
            ts_l.append(torch.zeros(self.batch_size, self.hidden_dim))
            ts_l.append(torch.zeros(self.batch_size, self.hidden_dim))
            self.lstm_hidden_ts.append(ts_l)

        self._reset_states()

    def forward(self, input_vb):
        raise NotImplementedError("not implemented in base calss")
