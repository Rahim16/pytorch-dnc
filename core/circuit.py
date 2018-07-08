from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import abc

class Circuit(nn.Module):   # NOTE: basically this whole module is treated as a custom rnn cell
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        super(Circuit, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

        # functional components
        self.controller_params = args.controller_params
        self.accessor_params = args.accessor_params

        # now we fill in the missing values for each module
        self.read_vec_dim = self.num_read_heads * self.mem_wid
        # controller
        self.controller_params.batch_size = self.batch_size
        self.controller_params.input_dim = self.input_dim
        self.controller_params.read_vec_dim = self.read_vec_dim
        self.controller_params.output_dim = self.output_dim
        self.controller_params.hidden_dim = self.hidden_dim
        self.controller_params.mem_hei = self.mem_hei
        self.controller_params.mem_wid = self.mem_wid
        self.controller_params.clip_value = self.clip_value
        # accessor: {write_heads, read_heads, memory}
        self.accessor_params.batch_size = self.batch_size
        self.accessor_params.hidden_dim = self.hidden_dim
        self.accessor_params.num_write_heads = self.num_write_heads
        self.accessor_params.num_read_heads = self.num_read_heads
        self.accessor_params.mem_hei = self.mem_hei
        self.accessor_params.mem_wid = self.mem_wid
        self.accessor_params.clip_value = self.clip_value

        self.logger.warning("<-----------------------------======> Circuit:    {Controller, Accessor}")

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<-----------------------------======> Circuit:    {Overall Architecture}")
        self.logger.warning(self)

    @abc.abstractmethod
    def forward(self, input_vb):
        pass
