from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

from core.circuits.fw_circuit import FW_Circuit
from core.controllers.lstm_controller import LSTMController as Controller
from core.accessors.dynamic_accessor_with_strength import DynamicAccessorWithStrength as Accessor

class FDNCCircuit(FW_Circuit):
    def __init__(self, args):
        super(FDNCCircuit, self).__init__(args)

        # each read vector has also strength
        self.read_vec_dim = self.num_read_heads * self.mem_wid + self.num_read_heads
        # controller
        self.controller_params.read_vec_dim = self.read_vec_dim

        # functional components
        self.controller = Controller(self.controller_params)
        self.accessor = Accessor(self.accessor_params)

        # build model
        self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)

        self._reset()

    def _init_weights(self):
        pass
