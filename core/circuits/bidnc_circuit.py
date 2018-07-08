import torch.nn as nn

from core.circuits.bi_circuit import BI_Circuit
from core.controllers.lstm_controller import LSTMController as Controller
from core.accessors.dynamic_accessor import DynamicAccessor as Accessor

class BiDNCCircuit(BI_Circuit):
    def __init__(self, args):
        super(BiDNCCircuit, self).__init__(args)

        # functional components
        self.fw_pass_controller = Controller(self.controller_params)
        self.bw_pass_controller = Controller(self.controller_params)
        self.accessor = Accessor(self.accessor_params)

        # build model
        self.hid_to_out = nn.Linear(2 * (self.hidden_dim + self.read_vec_dim), self.output_dim)

        self._reset()

    def _init_weights(self):
        pass
