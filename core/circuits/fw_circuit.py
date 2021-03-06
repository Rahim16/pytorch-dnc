import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from core.circuit import Circuit

# Unidirectional cell, abstract class to encapsulate common methods ( forward only )
class FW_Circuit(Circuit):   # NOTE: basically this whole module is treated as a custom rnn cell
    def __init__(self, args):
        super(FW_Circuit, self).__init__(args)

        # dependent on these
        self.controller = None
        self.accessor = None

        self.logger.warning("<-----------------------------======> FW_Circuit:    {Controller, Accessor}")

    def _reset_states(self): # should be called at the beginning of forwarding a new input sequence
        # we first reset the previous read vector
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        # we then reset the controller's hidden state
        self.controller._reset_states()
        # we then reset the write/read weights of heads
        self.accessor._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        # reset internal states
        self.read_vec_ts = torch.zeros(self.batch_size, self.read_vec_dim).fill_(1e-6)
        self._reset_states()

    def forward(self, input_vb):
        # NOTE: the operation order must be the following: control, access{write, read}, output

        # 1. first feed {input, read_vec_{t-1}} to controller
        hidden_vb = self.controller.forward(input_vb, self.read_vec_vb)
        # 2. then we write to memory_{t-1} to get memory_{t}; then read from memory_{t} to get read_vec_{t}
        self.read_vec_vb = self.accessor.forward(hidden_vb)
        # 3. finally we concat the output from the controller and the current read_vec_{t} to get the final output
        output_vb = self.hid_to_out(torch.cat((hidden_vb.view(-1, self.hidden_dim),
                                               self.read_vec_vb.view(-1, self.read_vec_dim)), 1))

        # we clip the output values here
        return torch.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(1, self.batch_size, self.output_dim)
