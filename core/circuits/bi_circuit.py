import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from core.circuit import Circuit

# Bidirectional circuit, abstract class to encapsulate common methods ( forward only )
class BI_Circuit(Circuit):   # NOTE: basically this whole module is treated as a custom rnn cell
    def __init__(self, args):
        super(BI_Circuit, self).__init__(args)

        # dependent on these
        self.fw_pass_controller = None
        self.bw_pass_controller = None
        self.accessor = None

        self.logger.warning("<-----------------------------======> BI_Circuit:    {FW_Controller, BW_Controller, Accessor}")

    def _reset_states(self): # should be called at the beginning of forwarding a new input sequence
        # we first reset the previous read vector
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        # we then reset the controllers' hidden state
        self.fw_pass_controller._reset_states()
        self.bw_pass_controller._reset_states()
        # we then reset the write/read weights of heads
        self.accessor._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        # reset internal states
        self.read_vec_ts = torch.zeros(self.batch_size, self.read_vec_dim).fill_(1e-6)
        self._reset_states()

    """As opposed to singe direction forward-only method bidirectional forward returns an array with predictions for all timesteps."""
    def forward(self, input):
        # NOTE: the operation order must be the following: control, access{write, read}, output
        self._reset_states()

        controller_outputs = []
        read_vectors = []

        sequence_length = input.size(0)
        for i in range(2 * sequence_length):
            idx = i if i < sequence_length else input.size(0) - 1 - i

            if idx >= 0:
                controller = self.fw_pass_controller
            else:
                controller = self.bw_pass_controller

                if idx == -1: # turning back, copy activations and memory of previous lstm cell
                    self.bw_pass_controller.lstm_hidden_vb = self.fw_pass_controller.lstm_hidden_vb

            # 1. first feed {input, read_vec_{t-1}} to controller
            hidden_vb = controller.forward(input[idx], self.read_vec_vb)
            # 2. then we write to memory_{t-1} to get memory_{t}; then read from memory_{t} to get read_vec_{t}
            self.read_vec_vb = self.accessor.forward(hidden_vb)

            controller_outputs.append(hidden_vb)
            read_vectors.append(self.read_vec_vb)

        outputs = []
        # now concat outputs from both directions and make predictions
        for i in range(sequence_length):
            back_idx = sequence_length + i

            output_vb = self.hid_to_out(torch.cat((controller_outputs[i].view(-1, self.hidden_dim),
                                                   read_vectors[i].view(-1, self.read_vec_dim),
                                                   controller_outputs[back_idx].view(-1, self.hidden_dim),
                                                   read_vectors[back_idx].view(-1, self.read_vec_dim)), 1))

            outputs.append(torch.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(1, self.batch_size, self.output_dim))

        return torch.stack(outputs)
