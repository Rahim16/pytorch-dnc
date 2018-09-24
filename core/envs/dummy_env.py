from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import torch
import math
import torch.nn.functional as F

from core.env import Env
from core.envs.interfaces import Measurable

class DummyEnv(Env, Measurable):
    def __init__(self, args, env_ind=0):
        super(DummyEnv, self).__init__(args, env_ind)

        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.range_min = args.range_min
        self.range_max = args.range_max
        self.separate_command = args.separate_command
        self.flag_errors = args.flag_errors
        self.mask_outputs = args.mask_outputs
        self.one_hot_encode = args.one_hot_encode
        self.many_hot_encode = args.many_hot_encode
        self.nums_to_one_hot = {} # cache
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def _preprocessState(self, state):
        # NOTE: state input in size: batch_size x num_words  x len_word
        # NOTE: we return as:        num_words  x batch_size x len_word
        # NOTE: to ease feeding in one row from all batches per forward pass
        for i in range(len(state)):
            state[i] = state[i].permute(1, 0, 2)
        return state

    @property
    def state_shape(self):
        # NOTE: we use this as the input_dim to be consistent with the sl & rl tasks
        index_shift = 2 if self.separate_command else 1
        input_width = 2

        if self.one_hot_encode:
            input_width = index_shift + self.range_max - self.range_min + 1
        elif self.many_hot_encode:
            input_width = index_shift + int(math.ceil(math.log(self.range_max - self.range_min + 1, 2)))

        return input_width

    @property
    def action_dim(self):
        # NOTE: we use this as the output_dim to be consistent with the sl & rl tasks
        return 1

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

    def generate_dataset(self, range_min, range_max):
        index_shift = 2 if self.separate_command else 1 # sequences are twice longer when putting command on another line
        sample_length = self.sequence_length * index_shift

        x = torch.zeros((self.batch_size, sample_length, 2))
        y = torch.zeros((self.batch_size, sample_length, 1))

        mask = (torch.zeros if self.mask_outputs else torch.ones)((self.batch_size, sample_length, 1))


        for sample_id in range(self.batch_size):
            pool = []
            for t in np.asarray(range(self.sequence_length)) * index_shift:
                idx_val = t + index_shift - 1 # next row if shift required
                save_op = random.random() < .5
                error = random.random() < .5
                if save_op or not pool:
                    new_num = random.randint(range_min, range_max)
                    pool.append(new_num)
                    x[sample_id][idx_val][1] = new_num
                else:
                    x[sample_id][t][0] = 1 # read operation
                    if self.mask_outputs:
                        mask[sample_id][idx_val][0] = 1 # only this output is important
                    if error:
                        while True:
                            err_num = random.randint(range_min, range_max)
                            if err_num not in pool:
                                break

                        x[sample_id][idx_val][1] = err_num
                        y[sample_id][idx_val][0] = 1 if self.flag_errors else 0 # error flag
                    else:
                        saved_num = random.choice(pool)
                        x[sample_id][idx_val][1] = saved_num
                        y[sample_id][idx_val][0] = 0 if self.flag_errors else 1 # correct flag

        return x, y, mask

    def num_to_many_hot(self, number, bit_length):
        if number not in self.nums_to_one_hot:
            many_hot = torch.zeros(bit_length)


            i = 0
            while number > 0:
                many_hot[i] = number % 2
                i += 1
                number //= 2

            self.nums_to_one_hot[number] = many_hot

        return self.nums_to_one_hot[number]



    # convert the tensor's value to many hot representation
    # TODO: MAYBE add support for separate_command flag
    def to_many_hot(self, tensor_x, range_min, range_max):
        index_shift = 2 if self.separate_command else 1 # sequences are twice longer when putting command on another line
        bit_length = int(math.ceil(math.log(range_max - range_min + 1, 2)))
        prefix_length = 2 if self.separate_command else 1 # different commands on separate rows

        tensor_padded = torch.zeros((tensor_x.size(0), tensor_x.size(1), prefix_length + bit_length)).to(self.device)
        if not self.separate_command:
            tensor_padded[:, :, 0] = tensor_x[:, :, 0]

        for sample_id in range(tensor_x.size(0)):
            rang = range(tensor_x.size(1))
            if self.separate_command:
                rang = filter(lambda x: x % 2, rang)

            for input_t in rang:
                if self.separate_command:
                    tensor_padded[sample_id, input_t - 1, int(tensor_x[sample_id, input_t - 1, 0])] = 1 # command bit

                tensor_padded[sample_id, input_t, prefix_length:] = \
                   self.num_to_many_hot(int(tensor_x[sample_id, input_t, 1]), bit_length)

        return tensor_padded

    def to_one_hot(self, tensor_x, range_min, range_max, encoding_length=-1):
        index_shift = 2 if self.separate_command else 1 # sequences are twice longer when putting command on another line
        bit_length = max(range_max - range_min + 1, encoding_length)
        prefix_length = 2 if self.separate_command else 1 # different commands on separate rows

        tensor_padded = torch.zeros((tensor_x.size(0), tensor_x.size(1), prefix_length + bit_length)).to(self.device)


        if not self.separate_command:
            tensor_padded[:, :, 0] = tensor_x[:, :, 0]

        for sample_id in range(tensor_x.size(0)):
            rang = range(tensor_x.size(1))
            if self.separate_command:
                rang = filter(lambda x: x % 2, rang)

            for input_t in rang:
                if self.separate_command:
                    tensor_padded[sample_id, input_t - 1, int(tensor_x[sample_id, input_t - 1, 0])] = 1 # command bit

                tensor_padded[sample_id, input_t, int(tensor_x[sample_id, input_t, 1]) - range_min + prefix_length] = 1

        return tensor_padded


    def _generate_sequence(self, train=True):
        # range_min = self.range_min if train else self.range_max + 1
        # range_max = self.range_max if train else self.range_max * 2
        range_min = self.range_min
        range_max = self.range_max
        self.exp_state1 = [data.to(self.device) for data in self.generate_dataset(range_min, range_max)]

        if self.one_hot_encode:
            self.exp_state1[0] = self.to_one_hot(self.exp_state1[0], range_min, range_max)
        elif self.many_hot_encode:
            self.exp_state1[0] = self.to_many_hot(self.exp_state1[0], range_min, range_max)


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

    def measure_error(self, true_y, predicted_y, mask):
        diff = torch.round(torch.abs(true_y - torch.round(predicted_y)) * mask)
        return { "total": diff.size(0), \
                 "errors": int(torch.sum(F.max_pool1d(diff, diff.size(-1))).item())}
