from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import visdom
import torch
import torch.nn as nn
import torch.optim as optim

from utils.helpers import loggerConfig

CONFIGS = [
# agent_type, env_type,      game, circuit_type
[ "empty",    "repeat-copy", "",   "none"      ],  # 0
[ "sl",       "copy",        "",   "ntm"       ],  # 1
[ "sl",       "repeat-copy", "",   "dnc"       ],  # 2
[ "sl",       "dummy",       "",   "dnc"       ],  # 3
[ "sl",       "dummy",       "",   "bidnc"     ],  # 4
[ "sl",       "dummy",       "",   "fdnc"      ],  # 5
]

class Params(object):   # NOTE: shared across all modules
    def __init__(self):
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)

        # training signature
        self.machine     = "gpu1"       # "machine_id"
        self.timestamp   = "18072300"   # "yymmdd##"
        # training configuration
        self.mode        = 2            # 1(train) | 2(test model_file) | 3 explore
        self.config      = 5

        self.seed        = 1
        self.render      = False        # whether render the window from the original envs or not
        self.visualize   = False         # whether do online plotting and stuff or not
        self.save_best   = False        # save model w/ highest reward if True, otherwise always save the latest model

        self.agent_type, self.env_type, self.game, self.circuit_type = CONFIGS[self.config]

        self.use_cuda    = torch.cuda.is_available()
        self.dtype       = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # prefix for model/log/visdom
        self.refs        = self.machine + "_" + self.timestamp # NOTE: using this as env for visdom
        self.root_dir    = os.getcwd()

        # model files
        # NOTE: will save the current model to model_name
        self.model_name  = self.root_dir + "/models/" + self.refs + ".pth"
        # NOTE: will load pretrained model_file if not None
        self.model_file  = None #self.root_dir + "/models/" + "dummy_model.pth" #self.root_dir + "/models/{TODO:FILL_IN_PRETAINED_MODEL_FILE}.pth"

        if self.mode in [2, 3]:
            self.model_file  = self.model_name  # NOTE: so only need to change self.mode to 2 to test the current training
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.refs = self.refs + "_test"     # NOTE: using this as env for visdom for testing, to avoid accidentally redraw on the training plots

        # logging configs
        self.log_name    = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger      = loggerConfig(self.log_name, self.verbose)
        self.logger.warning("<===================================>")

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.warning("bash$: python -m visdom.server")           # activate visdom server on bash
            self.logger.warning("http://localhost:8097/env/" + self.refs)   # open this address on browser

class EnvParams(Params):    # settings for network architecture
    def __init__(self):
        super(EnvParams, self).__init__()

        self.batch_size = None
        if self.env_type == "copy":
            self.len_word  = 8
            self.min_num_words = 5
            self.max_num_words = 10
        elif self.env_type == "repeat-copy":
            self.len_word  = 4
            self.min_num_words = 1
            self.max_num_words = 2
            self.min_repeats   = 1
            self.max_repeats   = 2
            self.max_repeats_norm = 10.
        elif self.env_type == "data-loader":
            self.input_file = "/amd/home/mammadli/repos/python_parrl/code_samples/dummy.pickled"
        elif self.env_type == "dummy":
            self.sequence_length = 8
            self.range_min = 1
            self.range_max = 10
            self.separate_command = False # put command and value on separate time steps
            self.flag_errors = False
            self.mask_outputs = False
            self.one_hot_encode = False
            self.many_hot_encode = True
            # self.input_file = "/amd/home/mammadli/repos/python_parrl/code_samples/data.pickled"

class ControllerParams(Params):
    def __init__(self):
        super(ControllerParams, self).__init__()

        self.batch_size     = None
        self.input_dim      = None  # set after env
        self.read_vec_dim   = None  # num_read_heads x mem_wid
        self.output_dim     = None  # set after env
        self.hidden_dim     = None  #
        self.mem_hei        = None  # set after memory
        self.mem_wid        = None  # set after memory
        self.depth          = 1  # number of layers

class HeadParams(Params):
    def __init__(self):
        super(HeadParams, self).__init__()

        self.num_heads = None
        self.batch_size = None
        self.hidden_dim = None
        self.mem_hei = None
        self.mem_wid = None
        self.num_allowed_shifts = 3

class WriteHeadParams(HeadParams):
    def __init__(self):
        super(WriteHeadParams, self).__init__()

class ReadHeadParams(HeadParams):
    def __init__(self):
        super(ReadHeadParams, self).__init__()
        if self.circuit_type in ["dnc", "bidnc", "fdnc"]:
            self.num_read_modes = None

class MemoryParams(Params):
    def __init__(self):
        super(MemoryParams, self).__init__()

        self.batch_size = None
        self.mem_hei = None
        self.mem_wid = None

class AccessorParams(Params):
    def __init__(self):
        super(AccessorParams, self).__init__()

        self.batch_size = None
        self.hidden_dim = None
        self.num_write_heads = None
        self.num_read_heads = None
        self.mem_hei = None
        self.mem_wid = None
        self.clip_value = None
        self.write_head_params = WriteHeadParams()
        self.read_head_params  = ReadHeadParams()
        self.memory_params     = MemoryParams()

class CircuitParams(Params):# settings for network architecture
    def __init__(self):
        super(CircuitParams, self).__init__()

        self.batch_size     = None
        self.input_dim      = None  # set after env
        self.read_vec_dim   = None  # num_read_heads x mem_wid
        self.output_dim     = None  # set after env

        if self.circuit_type == "ntm":
            self.hidden_dim      = 100
            self.num_write_heads = 1
            self.num_read_heads  = 1
            self.mem_hei         = 128
            self.mem_wid         = 20
            self.clip_value      = 20.   # clips controller and circuit output values to in between
        elif self.circuit_type in ["dnc", "bidnc", "fdnc"]:
            self.hidden_dim      = 64
            self.num_write_heads = 1
            self.num_read_heads  = 2
            self.mem_hei         = 20
            self.mem_wid         = 10
            self.clip_value      = 20.   # clips controller and circuit output values to in between

        self.controller_params = ControllerParams()
        self.accessor_params   = AccessorParams()

class AgentParams(Params):  # hyperparameters for drl agents
    def __init__(self):
        super(AgentParams, self).__init__()

        if self.agent_type == "sl":
            if self.circuit_type == "ntm":
                self.criteria       = nn.BCELoss()
                self.optim          = optim.RMSprop

                self.steps          = 1000    # max #iterations
                self.batch_size     = 16
                self.early_stop     = None      # max #steps per episode
                self.clip_grad      = 50.
                self.lr             = 1e-4
                self.optim_eps      = 1e-10     # NOTE: we use this setting to be equivalent w/ the default settings in tensorflow
                self.optim_alpha    = 0.9       # NOTE: only for rmsprop, alpha is the decay in tensorflow, whose default is 0.9
                self.eval_freq      = 500
                self.eval_steps     = 50
                self.prog_freq      = self.eval_freq
                self.test_nepisodes = 5
            elif self.circuit_type in ["dnc", "bidnc", "fdnc"]:
                self.criteria       = nn.BCELoss()
                self.optim          = optim.RMSprop

                self.steps          = 50000    # max #iterations
                self.batch_size     = 100 if self.mode != 3 else 1
                self.early_stop     = None      # max #steps per episode
                self.clip_grad      = 50.
                self.lr             = 1e-4
                self.optim_eps      = 1e-10     # NOTE: we use this setting to be equivalent w/ the default settings in tensorflow
                self.optim_alpha    = 0.9       # NOTE: only for rmsprop, alpha is the decay in tensorflow, whose default is 0.9
                self.eval_freq      = 100
                self.eval_steps     = 4
                self.prog_freq      = self.eval_freq
                self.test_nepisodes = 10
        elif self.agent_type == "empty":
            self.criteria       = nn.BCELoss()
            self.optim          = optim.RMSprop

            self.steps          = 10000    # max #iterations
            self.batch_size     = 16
            self.early_stop     = None      # max #steps per episode
            self.clip_grad      = 50.
            self.lr             = 1e-4
            self.optim_eps      = 1e-10     # NOTE: we use this setting to be equivalent w/ the default settings in tensorflow
            self.optim_alpha    = 0.9       # NOTE: only for rmsprop, alpha is the decay in tensorflow, whose default is 0.9
            self.eval_freq      = 500
            self.eval_steps     = 50
            self.prog_freq      = self.eval_freq
            self.test_nepisodes = 5

        self.env_params     = EnvParams()
        self.circuit_params = CircuitParams()

class Options(Params):
    agent_params  = AgentParams()
