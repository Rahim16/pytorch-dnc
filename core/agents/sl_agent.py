from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helpers import Experience
from core.agent import Agent
from core.envs.interfaces import Measurable

class SLAgent(Agent):   # for supervised learning tasks
    def __init__(self, args, env_prototype, circuit_prototype):
        super(SLAgent, self).__init__(args, env_prototype, circuit_prototype)
        self.logger.warning("<===================================> Agent:")

        # env
        self.env_params.batch_size = args.batch_size
        self.env = self.env_prototype(self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim  = self.env.action_dim

        # circuit
        self.circuit_params = args.circuit_params
        self.circuit_params.batch_size = args.batch_size
        self.circuit_params.input_dim = self.state_shape
        self.circuit_params.output_dim = self.action_dim
        self.circuit = self.circuit_prototype(self.circuit_params)
        self._load_model(self.model_file)   # load pretrained circuit if provided

        # to calculate the error rate in percentages
        self._reset_error_metrics()

        self.training = False # NOTE: maybe remove this, just added to avoid references to undefined in explore
        self._reset_experience()

    def _reset_training_loggings(self):
        self._reset_testing_loggings()
        self.training_loss_avg_log = []

    def _reset_testing_loggings(self):
        # setup logging for testing/evaluation stats
        self.loss_avg_log = []
        # placeholders for windows for online curve plotting
        if self.visualize:
            self.win_loss_avg = "win_loss_avg"

    def _preprocessState(self, state):
        if np.ndarray == type(state):
            state_ts = torch.from_numpy(state).type(self.dtype)
        else:
            state_ts = state

        return state_ts

    def _forward_sequential(self, input_ts):
        final_out = None

        for i in range(input_ts.size(0)):
            # feed in one row of the sequence per time
            output_vb = self.circuit.forward(Variable(input_ts[i]))
            if final_out is None:
                final_out = output_vb
            else:
                final_out = torch.cat((final_out, output_vb), 0)

            # NOTE: this part is for examine the heads' weights and memory usage
            # NOTE: only used during testing, cos visualization takes time
            if self.mode == 2 and self.visualize:
                self.env.visual(input_ts[i,0,:].unsqueeze(0).unsqueeze(1),
                                self.target_vb.data[i,0,:].unsqueeze(0).unsqueeze(1),
                                self.mask_ts[i,0,:].unsqueeze(0).unsqueeze(1),
                                final_out.data[i,0,:].unsqueeze(0).unsqueeze(1))
                self.circuit.accessor.visual()
                raw_input()

        return final_out

    def _forward(self, observation, verbose=False):
        # first we need to reset all the necessary states
        self.circuit._reset_states()
        # NOTE: we update the output_vb and target_vb here
        input_ts = self._preprocessState(observation[0])
        self.target_vb = Variable(self._preprocessState(observation[1]))
        self.mask_ts   = self._preprocessState(observation[2]).expand_as(self.target_vb)

        # NOTE: maybe come up with something better than hard-wiring circuit name
        if self.circuit_params.circuit_type == "bidnc":
            self.output_vb = self.circuit.forward(input_ts)
        else:
            self.output_vb = self._forward_sequential(input_ts)

        if verbose:
            print("Input: \n{}".format(input_ts.squeeze()))
            print("Expected: \n{}".format(self.target_vb.squeeze()))
            print("Predicted: \n{}".format(self.output_vb.squeeze().round()))
            print("Mask: \n{}".format(self.mask_ts.squeeze()))

        if isinstance(self.env, Measurable):
            stats = self.env.measure_error(self.target_vb.permute(1, 0, 2), self.output_vb.permute(1, 0, 2), self.mask_ts.permute(1, 0, 2))
            self.total_samples += stats["total"]
            self.incorrect_samples += stats["errors"]

        if not self.training and self.visualize:
            self.env.visual(input_ts, self.target_vb.data, self.mask_ts, self.output_vb.data)

        return 0    # for all the supervised tasks we just return a 0 to keep the same format as rl

    def _reset_error_metrics(self):
        self.total_samples = 0
        self.incorrect_samples = 0

    def _backward(self):
        # TODO: we need to have a custom loss function to take mask into account
        # TODO: pass in this way might be too unefficient, but it's ok for now
        if self.training:
            self.optimizer.zero_grad()
        loss_vb = F.binary_cross_entropy(input=self.output_vb.transpose(0, 1).contiguous().view(1, -1),
                                         target=self.target_vb.transpose(0, 1).contiguous().view(1, -1),
                                         weight=self.mask_ts.transpose(0, 1).contiguous().view(1, -1))
        loss_vb /= self.batch_size
        if self.training:
            loss_vb.backward()
            self.optimizer.step()

        return loss_vb.data.item()

    def fit_model(self):    # the most basic control loop, to ease integration of new envs
        # self.optimizer = self.optim(self.circuit.parameters(), lr=self.lr)              # adam
        self.optimizer = self.optim(self.circuit.parameters(), lr=self.lr, eps=self.optim_eps, alpha=self.optim_alpha)   # rmsprop

        self.logger.warning("<===================================> Training ...")
        self.training = True
        self._reset_training_loggings()

        self.start_time = time.time()
        self.step = 0

        should_start_new = True
        self._reset_error_metrics()
        while self.step < self.steps:
            if should_start_new:
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                should_start_new = False
            action = self._forward(self.experience.state1)
            self.experience = self.env.step(action)
            if self.experience.terminal1 or self.early_stop and (episode_steps + 1) >= self.early_stop:
                should_start_new = True

            # calculate loss
            loss = self._backward()
            self.training_loss_avg_log.append([loss])

            self.step += 1

            # report training stats
            if self.step % self.prog_freq == 0:
                self.logger.warning("Reporting       @ Step: " + str(self.step) + " | Elapsed Time: " + str(time.time() - self.start_time))
                self.logger.warning("Training Stats:   avg_loss:         {}".format(np.mean(np.asarray(self.training_loss_avg_log))))
                if isinstance(self.env, Measurable):
                    self.log_metrics()

            # evaluation & checkpointing
            if self.step % self.eval_freq == 0:
                # Set states for evaluation
                self.training = False
                self.logger.warning("Evaluating      @ Step: " + str(self.step))
                self._eval_model()

                # Set states for resume training
                self.training = True
                self.logger.warning("Resume Training @ Step: " + str(self.step))
                should_start_new = True

        print("total time:", time.time() - self.start_time)

    def _eval_model(self):
        self.training = False

        eval_start_time = time.time()
        eval_step = 0

        eval_loss_avg_log = []
        eval_should_start_new = True
        self._reset_error_metrics()
        while eval_step < self.eval_steps:
            if eval_should_start_new:
                self._reset_experience()
                self.experience = self.env.reset_for_eval()
                assert self.experience.state1 is not None
                # if self.visualize: self.env.visual()
                # if self.render: self.env.render()
                eval_should_start_new = False
            eval_action = self._forward(self.experience.state1)
            self.experience = self.env.step_eval(eval_action)
            if self.experience.terminal1:# or self.early_stop and (episode_steps + 1) >= self.early_stop:
                eval_should_start_new = True

            # calculate loss
            eval_loss = self._backward()
            eval_loss_avg_log.append([eval_loss])

            eval_step += 1

        # Logging for this evaluation phase
        self.loss_avg_log.append([self.step, np.mean(np.asarray(eval_loss_avg_log))]); del eval_loss_avg_log
        # plotting
        if self.visualize:
            self.win_loss_avg = self.vis.scatter(X=np.array(self.loss_avg_log), env=self.refs, win=self.win_loss_avg, opts=dict(title="loss_avg"))
        # logging
        self.logger.warning("Evaluation        Took: " + str(time.time() - eval_start_time))
        self.logger.warning("Iteration: {}; loss_avg: {}".format(self.step, self.loss_avg_log[-1][1]))
        if isinstance(self.env, Measurable):
            self.log_metrics()

        # save model
        self._save_model(self.step, 0.) # TODO: here should pass in the negative loss

    def log_metrics(self):
        error_percentage = float(self.incorrect_samples) * 100 / self.total_samples
        self.logger.warning("Misprediction rate: {:.2f}%   Total Samples: {}    Mispredicted: {}".format(error_percentage, self.total_samples, self.incorrect_samples))

    def test_model(self):
        self.logger.warning("<===================================> Testing ...")
        self.training = False
        self._reset_testing_loggings()

        self.start_time = time.time()
        self.step = 0

        test_loss_avg_log = []
        test_should_start_new = True
        self._reset_error_metrics()
        while self.step < self.test_nepisodes:
            if test_should_start_new:
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                test_should_start_new = False
            test_action = self._forward(self.experience.state1)
            self.experience = self.env.step(test_action)
            if self.experience.terminal1:# or self.early_stop and (episode_steps + 1) >= self.early_stop:
                test_should_start_new = True

            # calculate loss
            test_loss = self._backward()
            test_loss_avg_log.append([test_loss])

            self.step += 1

        # Logging for this evaluation phase
        self.loss_avg_log.append([self.step, np.mean(np.asarray(test_loss_avg_log))]); del test_loss_avg_log
        # plotting
        if self.visualize:
            self.win_loss_avg = self.vis.scatter(X=np.array(self.loss_avg_log), env=self.refs, win=self.win_loss_avg, opts=dict(title="loss_avg"))
        # logging
        self.logger.warning("Testing  Took: " + str(time.time() - self.start_time))
        self.logger.warning("Iteration: {}; loss_avg: {}".format(self.step, self.loss_avg_log[-1][1]))
        if isinstance(self.env, Measurable):
            self.log_metrics()

    def explore_model(self):
        self.logger.warning("<===================================> Exploring ...")
        self.start_time = time.time()

        explored = 0
        self._reset_experience()
        self.experience = self.env.reset_for_eval()
        assert self.experience.state1 is not None



        while explored < self.test_nepisodes:
            eval_action = self._forward(self.experience.state1, verbose=True)
            self.experience = self.env.step_eval(eval_action)
            # calculate loss
            eval_loss = self._backward()
            explored += 1
