import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import os
import csv
from agents.multi_agent import MultiAgent
from agents.replay_buffer import ReplayBuffer
from agents.episode_replay_buffer import EpisodeReplayBuffer
from agents.simple_agent import RandomAgent as NonLearningAgent
from agents.simple_agent import StaticAgent as StAgent
from agents.simple_agent import ActiveAgent as AcAgent
from agents.evaluation import Evaluation
from envs.gui import canvas


class Trainer(object):
    def __init__(self, args, env):
        # Device
        self.device = args.device

        # Environment
        self.env = env
        self.eval = Evaluation()
        self.n_predator = args.n_predator
        self.n_prey = args.n_prey
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.replay_buffer = EpisodeReplayBuffer(args)

        # Predator & Prey agents
        self.n_predator_agent = MultiAgent(args)
        self.prey_agent = AcAgent(args, action_dim=5)

        # Exploration parameters
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = (args.epsilon - args.min_epsilon) / args.epsilon_decay_steps

        # Training parameters
        self.batch_size = args.batch_size
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.training_start = args.training_start
        self.max_step = args.max_step
        self.training_step = args.training_step
        # self.eval_interval_episode = args.eval_interval_episode
        self.evaluate_times = args.evaluate_times
        self.evaluate_freq = args.evaluate_freq
        self.testing_step = args.testing_step
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.n_predator_agent.optimizer, step_size=args.step_size,
                                                            gamma=args.gamma)
        self.algorithm = args.mixing_network
        self.scenario = args.scenario
        self.gui = args.gui
        self.keyboard_input = args.keyboard_input
        self.penalty = args.penalty

        if args.load_nn:
            self.epsilon = self.min_epsilon

        if args.gui:
            self.canvas = canvas.Canvas(self.n_predator, self.n_prey, args.map_size)
            self.canvas.setup()

        # Tensorboard
        self.log_path = "runs/{}_penalty_{}_{}_{}".format(args.scenario, args.penalty, args.n_predator, args.n_prey)
        self.eval_log = []
        self.seed = args.seed
        if not args.use_random_update:
            self.log_path += "_sequential"
        if args.use_orthogonal_init:
            self.log_path += "_orthogonal"
        self.writer = SummaryWriter(log_dir=self.log_path)

    def learn(self):
        total_steps = 0
        episode_num = 0
        training_pbar = tqdm(initial=0, desc="Training", total=self.training_step, unit="total_steps")
        evaluate_num = -1

        while total_steps < self.training_step:
            episode_num += 1
            episode_step = 0
            obs = self.env.reset()

            # Extract predator information
            obs_n = obs[:self.n_predator]
            state = self.env.get_full_encoding()[:, :, 2]  # state.shape = (args.map_size, args.map_size)
            s = self.state_to_index(state)

            total_reward = 0
            total_reward_pos = 0
            total_reward_neg = 0

            # Initialize last onehot action
            last_onehot_a_n = np.zeros((self.n_predator, self.action_dim))
            # If use RNN, before the beginning of each episode，reset the hidden state of the DRQN
            h_state = torch.zeros([self.n_predator, self.rnn_hidden_dim]).to(self.device)

            while True:
                total_steps += 1
                training_pbar.update(1)
                episode_step += 1

                a_n, h_state, prey_a_n = self.n_predator_agent.choose_action(h_state, obs_n, last_onehot_a_n,
                                                                             self.epsilon, state, self.prey_agent)
                last_onehot_a_n = self.action_to_onehot(a_n)
                obs_prime, reward, done_n, _ = self.env.step(np.append(a_n, prey_a_n))

                # Extract predator information
                obs_n_prime = obs_prime[:self.n_predator]
                reward_n = reward[:self.n_predator]
                r = np.sum(reward_n)
                state_prime = self.env.get_full_encoding()[:, :, 2]
                s_prime = self.state_to_index(state_prime)
                done = sum(done_n) > 0

                self.replay_buffer.store_transition(episode_step - 1, obs_n, s, last_onehot_a_n, a_n, r, done)

                obs_n = obs_n_prime
                state = state_prime
                s = s_prime

                total_reward += r
                if r >= 0:
                    total_reward_pos += r
                else:
                    total_reward_neg -= r

                if self.is_episode_done(done, total_steps) or episode_step >= self.max_step:
                    self.replay_buffer.store_last_step(episode_step, obs_n, s)
                    break

                # Start training after enough amount of replay buffer
                if len(self.replay_buffer) >= self.training_start:
                    loss = self.n_predator_agent.train(self.replay_buffer)
                    self.writer.add_scalar("loss", loss, global_step=total_steps)

                # Epsilon Decaying
                self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)

            self.writer.add_scalar("score", total_reward, global_step=episode_num)
            # print(">>> episode: {}, total reward: {}, pos reward: {}, neg reward: {}".format(episode_num,
            # total_reward, total_reward_pos, total_reward_neg))

            # if episode_num % self.eval_interval_episode == 0:
            if total_steps // self.evaluate_freq > evaluate_num:
                self.test(total_steps, episode_num)
                evaluate_num += 1
                print(">>> current epsilon: {:.2f}%".format(self.epsilon * 100))

        self.test(total_steps, episode_num)
        training_pbar.close()
        self.eval.summarize()

        self.writer.flush()
        self.writer.close()

    def test(self, train_step=None, curr_ep=None):
        step = 0
        episode_num = 0
        test_flag = self.keyboard_input

        # Episode running average
        avg_reward = 0
        avg_reward_pos = 0
        avg_reward_neg = 0

        # while step < self.testing_step:
        while episode_num < self.evaluate_times:
            episode_num += 1
            episode_step = 0
            obs = self.env.reset()
            # Extract predator information
            obs_n = obs[:self.n_predator]
            state = self.env.get_full_encoding()[:, :, 2]

            total_reward = 0
            total_reward_pos = 0
            total_reward_neg = 0

            # initialize last onehot action
            last_onehot_a_n = np.zeros((self.n_predator, self.action_dim))
            # If use RNN, before the beginning of each episode，reset the hidden state of the DRQN
            h_state = torch.zeros([self.n_predator, self.rnn_hidden_dim]).to(self.device)

            if test_flag:
                print("\nInitialize\n", state)

            while True:
                step += 1
                episode_step += 1

                a_n, h_state, prey_a_n = self.n_predator_agent.choose_action(h_state, obs_n, last_onehot_a_n,
                                                                             self.epsilon, state, self.prey_agent,
                                                                             train=False)
                last_onehot_a_n = self.action_to_onehot(a_n)
                obs_prime, reward, done_n, info_n = self.env.step(np.append(a_n, prey_a_n))
                # Extract predator information
                obs_n_prime = obs_prime[:self.n_predator]
                reward_n = reward[:self.n_predator]
                r = np.sum(reward_n)
                state_prime = self.env.get_full_encoding()[:, :, 2]
                s_prime = self.state_to_index(state_prime)
                done = sum(done_n) > 0

                if self.gui:
                    self.canvas.draw(s_prime, done_n, "Score:" + str(total_reward) + ", Step:" + str(episode_step))

                if test_flag:
                    aa = input('>')
                    if aa == 'c':
                        test_flag = False
                    print(a_n)
                    print(state_prime)
                    print(a_n)

                obs_n = obs_n_prime
                state = state_prime

                # if r == 0.1:
                #     r = r * (-1.) * self.penalty
                total_reward += r  # * (args.df ** (episode_step-1))
                if r > 0:
                    total_reward_pos += r
                else:
                    total_reward_neg -= r

                if self.is_episode_done(done, step, "test") or episode_step >= self.max_step:
                    if self.gui:
                        self.canvas.draw(s_prime, done_n, "Hello",
                                         "Score:" + str(total_reward) + ", Step:" + str(episode_step))
                    break

            # Compute running average reward along episode
            avg_reward += total_reward
            avg_reward_pos += total_reward_pos
            avg_reward_neg += total_reward_neg

        if self.scenario == "pursuit":
            eval_metric = float(step) / episode_num
        else:
            eval_metric = avg_reward / episode_num

        curr_ep = curr_ep if curr_ep is not None else episode_num

        self.eval_log.append({
            "steps": train_step,
            "reward": eval_metric,
        })

        np.save(os.path.join(self.log_path, f"{self.algorithm}_seed_{self.seed}.npy"), np.array(self.eval_log))

        if self.writer is not None:
            self.writer.add_scalar("eval/metric", eval_metric, global_step=curr_ep)

        if self.scenario == "pursuit":
            print("Test result (average steps to capture): train {} episodes, average: {} steps".format(curr_ep, float(
                step) / episode_num))
            self.eval.update_value("Training result: ", float(step) / episode_num, curr_ep)
        elif self.scenario == "endless" or self.scenario == "endless2" or self.scenario == "endless3":
            print(
                "Test result: train {} with penatly {}, average reward: {}, average positive reward: {}, average negative reward: {}"
                .format(curr_ep, self.penalty, avg_reward / episode_num, avg_reward_pos / episode_num,
                        avg_reward_neg / episode_num))
            self.eval.update_value("Training result: ", avg_reward / episode_num, curr_ep)

    def is_episode_done(self, done, step, episode_type="train"):
        if episode_type == "test":
            if done > 0 or step >= self.testing_step:
                return True
            else:
                return False
        else:
            if done > 0 or step >= self.training_step:
                return True
            else:
                return False

    def state_to_index(self, state):
        s = np.zeros(2 * (self.n_predator + self.n_prey))
        for i in range(self.n_predator + self.n_prey):
            p = np.argwhere(np.array(state) == i + 1)[0]
            #p = self.get_pos_by_id(state, i+1)
            s[2 * i] = p[0]
            s[2 * i + 1] = p[1]

        return s

    # Apply onehot-encoding to joint actions
    def action_to_onehot(self, action):
        onehot_action = np.eye(self.action_dim)[action]
        return onehot_action

    # -------------------------- Not used -------------------------- #
    def random_action_generator(self):
        # Generate each probability of action by uniform random distribution
        action_prob = np.random.uniform(size=(self.n_predator, 5))
        # Normalize
        self.rand_action_prob = action_prob / np.sum(action_prob, axis=1, keepdims=True)

    def onehot_to_action(self, onehot_action):
        action = np.zeros([self.n_predator])
        for i in range(self.n_predator):
            action[i] = int(np.argmax(onehot_action[i]))
        return action

    def index_to_action(self, index):
        action_list = []
        for i in range(self.n_predator - 1):
            action_list.append(index % 5)
            index = index / 5
        action_list.append(index)
        return action_list

    def action_to_index(self, action):
        index = 0
        for i in range(self.n_predator):
            index += action[i] * 5 ** i
        return index
