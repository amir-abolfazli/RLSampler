import os
import pickle
import random
import sys
import inspect
import pathlib
import numpy as np
import pycosa.modeling as modeling
import pandas as pd
import gym
from gym import spaces

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.get_embedding_AE import get_AE_embedding
from utils import get_coverage, combinations
#from utils.get_data_n import get_data, get_data_indices
from utils.get_data_n import get_data, get_data_indices

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
#matplotlib.use('Qt5Agg')


class ConfSamplingEnv(gym.Env, object):
    def __init__(self, fm_name, t, seed, render_mode, rl_model_name="RLSampler", mode='train'):
        """
        render_mode: True/False
        fm: feature model as a CNF formula or a DIMACS file
        t: parameter 't' for the t-wise sampler
        fm_name: feature model's name
        rl_name: RL method name (RL-based sampler)
        """

        self.start_time = time.time()
        self.render_mode = render_mode
        self.fm = None
        self.fm_name = fm_name
        self.mode = mode

        self.rl_model_name = rl_model_name
        self.t = t
        self.seed = seed

        self.path = f'./results_IC/{self.rl_model_name}/{self.fm_name}/t_{self.t}/fig'
        self.model_path  = f'{self.path}/../model_{self.fm_name}_t_{self.t}.net'
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)

        self.emb_size = 5
        self.import_fm(t=self.t)

        self.sample_labels = []
        self.ep_rewards_n = []
        self.ep_rewards_nn = []

        self.N_ep_rewards = []
        self.N_sample_sizes = []

        self.best_config = None
        self.protected_pairs_values = []
        self.prev_episode_difficult_pairs = {}
        self.prev_ts_difficult_pairs_N = {}
        self.current_unsatisfied_pairs_N = {}



        self.CL_nextState_pairs = {}
        self.coverage_CL = 0.0

        self.prev_size = 1000

        self.time_step = 0

        self.episode_time_step = 0


        self.minNumConfigs = 5
        self.best_c_sampleSize = 1000000
        self.coverage_vals = []
        self.reward_vals = []
        self.ep_rewards = []
        self.ep_rew = 0.0
        self.ep_id = 1
        self.eps = []
        self.c_ep_ts = 0


        self.first_episode = True

        self.init_subplots()

    def get_model_path(self):
        return self.model_path

    def import_fm(self, t):
        self.csv_path = (
            f"./twise_sampler_initial_solution/{self.fm_name}/{self.fm_name}_TWISE_sample_{t}.csv")

        self.dimacs_path = (
            f"./feature_models/{self.fm_name}/{self.fm_name}.dimacs")

        self.comb_file = (
            f"./feature_models/{self.fm_name}/comb/{self.fm_name}_{t}.comb")

        self.fm = modeling.CNFExpression()

        self.fm.from_dimacs(self.dimacs_path)

        self.original_data = pd.read_csv(self.csv_path)

        self.original_data.drop(columns=self.original_data.columns[0], axis=1, inplace=True)



        self.embedding = get_AE_embedding(self.original_data.values, emb_size=self.emb_size)

        self.action_space = spaces.Tuple((
            spaces.Discrete(2),
            spaces.Discrete(self.original_data.shape[1]),
        ))

        self.n_actions = self.original_data.shape[1]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.emb_size,))

        self.total_numConfigs = self.original_data.shape[1]


        self.n_configs = self.original_data.shape[1]
        self.n_features = self.original_data.shape[0]

    def set_t(self, t):
        self.t = t
        self.comb_file = (f"./feature_models/{self.fm_name}/comb/{self.fm_name}_{t}.comb")
        self.csv_path = (f"./feature_models/{self.fm_name}/{self.fm_name}_TWISE_sample_{t}.csv")
        self.import_fm(t)




    def get_embedding(self, indices):
        obs = self.embedding[indices, :].mean(axis=0).cpu().detach().numpy()
        obs = np.around(obs, 1)
        return obs


    def init_subplots(self):
        scale = 1  # scaling factor for the plot
        subplot_abs_width = 2 * scale  # Both the width and height of each subplot
        subplot_abs_spacing_width = 0.2 * scale  # The width of the spacing between subplots
        subplot_abs_excess_width = 0.3 * scale  # The width of the excess space on the left and right of the subplots
        subplot_abs_excess_height = 0.3 * scale  # The height of the excess space on the top and bottom of the subplots

        # for cols in range(num_subplots):
        fig_width = (2 * subplot_abs_width) + ((2 - 1) * subplot_abs_spacing_width) + subplot_abs_excess_width
        fig_height = subplot_abs_width + subplot_abs_excess_height

        self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width * 3, fig_height * 3))

        self.axes[0, 0].set_xlim([0, self.total_numConfigs])
        self.axes[0, 1].set_xlim([0, self.total_numConfigs])
        # self.axes[1, 0].set_xlim([0, 30])
        # self.axes[1, 1].set_xlim([0, 30])
        self.axes[0, 0].set_ylim([0, self.n_features - 1])
        self.axes[0, 0].invert_yaxis()
        self.axes[0, 1].invert_yaxis()
        self.axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[0, 1].set_ylim([0, self.n_features - 1])
        self.axes[1, 0].set_ylim([0, 1])
        self.axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

        self.axes[1, 1].set_xticks([0, len(self.ep_rewards)])
        # self.axes[1, 1].set_ylim([0, 30])

        self.axes[0, 0].set_title("Sample")
        self.axes[0, 0].set_xlabel("Configurations")
        self.axes[0, 0].set_ylabel("Features")

        self.axes[0, 1].set_title("Valid Configurations")
        self.axes[0, 1].set_xlabel("Configurations")
        self.axes[0, 1].set_ylabel("Features")

        self.axes[1, 0].set_xlabel("Time-step")
        self.axes[1, 0].set_ylabel("Pairwise Coverage")

        self.axes[1, 1].set_xlabel("Episode")
        self.axes[1, 1].set_ylabel("Average Reward")


    def set_episode_info(self, ep_id, ts):
        self.ep_id = ep_id
        self.ts = ts
        self.time_step +=1

    def get_rank(self):
        if self.sample.shape[1] > 0:
            rank = np.linalg.matrix_rank(self.sample)
        else:
            rank = 0
        return rank

    def get_coverage(self):
        if self.sample.shape[0] > 0:
            coverage, comb_dict = get_coverage.check_coverage(self.sample.T.tolist(), self.comb_file, self.fm.clauses)
            return coverage, comb_dict
        else:
            return 0.0, {}

    def get_coverage_full(self):
        if self.sample.shape[0] > 0:
            coverage, comb_dict, comb_config_dict = get_coverage.check_coverage_full(self.sample.T.tolist(), self.comb_file, self.fm.clauses)
            return coverage, comb_dict, comb_config_dict
        else:
            return 0.0, {}

    def get_coverage_conf_sample(self, conf):
        if self.sample.shape[0] > 0:
            coverage, comb_dict = get_coverage.check_coverage_smaple(conf.T.tolist(), self.comb_file, self.fm.clauses)
            return coverage, comb_dict
        else:
            return 0.0, {}

    def get_coverage_conf_sample_pairs(self, conf):
        if self.sample.shape[0] > 0:
            coverage, comb_dict = get_coverage.check_coverage_smaple_allpairs(conf.T.tolist(), self.comb_file, self.fm.clauses)
            return coverage, comb_dict
        else:
            return 0.0, {}


    def get_coverage_list(self, conf, pairs):
        if self.sample.shape[0] > 0:
            coverage, comb_dict = get_coverage.check_coverage_list(conf.T.tolist(), pairs, self.fm.clauses)
            return coverage, comb_dict
        else:
            return 0.0, {}
    def get_coverage_sample_pairsList(self, pairs):
        if self.sample.shape[0] > 0:
            coverage, comb_dict = get_coverage.coverage_sample_pairsList(self.sample.T.tolist(), pairs, self.fm.clauses)
            return coverage, comb_dict
        else:
            return 0.0, {}


    def sum_values(self, d):
        return int(sum(d.values()))

    def num_different_pairs(self, dict1, dict2):
        set1 = set(dict1.items())
        set2 = set(dict2.items())
        num = len(set1 ^ set2) // 2
        return num

    def get_satisfied_pairs(self, dict1):
        s = {k: v for (k, v) in dict1.items() if v == 1}
        s_size = len(s)
        return s, s_size

    def get_unsatisfied_pairs(self, dict1):
        s = {k: v for (k, v) in dict1.items() if v == 0}
        s_size = len(s)
        return s, s_size

    def get_satisfied_pairs_keys(self, dict1):
        s = {k for (k, v) in dict1.items() if v == 1}
        n_pairs = len(s)
        return s
    def get_unsatisfied_pairs_keys(self, dict1):
        s = {k for (k, v) in dict1.items() if v == 0}
        n_pairs = len(s)
        return s

    def get_difficult_pairs_keys(self, dict1, threshold):
        s = {k for (k, v) in dict1.items() if v < threshold}
        n_pairs = len(s)
        return s, n_pairs


    def normalize_reward(self, data):

        return (data - np.min(data)) / (np.max(data) - np.min(data) + 0.02)

    def normalize_array(self, array):
        """
        Min-Max normalization of the given array
        :param array: (np.ndarray) Data array
        :returns: (np.ndarray) Normalized array
        """
        max_value = np.max(array)
        min_value = np.min(array)
        val = np.array([0.0]) if (max_value - min_value) == 0.0 else (array - min_value) / (max_value - min_value)

        return val * 2 - 1


    def terminal(self):
        """
        Returns whether the current state is terminal
        :return:
        """
        return True if (self.get_coverage()[0] == 1.0) else False


    def step(self, action):
        """
        Implements the step function the environment given an action tuple a(a_1, a_2)
        The list of transitions is returned afterwards.
        Each transition is a tuple(next_state, reward, done, info).
        Currently, info is an empty dict
        :param state:
        :param action:
        :return: list of transitions: tuple(next_state, reward, done, info)
        """
        # if action is None:


        self.episode_time_step +=1

        action_1 = int(action[0])
        action_2 = int(action[1])

        last_node = False

        final_reward = 0.0

        coverage, dd = self.get_coverage()

        if int(coverage) == 1 and self.sample.shape[1] >= self.best_c_sampleSize:
            rnd_action = random.choice(self.sample_labels)
            idx = self.sample_labels.index(rnd_action)

            _, current_pairs1 = self.get_coverage()
            all_satisfied_pairs1 = self.get_satisfied_pairs_keys(current_pairs1)

            self.sample = np.delete(self.sample, idx, axis=1)
            self.sample_labels.remove(rnd_action)

            _, current_pairs2 = self.get_coverage()

            all_satisfied_pairs2 = self.get_satisfied_pairs_keys(current_pairs2)

            if len(all_satisfied_pairs2) == len(all_satisfied_pairs1):
                final_reward = 1.0
            else:
                final_reward = -1.0
                k = self.possible_configs[:, action_2][..., np.newaxis]

                self.sample = np.hstack((self.sample, k))
                self.sample_labels.append(action_2)



        else:

            if action_1 == 0:  # add
                if action_2 not in self.sample_labels:

                    _, current_pairs = self.get_coverage()

                    cov_sample, pairs_N = self.get_coverage_conf_sample(self.possible_configs[:, action_2])

                    all_satisfied_pairs = self.get_satisfied_pairs_keys(pairs_N)

                    res = any(ele in self.get_unsatisfied_pairs_keys(current_pairs) for ele in all_satisfied_pairs)

                    if res:
                        k = self.possible_configs[:, action_2][..., np.newaxis]

                        self.sample = np.hstack((self.sample, k))
                        self.sample_labels.append(action_2)

                        final_reward = 1.0
                    else:
                        final_reward = -1.0
                else:
                    final_reward = 0.0



            elif action_1 == 1:  # remove

                if (action_2 in self.sample_labels):
                    _, current_pairs1 = self.get_coverage()

                    #
                    all_satisfied_pairs1 = self.get_satisfied_pairs_keys(current_pairs1)
                    idx = self.sample_labels.index(action_2)
                    self.sample = np.delete(self.sample, idx, axis=1)
                    self.sample_labels.remove(action_2)

                    _, current_pairs2 = self.get_coverage()

                    all_satisfied_pairs2 = self.get_satisfied_pairs_keys(current_pairs2)

                    if len(all_satisfied_pairs2) == len(all_satisfied_pairs1):
                        final_reward = 1.0
                    else:
                        final_reward = -1.0
                        k = self.possible_configs[:, action_2][..., np.newaxis]

                        self.sample = np.hstack((self.sample, k))
                        self.sample_labels.append(action_2)
                else:
                    final_reward = 0.0
        # if action_1 == 0: # add
        #     if action_2 not in self.sample_labels:
        #
        #         _, current_pairs = self.get_coverage()
        #
        #         cov_sample, pairs_N = self.get_coverage_conf_sample(self.possible_configs[:, action_2])
        #         all_satisfied_pairs = self.get_satisfied_pairs_keys(pairs_N)
        #
        #         res = any(ele in self.get_unsatisfied_pairs_keys(current_pairs) for ele in all_satisfied_pairs)
        #
        #         if res:
        #             k = self.possible_configs[:, action_2][..., np.newaxis]
        #
        #             self.sample = np.hstack((self.sample, k))
        #             self.sample_labels.append(action_2)
        #
        #             final_reward = 1.0
        #         else:
        #             final_reward = -1.0
        #     else:
        #         final_reward = 0.0
        #
        #
        #
        # elif action_1 == 1: # remove
        #
        #     if (action_2 in self.sample_labels):
        #         _, current_pairs1 = self.get_coverage()
        #
        #         all_satisfied_pairs1 = self.get_satisfied_pairs_keys(current_pairs1)
        #         idx = self.sample_labels.index(action_2)
        #         self.sample = np.delete(self.sample, idx, axis=1)
        #         self.sample_labels.remove(action_2)
        #
        #         _, current_pairs2 = self.get_coverage()
        #
        #         all_satisfied_pairs2 = self.get_satisfied_pairs_keys(current_pairs2)
        #
        #         if len(all_satisfied_pairs2) == len(all_satisfied_pairs1):
        #             final_reward = 1.0
        #         else:
        #             final_reward = -1.0
        #             k = self.possible_configs[:, action_2][..., np.newaxis]
        #
        #             self.sample = np.hstack((self.sample, k))
        #             self.sample_labels.append(action_2)
        #     else:
        #         final_reward = 0.0


        current_coverage, current_comb_dict = self.get_coverage()
        current_size = self.sample.shape[1]

        current_pairs, u_s_size  = self.get_unsatisfied_pairs(current_comb_dict)

        _, current_pairs = self.get_coverage()

        reward = final_reward


        self.coverage_vals.append(current_coverage)

        done = self.terminal()

        if self.episode_time_step % 20 == 0:
            c, n = self.get_coverage()
            print(
                f"Feature Model: {self.fm_name} | t={self.t} | Episode {self.ep_id} | Sample Size: {self.sample.shape[1]} | Coverage: {c:.0%}")

        self.ep_rew += reward

        if done:

            c, n = self.get_coverage()
            print(f"Feature Model: {self.fm_name} | t={self.t} | Seed: {self.seed} | Episode {self.ep_id} finished | Sample Size: {self.sample.shape[1]} | Coverage: {c:.0%} \n Best Size: {self.best_c_sampleSize}")
            self.episode_time_step = 0


            current_coverage, current_comb_dict = self.get_coverage()

            self.prev_size = self.sample.shape[1]

            current_coverage_full, current_comb_dict_full, comb_config_dict = self.get_coverage_full()

            comb_config_dict2 = {key: [self.sample_labels[int(v)] for v in value] for key, value in
                                 comb_config_dict.items()}

            min_val = max(1, min([len(ele) for k, ele in comb_config_dict2.items()]))

            # print(f"Min Val: {min_val}")

            if self.ep_id == 1:
                result = {key: value for key, value in comb_config_dict2.items() if
                          len(value) <= min_val+2}
            else:
                result = {key: value for key, value in comb_config_dict2.items() if
                      len(value) == min_val}



                self.N_sample_sizes.append(self.sample.shape[1])
                self.N_ep_rewards.append(self.ep_rew)

                with open(f'{self.path}/../{self.fm_name}_t_{self.t}_sampleSize_seed_{self.seed}.npy', 'wb') as f1:
                    pickle.dump(self.N_sample_sizes, f1)

                with open(f'{self.path}/../{self.fm_name}_t_{self.t}_epRewards_seed_{self.seed}.npy', 'wb') as f2:
                    pickle.dump(self.N_ep_rewards, f2)



            self.next_episode_initial_state = sorted(set([item for sublist in result.values() for item in sublist]))


            if (current_size < self.best_c_sampleSize) and (int(current_coverage) == 1):


                self.best_c_sampleSize = current_size
                self.save_best_results()

            self.ep_rew = 0.0
            self.ep_id += 1


        state = self.get_embedding(self.sample_labels)
        state = np.around(state, 1)

        cov_sample, pairs_N = self.get_coverage_sample_pairsList(list(self.CL_nextState_pairs))
        d_c = 1 if any(ele in self.CL_nextState_pairs for ele in pairs_N) else 0

        return state, reward, done, {"n_uc": u_s_size,
                                     "difficult_clause":d_c,
                                     "last_node": last_node}

    def sample(self):
        pass

    def save_best_results(self):
        self.save_plot(self.ep_id)
        pd.DataFrame(self.sample).to_csv(f'{self.path}/../{self.fm_name}_t_{self.t}_best_sample_seed_{self.seed}.csv', index=False)


    def reset(self):
        self.sample_labels = []
        # mode == 'train'
        if self.mode == 'train':
            if self.first_episode:
                self.sample, self.possible_configs, indices = get_data(self.original_data)

                self.sample_labels.extend(indices)
                self.first_episode = False
            else:
                self.sample, self.possible_configs = get_data_indices(self.original_data, self.next_episode_initial_state)
                self.sample_labels.extend(self.next_episode_initial_state)
        # mode == 'eval'
        else:
            if self.first_episode:
                indices = [0]
                self.sample, self.possible_configs = get_data_indices(self.original_data, indices)

                self.sample_labels.extend(indices)
                self.first_episode = False
            else:
                self.sample, self.possible_configs = get_data_indices(self.original_data,
                                                                      self.next_episode_initial_state)
                self.sample_labels.extend(self.next_episode_initial_state)

        self.ep_rewards_n = []
        self.ep_rewards_nn = []
        self.trajectories = []

        self.time_step = 0

        self.coverage_vals = []
        self.reward_vals = []


        self.axes[0, 0].cla()
        self.axes[0, 1].cla()
        self.axes[1, 0].cla()

        state = self.get_embedding(self.sample_labels)
        state = np.around(state, 1)


        return state

    def render(self, render_mode='human'):
        self.axes[0, 0].cla()
        self.axes[0, 1].cla()
        self.axes[1, 0].cla()
        self.axes[0, 0].set_xlim([0, self.total_numConfigs - 1])
        self.axes[0, 1].set_xlim([0, self.total_numConfigs - 1])
        self.axes[0, 0].set_ylim([0, self.n_features - 1])
        self.axes[0, 1].set_ylim([0, self.n_features - 1])
        self.axes[0, 0].invert_yaxis()
        self.axes[0, 1].invert_yaxis()
        self.axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

        self.axes[0, 0].set_title("Sample")
        self.axes[0, 0].set_xlabel("Configurations")
        self.axes[0, 0].set_ylabel("Features")

        self.axes[0, 1].set_title("Valid Configurations")
        self.axes[0, 1].set_xlabel("Configurations")
        self.axes[0, 1].set_ylabel("Features")

        self.axes[1, 0].set_xlabel("Time-step")
        self.axes[1, 0].set_ylabel("Pairwise Coverage")

        self.axes[1, 1].set_xlabel("Episode")
        self.axes[1, 1].set_ylabel("Average Reward")
        self.axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

        if len(self.ep_rewards) > 0:
            self.axes[1, 1].set_xticks(list(range(1, len(self.ep_rewards) + 1)))
            self.axes[1, 1].set_xscale('linear')
            self.axes[1, 1].set_xlim(left=0, right=self.ep_id)
            self.axes[1, 1].set_ylim(bottom=min(self.ep_rewards) - 1, top=max(self.ep_rewards) + 1)
            im4 = self.axes[1, 1].plot(list(range(1, len(self.ep_rewards) + 1)), self.ep_rewards, 'k')
        plt.suptitle(f"Episode: {self.ep_id} | Sample Size: {self.best_c_sampleSize}", fontsize=10)

        self.axes[1, 0].set_ylim([0, 1.1])

        clist = [(0, "red"), (1, "blue")]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)
        im1 = self.axes[0, 0].imshow(self.sample, cmap=cmap)
        im2 = self.axes[0, 1].imshow(self.possible_configs, cmap=cmap)
        im3 = self.axes[1, 0].plot(list(range(len(self.coverage_vals))), self.coverage_vals, 'k')

        plt.pause(0.001)

    def save_plot(self, ep_id):
        self.axes[0, 0].cla()
        self.axes[0, 1].cla()
        self.axes[1, 0].cla()
        self.axes[0, 0].set_xlim([0, self.total_numConfigs - 1])
        self.axes[0, 1].set_xlim([0, self.total_numConfigs - 1])
        self.axes[0, 0].set_ylim([0, self.n_features - 1])
        self.axes[0, 1].set_ylim([0, self.n_features - 1])
        self.axes[0, 0].invert_yaxis()
        self.axes[0, 1].invert_yaxis()
        self.axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

        self.axes[0, 0].set_title("Sample")
        self.axes[0, 0].set_xlabel("Configurations")
        self.axes[0, 0].set_ylabel("Features")

        self.axes[0, 1].set_title("Valid Configurations")
        self.axes[0, 1].set_xlabel("Configurations")
        self.axes[0, 1].set_ylabel("Features")

        self.axes[1, 0].set_xlabel("Time-step")
        self.axes[1, 0].set_ylabel("Pairwise Coverage")

        self.axes[1, 1].set_xlabel("Episode")
        self.axes[1, 1].set_ylabel("Average Reward")
        self.axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

        if len(self.ep_rewards) > 0:
            self.axes[1, 1].set_xticks(list(range(1, len(self.ep_rewards) + 1)))
            self.axes[1, 1].set_xscale('linear')
            self.axes[1, 1].set_xlim(left=0, right=self.ep_id)
            self.axes[1, 1].set_ylim(bottom=min(self.ep_rewards) - 1, top=max(self.ep_rewards) + 1)
            im4 = self.axes[1, 1].plot(list(range(1, len(self.ep_rewards) + 1)), self.ep_rewards, 'k')

        plt.suptitle(f"Episode: {self.ep_id} | Sample Size: {self.best_c_sampleSize}", fontsize=10)

        self.axes[1, 0].set_ylim([0, 1.1])

        clist = [(0, "red"), (1, "blue")]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)

        im1 = self.axes[0, 0].imshow(self.sample, cmap=cmap)

        im2 = self.axes[0, 1].imshow(self.possible_configs, cmap=cmap)
        im3 = self.axes[1, 0].plot(list(range(len(self.coverage_vals))), self.coverage_vals, 'k')

        self.fig.savefig(f'{self.path}/{self.fm_name}_t_{self.t}_seed_{self.seed}_episode_{ep_id}.png', format="png")

    def close(self):
        raise NotImplementedError
