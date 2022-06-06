"""
Automatic Cryptocurrency trading using Deep RL
Nick Kaparinos
2022
"""

import gym
import numpy as np
from os import makedirs
import pandas as pd
import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import save_list_to_txt


class TradeEnv(gym.Env):
    """ Crypto trading environment """
    reward_range = (-float('inf'), float('inf'))
    metadata = {'render.modes': None}

    def __init__(self, crypto_files=(), timeseries_step='m', test=False, n_previous_timesteps=5, max_episode_steps=500):
        super().__init__()
        self.n_timeseries = len(crypto_files)
        self.n_features = 6
        self.n_previous_timesteps = n_previous_timesteps
        self.observation_space = gym.spaces.Discrete(
            (1 + self.n_previous_timesteps) * (self.n_features * self.n_timeseries) + self.n_timeseries + 1)
        self.action_space = gym.spaces.Discrete(self.n_timeseries + 1)
        self.max_episode_steps = max_episode_steps
        self.current_step = None
        self.episode_starting_step = None
        self.timeseries_step = timeseries_step

        self.starting_balance = 1_000
        self.balance = self.starting_balance

        self.portfolio = np.zeros(self.n_timeseries)
        self.fee = 0.001

        # Read
        self.crypto_names = [crypto[5:8] for crypto in crypto_files]
        cryptos = [pd.read_hdf(crypto).resample(timeseries_step).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}) for crypto in tqdm(crypto_files)]
        common_start, common_end = self._find_common_timespan(cryptos)
        for i in range(len(cryptos)):
            cryptos[i] = cryptos[i][common_start <= cryptos[i].index]
            cryptos[i] = cryptos[i][cryptos[i].index <= common_end]

        self.data = pd.DataFrame()
        for i in range(self.n_timeseries):
            cryptos[i]['month'] = cryptos[i].index.month
            cryptos[i]['day'] = cryptos[i].index.day
            cryptos[i]['hour'] = cryptos[i].index.hour
            cryptos[i].columns = [f'open{i}', f'high{i}', f'low{i}', f'close{i}', f'month{i}', f'day{i}', f'hour{i}']
            self.data = pd.concat([self.data, cryptos[i]], axis=1)
        self.data = self.data.interpolate()

        # Split train test
        open_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                'open' in i]
        n = self.data.shape[0]
        self.means = []
        self.stds = []
        for idx, i in enumerate(open_columns_numbers):
            self.means.append(self.data.iloc[int(n * 0.8):, i].mean())
            self.stds.append(self.data.iloc[int(n * 0.8):, i].std())
        if test:
            self.data = self.data.iloc[int(n * 0.8):]
        else:
            self.data = self.data.iloc[:int(n * 0.8)]

    def step(self, action):
        open_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                'open' in i]
        close_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                 'close' in i]
        open_stock_prices = self.data.iloc[
            self.episode_starting_step + self.current_step + 1, open_columns_numbers].values
        close_stock_prices = self.data.iloc[
            self.episode_starting_step + self.current_step + 1, close_columns_numbers].values
        episode_starting_portfolio_value = self.balance + np.inner(self.portfolio, open_stock_prices)

        if action == self.action_space.n - 1:  # sell all
            self.balance += np.inner(self.portfolio, open_stock_prices)
            self.portfolio = np.zeros(self.n_timeseries)
        elif self.portfolio[action] == 0 or self.balance != 0:  # Sell portfolio and buy, otherwise hold
            self.balance += np.inner(self.portfolio, open_stock_prices)
            self.portfolio = np.zeros(self.n_timeseries)
            stock_bought = self.balance / (open_stock_prices[action] * (1 + self.fee))
            self.portfolio[action] = stock_bought
            self.balance = 0

        self.current_step += 1
        if self.current_step == self.max_episode_steps:
            done = True
            obs = np.zeros(self.observation_space.n)
        else:
            done = False
            obs = self._get_obs(open_columns_numbers)

        episode_ending_portfolio_value = self.balance + np.inner(self.portfolio, close_stock_prices)
        reward = episode_ending_portfolio_value - episode_starting_portfolio_value

        return obs, reward, done, {}

    def seed(self, seed=None):
        return seed

    def reset(self):
        self.episode_starting_step = random.randint(self.n_previous_timesteps,
                                                    self.data.shape[0] - self.max_episode_steps - 1)
        self.portfolio = [0] * self.n_timeseries
        self.balance = self.starting_balance
        self.current_step = 0

        open_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                'open' in i]
        obs = self._get_obs(open_columns_numbers)
        return obs

    def _get_obs(self, open_columns_numbers):
        """ Get observation """
        all_feature_columns = list(set([i for i in range(self.data.shape[1])]) - set(open_columns_numbers))
        obs = np.empty((0, 0), float)

        for i in range(self.n_timeseries):
            feature_columns = [column for column in all_feature_columns if str(i) in self.data.columns[column]]
            timeseries_obs = self.data.iloc[
                             self.episode_starting_step + self.current_step - self.n_previous_timesteps:
                             self.episode_starting_step + self.current_step + 1,
                             feature_columns].values
            # Standardisation
            timeseries_obs[:, :3] /= 2 * self.means[i]
            # timeseries_obs[:, :3] = (timeseries_obs[:, :3] - self.means[i])/self.stds[i]
            timeseries_obs[:, 3] /= 12
            timeseries_obs[:, 3] /= 31
            timeseries_obs[:, 3] /= 24
            if obs.shape[0] == 0:
                obs = timeseries_obs.copy()
            else:
                obs = np.hstack([obs, timeseries_obs])
        portfolio_state1 = np.array(self.portfolio) > 0
        portfolio_state2 = np.array([np.logical_not(portfolio_state1.sum())])
        portfolio_state = np.concatenate((portfolio_state1, portfolio_state2)) * 1
        obs = np.concatenate([obs.ravel(), portfolio_state])
        return obs

    @staticmethod
    def _find_common_timespan(timeseries):
        starts = [i.index[0] for i in timeseries]
        ends = [i.index[-1] for i in timeseries]
        return max(starts), min(ends)


class TestTradeEnv(TradeEnv):
    """ Test environment for cryptocurrency trading
    Includes episode and epoch performance visualisations over base environment
    """

    def __init__(self, log_dir, num_test_episodes, **kwargs):
        super().__init__(test=True, **kwargs)
        self.log_dir = log_dir
        self.epoch = -1
        self.episode = num_test_episodes
        self.num_test_episodes = num_test_episodes

        self.actions_chosen = dict()
        self.reward_per_timeseries = np.zeros(self.n_timeseries)
        self.portfolio_value_list = []
        self.reward_list = []
        self.episode_ending_portfolio_value_list = []
        self.epoch_distributions = []
        self.epoch_versus_bah_distributions = []

    def step(self, action):
        open_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                'open' in i]
        close_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                 'close' in i]
        open_stock_prices = self.data.iloc[
            self.episode_starting_step + self.current_step + 1, open_columns_numbers].values
        close_stock_prices = self.data.iloc[
            self.episode_starting_step + self.current_step + 1, close_columns_numbers].values
        episode_starting_portfolio_value = self.balance + np.inner(self.portfolio, open_stock_prices)

        if action == self.action_space.n - 1:  # sell all
            self.balance += np.inner(self.portfolio, open_stock_prices)
            self.portfolio = np.zeros(self.n_timeseries)
            self.actions_chosen[self.current_step] = action
        elif self.portfolio[action] == 0 or self.balance != 0:  # otherwise, hold
            self.balance += np.inner(self.portfolio, open_stock_prices)
            self.portfolio = np.zeros(self.n_timeseries)
            stock_bought = self.balance / (open_stock_prices[action] * (1 + self.fee))
            self.portfolio[action] = stock_bought
            self.balance = 0

            self.actions_chosen[self.current_step] = action

        episode_ending_portfolio_value = self.balance + np.inner(self.portfolio, close_stock_prices)
        reward = episode_ending_portfolio_value - episode_starting_portfolio_value

        if action != self.action_space.n - 1:
            self.reward_per_timeseries[action] += reward
        self.portfolio_value_list.append(episode_ending_portfolio_value)
        self.reward_list.append(reward)

        self.current_step += 1
        if self.current_step == self.max_episode_steps:
            done = True
            obs = np.zeros(self.observation_space.n)

            self.episode_ending_portfolio_value_list.append(episode_ending_portfolio_value)
            self.episode_ending_value_versus_bah_list.append(episode_ending_portfolio_value / self.bah_ending_value)
            self._save_episode_results()
            self.episode += 1
        else:
            done = False
            obs = self._get_obs(open_columns_numbers)

        return obs, reward, done, {}

    def reset(self):
        self.actions_chosen = dict()
        self.reward_per_timeseries = np.zeros(self.n_timeseries)
        self.portfolio_value_list = []
        self.reward_list = []

        obs = super().reset()

        open_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                'open' in i]
        close_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                 'close' in i]
        starting_stock_prices = self.data.iloc[
            self.episode_starting_step, open_columns_numbers].values
        ending_stock_prices = self.data.iloc[
            self.episode_starting_step + self.max_episode_steps, close_columns_numbers].values
        self.bah_ending_value = (
                self.starting_balance / self.n_timeseries * (
                    ending_stock_prices / (starting_stock_prices * (1 + self.fee)))).sum()

        # Reset epoch episode counter
        if self.episode == self.num_test_episodes:
            self.episode = 0
            self.epoch += 1
            self.episode_ending_portfolio_value_list = []
            self.episode_ending_value_versus_bah_list = []
            makedirs(f'{self.log_dir}epoch-{self.epoch}', exist_ok=True)
        return obs

    def _save_episode_results(self):
        """ Plot timeseries and agent`s actions. Then save the figure """

        # Agents actions plot
        sns.set()
        self._calc_buy_and_sell_timesteps()
        open_columns_numbers = [list(self.data.columns.values).index(i) for i in self.data.columns.values if
                                'open' in i]
        plt.figure(0)
        plt.clf()

        fig, axs = plt.subplots(self.n_timeseries, figsize=(14, 10))
        plt.suptitle('Episode Agent`s actions Visualization')
        plt.xlabel('Time')
        for i in range(self.n_timeseries):
            labels = [None, None, None]
            if i == 0:
                labels = ['timeseries', 'buy', 'sell']
            timeseries = self.data.iloc[self.episode_starting_step:self.episode_starting_step + self.max_episode_steps,
                         open_columns_numbers[i]].to_frame()
            timeseries.columns = [f'open {self.crypto_names[i]}']
            sns.lineplot(x=timeseries.index, y=timeseries.iloc[:, 0], label=labels[0], zorder=1, ax=axs[i])
            sns.scatterplot(x=timeseries.index[self.buy_and_sell_timesteps[i]['buy']],
                            y=timeseries.iloc[self.buy_and_sell_timesteps[i]['buy'], 0], label=labels[1], s=40,
                            marker='^', zorder=2, ax=axs[i])
            sns.scatterplot(x=timeseries.index[self.buy_and_sell_timesteps[i]['sell']],
                            y=timeseries.iloc[self.buy_and_sell_timesteps[i]['sell'], 0], label=labels[2], s=35,
                            marker='v', zorder=3, ax=axs[i])
        plt.savefig(f'{self.log_dir}epoch-{self.epoch}/test_episode-{self.episode}-actions.png', dpi=300)

        # Portfolio value plot
        plt.figure(0)
        plt.clf()
        sns.lineplot(x=[i for i in range(len(self.portfolio_value_list))], y=self.portfolio_value_list)
        plt.title('Portfolio value')
        plt.xlabel('Time')
        plt.ylabel('Portfolio value')
        plt.savefig(f'{self.log_dir}epoch-{self.epoch}/test_episode-{self.episode}-portfolio.png', dpi=200)

        # Reward plot
        plt.figure(0)
        plt.clf()
        sns.lineplot(x=[i for i in range(len(self.reward_list))], y=self.reward_list)
        plt.title('Episode reward per timestep')
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.savefig(f'{self.log_dir}epoch-{self.epoch}/test_episode-{self.episode}-reward.png', dpi=200)

        # Reward per timeseries plot
        plt.figure(0)
        plt.clf()
        sns.barplot(x=[f'series {i}' for i in range(self.n_timeseries)], y=self.reward_per_timeseries)
        plt.title('Episode reward per timeseries')
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.savefig(f'{self.log_dir}epoch-{self.epoch}/test_episode-{self.episode}-reward-per-series.png', dpi=200)

        if self.episode == self.num_test_episodes - 1:  # Save epoch ending portfolio value distribution plot
            self.epoch_distributions.append(self.episode_ending_portfolio_value_list)
            self.epoch_versus_bah_distributions.append(self.episode_ending_value_versus_bah_list)

            plt.figure(0)
            plt.clf()
            ax = sns.boxplot(y=self.episode_ending_portfolio_value_list)
            plt.title('Test episode ending portfolio value distribution')
            plt.xlabel('Portfolio end value')
            plt.savefig(f'{self.log_dir}epoch-{self.epoch}/boxplot_distribution', dpi=100)

            plt.figure(0)
            plt.clf()
            ax = sns.boxplot(y=self.episode_ending_value_versus_bah_list)
            plt.title('Test episode ending portfolio value / b&h value distribution')
            plt.xlabel('Portfolio end value ratio')
            plt.savefig(f'{self.log_dir}epoch-{self.epoch}/boxplot_versus_bah_distribution', dpi=100)

            # Save epoch test episode portfolio ending values
            self.episode_ending_portfolio_value_list.sort()
            self.episode_ending_value_versus_bah_list.sort()
            save_list_to_txt(self.episode_ending_portfolio_value_list,
                             f'{self.log_dir}epoch-{self.epoch}/epoch_distribution.txt')
            save_list_to_txt(self.episode_ending_value_versus_bah_list,
                             f'{self.log_dir}epoch-{self.epoch}/epoch_versus_bah_distribution.txt')

            # Plot all epochs` distributions in one plot
            df = pd.DataFrame(data=self.epoch_distributions).T
            df.columns = [f'Epoch {i}' for i in range(len(self.epoch_distributions))]

            plt.figure(0)
            plt.clf()
            ax = sns.boxplot(data=df)
            plt.title('Test episode ending portfolio value distribution')
            plt.ylabel('Portfolio value')
            plt.savefig(f'{self.log_dir}epoch-{self.epoch}/epoch_boxplot_distribution', dpi=100)

            df = pd.DataFrame(data=self.epoch_versus_bah_distributions).T
            df.columns = [f'Epoch {i}' for i in range(len(self.epoch_versus_bah_distributions))]

            plt.figure(0)
            plt.clf()
            ax = sns.boxplot(data=df)
            plt.title('Test episode ending portfolio value / b&h value distribution')
            plt.ylabel('Portfolio value ratio')
            plt.savefig(f'{self.log_dir}epoch-{self.epoch}/epoch_bah_boxplot_distribution', dpi=100)

    def _calc_buy_and_sell_timesteps(self):
        """ Process self.actions_chosen. Calculate buy and sell time steps for each stock """
        self.buy_and_sell_timesteps = [dict() for _ in range(self.n_timeseries)]

        for i in range(self.n_timeseries):
            self.buy_and_sell_timesteps[i]['buy'] = []
            self.buy_and_sell_timesteps[i]['sell'] = []

        for i, (timestep, action) in enumerate(self.actions_chosen.items()):
            if i != 0:
                if previous_action != self.action_space.n - 1:  # noqa
                    self.buy_and_sell_timesteps[previous_action]['sell'].append(timestep)  # noqa
            if action != self.action_space.n - 1:
                self.buy_and_sell_timesteps[action]['buy'].append(timestep)
            previous_action = action


max_episode_steps = 1000
gym.envs.register(id='TradeEnv-v0', entry_point=TradeEnv,
                  max_episode_steps=max_episode_steps)
gym.envs.register(id='TestTradeEnv-v0', entry_point=TestTradeEnv,
                  max_episode_steps=max_episode_steps)
