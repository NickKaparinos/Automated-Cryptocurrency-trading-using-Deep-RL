"""
Automatic Cryptocurrency trading using Deep RL
Nick Kaparinos
2022
"""

import random
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pickle import dump, load
from typing import Callable, Optional, Tuple, List, Sequence
import bz2


def make_learning_curve(project: str, model_name: str, previous_runs: Sequence[str], log_dir: str, windows: List[int],
                        continue_from_previous_run: Optional[bool] = False) -> None:
    """ Make learning curve and log to wandb """
    api = wandb.Api()
    run = api.run(f'nickkaparinos/{project}/{model_name}')
    history = run.scan_history()
    episode_rewards = get_episode_rewards(history)

    # Download previous runs data
    for previous_run in reversed(previous_runs):
        if continue_from_previous_run:
            run = api.run('nickkaparinos/' + project + '/' + previous_run)
            history = run.scan_history()
            episode_rewards_temp = get_episode_rewards(history)
            episode_rewards = pd.concat([episode_rewards_temp, episode_rewards], axis=0).reset_index(drop=True)

    # Learning Curve
    for window in windows:
        learning_curve(episode_rewards=episode_rewards, log_dir=log_dir, window=window)


def learning_curve(episode_rewards: pd.DataFrame, log_dir: str, window: int) -> None:
    # Calculate rolling window metrics
    rolling_average = episode_rewards.rolling(window=window, min_periods=1).mean().dropna()
    rolling_max = episode_rewards.rolling(window=window, min_periods=1).max().dropna()
    rolling_min = episode_rewards.rolling(window=window, min_periods=1).min().dropna()

    # Change column name
    rolling_average.columns = ['Average Reward']
    rolling_max.columns = ['Max Reward']
    rolling_min.columns = ['Min Reward']
    rolling_data = pd.concat([rolling_average, rolling_max, rolling_min], axis=1)

    # Plot
    sns.set()
    plt.figure(0, dpi=200)
    plt.clf()
    ax = sns.lineplot(data=rolling_data)
    ax.fill_between(rolling_average.index, rolling_min.iloc[:, 0], rolling_max.iloc[:, 0], alpha=0.2)
    ax.set_title('Learning Curve', fontsize=16)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episodes')

    img_path = f'{log_dir}learning_curve{window}.png'
    plt.savefig(img_path, dpi=200)

    # Log figure
    image = plt.imread(img_path)
    wandb.log({f'Learning_Curve_{window}': [wandb.Image(image, caption="Learning_curve")]})


def get_episode_rewards(history: Sequence) -> pd.DataFrame:
    """ Get episode rewards from wandb scan history """
    episode_rewards_temp = []
    for i in history:
        if 'train/reward' in i.keys():
            episode_rewards_temp.append(i['train/reward'])
    return pd.DataFrame(data=episode_rewards_temp, columns=['Rewards'])


def build_test_fn(policy, optim, log_dir: str, model_name: str, train_collector,
                  save_train_buffer: Optional[bool] = True, model: Optional[str] = 'DQN') -> Callable:
    """ Build custom test function """

    def custom_test_fn(epoch, env_step):
        # Save agent
        print(f"Epoch = {epoch}")
        if model == 'DQN' or model == 'PPO':
            torch.save({'model': policy.state_dict(), 'optim': optim.state_dict()},
                       log_dir + model_name + f'_epoch{epoch}.pth')
        else:
            torch.save({'model': policy.state_dict(), 'actor_optim': optim['actor_optim'].state_dict(),
                        'critic1_optim': optim['critic1_optim'].state_dict(),
                        'critic2_optim': optim['critic1_optim'].state_dict()},
                       log_dir + model_name + f'_epoch{epoch}.pth')
        if save_train_buffer:
            with bz2.BZ2File(log_dir + f'epoch{epoch}_train_buffer' + '.pbz2', 'w') as f:
                dump(train_collector.buffer, f)

    return custom_test_fn


def build_epsilon_schedule(policy, max_epsilon: Optional[float] = 0.5, min_epsilon: Optional[float] = 0.0,
                           num_episodes_decay: Optional[int] = 10000) -> Callable:
    """ Build epsilon schedule function """

    def custom_epsilon_schedule(epoch, env_step):
        decay_step = (max_epsilon - min_epsilon) / num_episodes_decay
        current_epsilon = max_epsilon - env_step * decay_step
        if current_epsilon < min_epsilon:
            current_epsilon = min_epsilon
        policy.set_eps(current_epsilon)
        wandb.log({"train/env_step": env_step, 'epsilon': current_epsilon})

    return custom_epsilon_schedule


def load_previous_run(previous_runs: List[str], latest_run_epoch: int, policy, train_collector, env_id: str) -> None:
    """ Load model, optimizers and buffer from previous run"""
    assert latest_run_epoch > 0
    latest_run = previous_runs[-1]
    checkpoint = torch.load(f'logs/trading/{env_id[:-3]}/{latest_run}/{latest_run}_epoch{latest_run_epoch}.pth')
    policy.load_state_dict(checkpoint['model'])
    policy.optim.load_state_dict(checkpoint['optim'])
    with bz2.BZ2File(f'logs/trading/{env_id[:-3]}/{latest_run}/epoch{latest_run_epoch}_train_buffer.pbz2') as f:
        train_collector.buffer = load(f)


def save_dict_to_txt(dictionary: dict, path: str, txt_name: Optional[str] = 'hyperparameter_dict') -> None:
    """ Save dictionary as txt file """
    with open(f'{path}/{txt_name}.txt', 'w') as f:
        f.write(str(dictionary))


def save_list_to_txt(my_list: list, name: Optional[str] = 'epoch_distribution.txt') -> None:
    """ Save list as txt file """
    with open(name, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


def set_all_seeds(seed: Optional[int] = 0) -> None:
    """ Set all seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # noqa
