"""
Automatic Cryptocurrency trading using Deep RL
Nick Kaparinos
2022
"""

import time
from trading_utilities import *
from utilities import *
from trading_network_utilities import MLP
import torch.optim
import tianshou as ts
from tianshou.policy import DQNPolicy
from tianshou.utils import WandbLogger
from tianshou.data import PrioritizedVectorReplayBuffer as PVRB, VectorReplayBuffer as VRB
from torch.utils.tensorboard import SummaryWriter
import wandb


def main():
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Trading Environment
    env_id = 'TradeEnv-v0'
    test_env_id = 'TestTradeEnv-v0'

    # Config
    num_envs = 1
    continue_from_previous_run, latest_run_epoch = False, -1
    previous_runs = ['']
    trainer_hyperparameters = {'max_epoch': 6, 'step_per_epoch': 150_000, 'step_per_collect': 100,
                               'episode_per_test': 25, 'batch_size': 128, 'update_per_step': 0.1}
    policy_hyperparameters = {'discount_factor': 0.99, 'estimation_step': 1, 'target_update_freq': 20,
                              'is_double': True}
    epsilon_schedule_hyperparameters = {'max_epsilon': 0.6, 'min_epsilon': 0.0, 'num_episodes_decay': int(
        trainer_hyperparameters['step_per_epoch'] * trainer_hyperparameters['max_epoch'] * 0.9)}
    replay_buffer_hyperparameters = {'total_size': 500_000, 'buffer_num': num_envs, 'alpha': 0.7, 'beta': 0.5}
    net_hyperparameters = {'encoder_type': 'LSTM', 'n_neurons': 128, 'encoder_n_linear_layers': 2,
                           'q_n_linear_layers': 2, 'v_n_linear_layers': 2, 'n_posmlp_layers': 2,
                           'n_head_layers': 1, 'n_attention_blocks': 1, 'n_cnn_layers': 2, 'dueling': True}
    env_kwargs = {
        'crypto_files': ['Data/ADAUSDT_minutes.hdf5', 'Data/BTCUSDT_minutes.hdf5', 'Data/ETHUSDT_minutes.hdf5',
                         'Data/LTCUSDT_minutes.hdf5'], 'timeseries_step': '1H', 'max_episode_steps': max_episode_steps,
        'n_previous_timesteps': 23}
    misq_dict = {'learning_rate': 1e-4, 'seed': seed, 'use_prioritised_replay_buffer': True, 'optimizer': 'Adam',
                 'continue_from_previous_run': continue_from_previous_run, 'algorithm': 'DQN',
                 'latest_run_epoch': latest_run_epoch, 'previous_runs': previous_runs}
    config = dict(policy_hyperparameters, **trainer_hyperparameters, **net_hyperparameters,
                  **epsilon_schedule_hyperparameters, **replay_buffer_hyperparameters, **env_kwargs, **misq_dict)

    # Logging
    model_name = f'{config["algorithm"]}_{time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime())}'
    log_dir = f'logs/trading/{env_id[:-3]}/{model_name}/'
    makedirs(log_dir, exist_ok=True)
    project = 'Gym-' + env_id[:-3]
    logger = WandbLogger(train_interval=1, save_interval=1, project=project, entity='nickkaparinos', name=model_name,
                         run_id=model_name, config=config)  # type: ignore
    logger.load(SummaryWriter(log_dir))

    # Environment
    train_envs = ts.env.SubprocVectorEnv([lambda: gym.make(env_id, **env_kwargs) for _ in range(num_envs)])
    test_envs = ts.env.SubprocVectorEnv(
        [lambda: gym.make(test_env_id, log_dir=log_dir, num_test_episodes=config['episode_per_test'], **env_kwargs) for
         _ in range(num_envs)])
    train_envs.seed(seed)
    test_envs.seed(seed)
    env = gym.make(env_id, **env_kwargs)

    # Neural network
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = MLP(state_shape, action_shape, env.n_features, env.n_previous_timesteps, env.n_timeseries,
              **net_hyperparameters, device=device).to(device)

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f'Optimizer: {config["optimizer"]} not supported')
    optim = optimizer(net.parameters(), lr=config['learning_rate'])

    # Policy
    policy = DQNPolicy(net, optim, **policy_hyperparameters)

    # Collectors
    if config['use_prioritised_replay_buffer']:
        train_collector = ts.data.Collector(policy, train_envs, PVRB(**replay_buffer_hyperparameters),
                                            exploration_noise=True)
    else:
        train_collector = ts.data.Collector(policy, train_envs,
                                            VRB(total_size=replay_buffer_hyperparameters['total_size'] * num_envs,
                                                buffer_num=num_envs), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    # Load previous run
    if continue_from_previous_run:
        load_previous_run(previous_runs, latest_run_epoch, policy, train_collector, env_id)

    # Training
    policy.set_eps(epsilon_schedule_hyperparameters['max_epsilon'])
    save_dict_to_txt(config, path=log_dir, txt_name='config')
    train_fn = build_epsilon_schedule(policy=policy, **epsilon_schedule_hyperparameters)
    test_fn = build_test_fn(policy, optim, log_dir, model_name, train_collector, True, model=config['algorithm'])
    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=train_fn, test_fn=test_fn, stop_fn=None, logger=logger)
    print(f'Finished training! Duration {result["duration"]}')

    # Learning curve
    windows = [25, 50]
    make_learning_curve(project, model_name, previous_runs, log_dir, windows, continue_from_previous_run)

    # Execution Time
    wandb.finish()
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
