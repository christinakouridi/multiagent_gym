import datetime as dt
from pathlib import Path
import time
from typing import List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import ma_gym
from ma_gym.wrappers import Monitor

from agent import DQNAgent
from agent_args import AGENT_ARGS

import warnings
warnings.filterwarnings("ignore")


def wrap_env(env):
    return Monitor(env, './video', force=True)


def _concat_obs(obs_n: List) -> np.ndarray:
    obs_out = []
    for a, agent_obs in enumerate(obs_n):
        t_obs = []
        for obs in obs_n:
            t_obs.extend(obs)
        t_obs.append(a/len(obs_n))  # normalise agent indices
        obs_out.append(t_obs)

    return np.array(obs_out, dtype=np.float)


def train_ma(
        training_episodes: int,
        env_name: str,
        agent_kwargs: dict,
        log_every: int = 500,
        render: bool = True
):

    env = gym.make(env_name)

    # Obs space = both observations plus a unique agent key
    agent_kwargs['observation_space_dim'] = env.observation_space[0].shape[0] * env.n_agents + 1
    agent_kwargs['n_actions'] = env.action_space[0].n

    agents = [DQNAgent(**agent_kwargs) for _ in range(env.n_agents)]

    print('Agents initialised. Training...')

    episode_rewards = []
    start_time = time.time()

    for episode in range(1, training_episodes + 1):

        obs_n = _concat_obs(env.reset())
        done_n = [False for _ in agents]
        agent_losses = 0
        ep_reward = 0
        ep_step = 0

        while not all(done_n):
            action_n = [-1 for _ in agents]
            for a, agent in enumerate(agents):
                action_n[a] = agent.act(np.array(obs_n[a]))

            next_obs_n, reward_n, done_n, info = env.step(action_n)
            next_obs_n = _concat_obs(next_obs_n)

            if render and not episode % log_every:
                env.render()
                time.sleep(0.1)

            ep_reward += sum(reward_n)

            for a, agent in enumerate(agents):
                loss = agent.step(
                    state=np.array(obs_n[a]),
                    action=action_n[a],
                    reward=reward_n[a] if not done_n[a] else PER_AGENT_REWARD,
                    next_state=np.array(next_obs_n[a]),
                    done=done_n[a]
                )

                agent_losses += loss if loss else 0

            obs_n = next_obs_n
            ep_step += 1

        episode_rewards.append(ep_reward)

        TB_WRITER.add_scalar('Loss', agent_losses, episode)
        TB_WRITER.add_scalar('Episode reward', ep_reward, episode)
        TB_WRITER.add_scalar('Epsilon', agents[0].epsilon, episode)

        if not episode % log_every:
            current_time = time.time()

            if render:
                time.sleep(0.2)  # pause to see final state

            print(f'Ep: {episode} / '
                  f'(Last {log_every:,.0f}) Mean: {np.mean(episode_rewards[-log_every:]):.1f} / '
                  f'Min: {np.min(episode_rewards[-log_every:]):.1f} / '
                  f'Max: {np.max(episode_rewards[-log_every:]):.1f} / '
                  f'EPS: {episode / (current_time - start_time):.1f} / '
                  f'Agent epsilon: {agents[0].epsilon:.2f}'
                  )

    print('Done training!\n')
    env.close()

    return agents, episode_rewards


def test_ma(agents: List[DQNAgent], test_eps):
    env = wrap_env(gym.make(ENV_NAME))
    ep_rewards = []

    for test_ep in range(test_eps):
        obs_n = _concat_obs(env.reset())
        done_n = [False for _ in agents]

        ep_reward = 0
        ep_step = 0

        while not all(done_n):

            action_n = [agent.act(np.array(obs_n[a]), evaluate=True)
                        for a, agent in enumerate(agents)]
            next_obs_n, reward_n, done_n, _ = env.step(action_n)
            next_obs_n = _concat_obs(next_obs_n)

            env.render()

            obs_n = next_obs_n

            ep_reward += sum(reward_n)
            ep_step += 1

        ep_rewards.append(ep_reward)
        time.sleep(0.5)

    print('\n')
    print('=== Test performance ===')
    print(f'Mean: {np.mean(ep_rewards):.1f} / '
          f'Min: {np.min(ep_rewards):.1f} / '
          f'Max: {np.max(ep_rewards):.1f}')

    env.close()
    return ep_rewards


if __name__ == '__main__':

    ENV_NAME = 'Switch4-v0'
    MODEL_NAME = 'AC-LINx3-128'

    LOG_EVERY = 200
    STEPS_PER_EPISODE = 50
    PER_AGENT_REWARD = 5.0

    LOGGING_DEST = Path.cwd().joinpath(
        Path(f"logs/{MODEL_NAME}-{ENV_NAME}-{dt.datetime.now().strftime('%y%m%d-%H%M%S')}"))

    TB_WRITER = SummaryWriter(str(LOGGING_DEST))

    TRAINING_EPISODES = 1_000

    print('Beginning training')
    print('Logging to:', LOGGING_DEST)

    trained_agents, training_rewards = train_ma(TRAINING_EPISODES, ENV_NAME, AGENT_ARGS,
                                                log_every=LOG_EVERY, render=True)
    test_ma(trained_agents, 1)  # this will also save a video
