import datetime as dt
import numpy as np
from pathlib import Path
import time

import gym
import ma_gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agents import ComaAgent
from config import Args, Episode

ENV_NAME = 'Combat-v0'
COMBAT_AGENTS = 10

MODEL_NAME = 'COMA'
LOGGING_DEST = Path.cwd().joinpath(Path(f"logs/{MODEL_NAME}-{ENV_NAME}-{dt.datetime.now().strftime('%y%m%d-%H%M%S')}"))


if __name__ == '__main__':
    tb_writer = SummaryWriter(str(LOGGING_DEST))

    env = gym.make(ENV_NAME, grid_shape=(20, 20), n_agents=COMBAT_AGENTS, n_opponents=COMBAT_AGENTS)

    n_obs = env.observation_space[0].shape[0]
    n_actions = env.action_space[0].n
    n_agents = env.n_agents

    ARGS = Args(
        n_agents=n_agents,
        n_actions=n_actions,
        state_shape=n_obs * n_agents,  # could also incorporate action history
        obs_shape=n_obs,
        log_every=20
    )
    agents = ComaAgent(ARGS)

    print('\n')
    print(f'Starting env {ENV_NAME} | Action space: {env.action_space} | Obs space: {env.observation_space}')
    print(f'Using device {"CUDA" if ARGS.cuda else "CPU"}')
    print(f'Logging results to: {LOGGING_DEST.expanduser()}')
    print('\n')

    episode_rewards = []
    epsilon = 0 if ARGS.evaluate else ARGS.epsilon

    for episode_idx in range(1, ARGS.n_episodes + 1):
        agents.policy.init_hidden()

        if epsilon > ARGS.min_epsilon:
            epsilon -= ARGS.anneal_epsilon

        current_obs_n = env.reset()

        done_n = [False for a in range(env.n_agents)]
        done = all(done_n)

        ep_reward = 0
        ep_step = 0

        last_actions_c = np.zeros((env.n_agents, ARGS.n_actions))

        obs_h, state_h, actions_h, actions_onehot_h, rewards_h, obs_next_h, state_next_h, \
            terminated_h = [], [], [], [], [], [], [], []

        while not done:
            actions_c, actions_onehot_c = [], []

            for agent_id in range(env.n_agents):
                action_c = agents.act(obs=current_obs_n[agent_id], last_action=last_actions_c[agent_id],
                                      agent_num=agent_id, epsilon=epsilon, evaluate=False)

                action_onehot_c = np.zeros(env.action_space[0].n)
                action_onehot_c[action_c] = 1
                last_actions_c[agent_id] = action_onehot_c

                actions_c.append(action_c)
                actions_onehot_c.append(action_onehot_c)

            next_obs_n, reward_n, done_n, _ = env.step(actions_c)

            if not episode_idx % ARGS.log_every:
                env.render()

            done = all(done_n)

            state = []
            for obs in current_obs_n:
                state.extend(obs)

            next_state = []
            for next_obs in next_obs_n:
                next_state.extend(next_obs)

            obs_h.append(current_obs_n)
            obs_next_h.append(next_obs_n)

            state_h.append(state)
            state_next_h.append(next_state)

            actions_h.append(np.reshape(actions_c, [n_agents, 1]))
            actions_onehot_h.append(actions_onehot_c)

            rewards_h.append([sum(reward_n)])
            terminated_h.append([done])

            current_obs_n = next_obs_n

            ep_reward += sum(reward_n)
            ep_step += 1

        episode = Episode(
            obs=torch.tensor(obs_h, dtype=torch.float, device=ARGS.device),
            state=torch.tensor(state_h, dtype=torch.float, device=ARGS.device),
            actions=torch.tensor(actions_h, dtype=torch.long, device=ARGS.device),
            actions_onehot=torch.tensor(actions_onehot_h, dtype=torch.float, device=ARGS.device),
            rewards=torch.tensor(rewards_h, dtype=torch.float, device=ARGS.device),
            obs_next=torch.tensor(obs_next_h, dtype=torch.float, device=ARGS.device),
            state_next=torch.tensor(state_next_h, dtype=torch.float, device=ARGS.device),
            terminated=torch.tensor(terminated_h, dtype=torch.bool, device=ARGS.device)
        )

        loss = agents.train(episode, episode_idx, epsilon=epsilon)

        episode_rewards.append(ep_reward)

        tb_writer.add_scalar('Reward', ep_reward, episode_idx)
        tb_writer.add_scalar('COMA loss', loss, episode_idx)
        tb_writer.add_scalar('Epsilon', epsilon, episode_idx)

        if not episode_idx % ARGS.log_every:
            time.sleep(0.1)  # pause to show env final state
            print(f'On episode {episode_idx:,d} // '
                  f'Epsilon: {epsilon:.2f} // '
                  f'Mean reward: {np.mean(episode_rewards[-ARGS.log_every:]):.1f} // '
                  f'Min reward {np.min(episode_rewards[-ARGS.log_every:]):.1f} // '
                  f'Max reward {np.max(episode_rewards[-ARGS.log_every:]):.1f}')

            # # Replay state - action - reward episode sequence
            # for t in range(ep_step - 1):
            #     print(obs_h[t], actions_h[t], rewards_h[t], terminated_h[t])
