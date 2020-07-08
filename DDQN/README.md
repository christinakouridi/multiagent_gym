
## Overview

Adaption of DQN ([Minh et al 2015](https://www.nature.com/articles/nature14236)) and DDQN ([van Hassalt et al 2015](https://arxiv.org/abs/1509.06461)) for the multi-agent Gym environments [Switch 2 and 4](https://github.com/koulanurag/ma-gym), as part of my Multi-agent AI module assignment at UCL.

## Files

- `agent.py`: network and agent definitions
- `agent_args.py`: simple dict for agent hyperparameters
- `train_switch_simple.py`: multi-agent Switch, game from ma-gym. Train all agents concurrently on their own observations (highly unlikely to be successful)
- `train_switch_joint_obs.py`: train agents concurrently, but on a joint observation space that concatenates all agent observations + a unique agent id
- `train_switch_curriculum.py`: encourage Switch training by starting training for only one agent at a time, leaving untrained agents to a no-op (best performance)
- `DDQN_magym.ipynb`: colab version of train_switch_curriculum.py best viewed directly in [Colab](https://colab.research.google.com/drive/1RV-anR5C1PqKQWXza2XubYyb4UPpFwQ0)

## Usage

Run all cells in [the Colab version](https://colab.research.google.com/drive/1RV-anR5C1PqKQWXza2XubYyb4UPpFwQ0)

or

Run `$ python train_switch_curriculum.py`

Edit hyperparams in `agent_args.py`.

## Requirements

- Python 3.6+
- PyTorch 1.4+
- Gym

## Results

<center> <b> Switch 2 </b> </center>

![Alt Text](https://github.com/christinakouridi/multiagent_gym/blob/master/DDQN/results/switch2_test.gif)

![Alt Text](https://github.com/christinakouridi/multiagent_gym/blob/master/DDQN/results/switch2_learningcurve.png)

<center> <b> Switch 4 </b> </center>

![Alt Text](https://github.com/christinakouridi/multiagent_gym/blob/master/DDQN/results/switch4_test.gif)

![Alt Text](https://github.com/christinakouridi/multiagent_gym/blob/master/DDQN/results/switch4_learningcurves.png)

## Comments

We are able to demonstrate near-optimal performance on `Switch2-v0` with two key modifications to the training conditions of a DDQN agent (implemented in PyTorch):

1. **Environment-wide rewards:** at each timestep in a training episode both agents are passed the current reward profile of the entire environment, rather than the agent-specific reward (ie `sum(reward_n)` vs `reward_n[{0, 1}]`). Given the cooperative nature of `Switch`, providing a learning trajectory that optimises for joint behaviour is critical to avoid the emergence of sub-optimal policies that were observed before making this change – such as each agent rushing to pass through the corridor at the beginning of each episode, blocking each other.

2. **Staggered training:** another key element for convergence is the staggered training of agents. Broadly, for a limited period of initial training we train only one agent, while holding the other to a no-op at every time step. This allows a single agent to quickly learn the optimal strategy (to head straight through the corridor to their target square – as observed via rendering the environment through training for debugging purposes). When the second agent begins to train (conditioned on both agents' rewards at every step) it is able to successfully adapt to the first agent's learnt behaviour, waiting for it to traverse the corridor before following.

