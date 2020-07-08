# COMA implementation in PyTorch

## Overview

An implementation of the COMA algorithm from [Foerster et al 2017](https://arxiv.org/pdf/1705.08926.pdf). Includes easily introspectable training scripts for multi-agent Gym environments, such as Switch 4 and Combat found in [ma-gym](https://github.com/koulanurag/ma-gym).

## Files
- `agents.py`: COMA network and agent definitions
- `config.py`: agent hyperparameters
- `networks.py`: definition of Actor and Critic networks
- `train_combat.py`:  train 10 agents concurrently in the Combat multi-agent gym environment
- `train_switch.py`: train 4 agents concurrently in the Switch multi-agent gym environment
- `COMA_magym.ipynb`: colab version of train_combat.py, best viewed in [Colab](https://colab.research.google.com/drive/1jFdwDIuhgm_frIHNSr2MN-n2B2ERpyxN#scrollTo=7quwWZTJlwfV)

## Usage

Run all cells in [the Colab version](https://colab.research.google.com/drive/1jFdwDIuhgm_frIHNSr2MN-n2B2ERpyxN#scrollTo=7quwWZTJlwfV)

or 

Run `$ python train_combat.py`

Edit hyperparams in `config.py`

## Requirements

- Python 3.6+
- PyTorch 1.4+
- Gym
