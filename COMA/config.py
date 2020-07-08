from collections import namedtuple
import torch

Episode = namedtuple('Episode', ['obs', 'state', 'actions', 'actions_onehot', 'rewards', 'obs_next', 'state_next',
                                 'terminated'])


class Args:
    def __init__(self, n_actions=None, n_agents=None, state_shape=None,
                 obs_shape=None, seed=123, rnn_hidden_dim=64, critic_dim=128,
                 lr_actor=.0005, lr_critic=.0005, epsilon=0.6, anneal_epsilon=.0005, min_epsilon=.02,
                 td_lambda=0.9, grad_norm_clip=5.0, gamma=0.99, target_update_cycle=10,
                 log_every=50, n_episodes=5_000, evaluate=False):

        self.n_actions = n_actions
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.obs_shape = obs_shape

        self.gamma = gamma
        self.evaluate = evaluate

        self.grad_norm_clip = grad_norm_clip

        self.cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'

        self.seed = seed

        self.rnn_hidden_dim = rnn_hidden_dim
        self.critic_dim = critic_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.epsilon = epsilon
        self.anneal_epsilon = anneal_epsilon
        self.min_epsilon = min_epsilon

        self.td_lambda = td_lambda

        self.n_episodes = n_episodes
        self.log_every = log_every
        self.target_update_cycle = target_update_cycle
