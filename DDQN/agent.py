from collections import deque, namedtuple

import numpy as np
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple(
            'transition', 
            ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, buffer_size: int) -> None:
        self.capacity = buffer_size

        # Sampling would be faster if a list, but deque takes care of pushing stale samples off the stack for us.
        self.memory = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.memory)

    def add(self, state: torch.tensor, action: torch.tensor, reward: torch.tensor, next_state: torch.tensor,
            done: torch.tensor) -> None:
        self.memory.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done))

    def sample(self, batch_size: int) -> List[Transition]:
        # Note: doesn't remove samples from memory
        return random.sample(self.memory, k=batch_size)


class DQN(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layers_size)
        self.fc2 = nn.Linear(hidden_layers_size, hidden_layers_size // 2)
        self.fc3 = nn.Linear(hidden_layers_size // 2, hidden_layers_size // 4)
        self.fc4 = nn.Linear(hidden_layers_size // 4, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(
        self, 
        observation_space_dim: int, 
        n_actions: int,
        hidden_layers_size: int = 128,
        buffer_size: int = 5_000,
        batch_size: int = 64,
        start_learning_steps: int = 1_000,
        update_target_steps: int = 300,
        agent_type: str = 'ddqn',
        loss_func: str = 'huber',
        learning_rate: float = 0.005,
        grads_clip_lim: Optional[float] = None,
        gamma: float = 0.9,
        eps_start: float = 0.99,
        eps_end: float = 0.02,
        eps_decay_steps: int = 2_500
    ):
        assert loss_func in ('mse', 'huber')
        assert agent_type in ('dqn', 'ddqn')

        self.n_actions = n_actions
        self.observation_space_dim = observation_space_dim
        self.hidden_layers_size = hidden_layers_size

        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_learning_steps = start_learning_steps

        self.agent_type = agent_type
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grads_clip_lim = grads_clip_lim

        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.update_target_steps = update_target_steps
        self.timestep = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy_net = DQN(in_features=observation_space_dim, out_features=n_actions,
                              hidden_layers_size=self.hidden_layers_size).to(self.device)
        self.target_net = DQN(in_features=observation_space_dim, out_features=n_actions,
                              hidden_layers_size=self.hidden_layers_size).to(self.device)

        # Set the target network's weights equal to our randomly initialised policy_net
        self._update_target_net()

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(),
            lr=self.learning_rate)

        print('Initialised agent. Using device', self.device)

    def _store_transitions(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        state_t = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float, device=self.device)
        done_t = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.memory.add(state=state_t, action=action_t, reward=reward_t, next_state=next_state_t, done=done_t)

    def _update_epsilon(self):
        # Note: we update epsilon relative to experience beyond `start_learning_steps`
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                       np.exp(-(self.timestep - self.start_learning_steps) / self.eps_decay_steps)

    def _update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
             done: bool) -> Optional[float]:
        self.timestep += 1

        self._store_transitions(state, action, reward, next_state, done)

        # Avoid sampling if we don't yet have enough experience
        if self.timestep < self.start_learning_steps or len(self.memory) < self.batch_size:
            return

        self._update_epsilon()

        transitions = self.memory.sample(self.batch_size)
        return self.train(transitions)

    def act(self, state: np.ndarray, evaluate: bool = False) -> int:
        if random.random() < self.epsilon and not evaluate:
            return random.randint(0, self.n_actions - 1)
        else:
            # Convert to shape 1, n_obs
            state_t = torch.from_numpy(state).type(torch.float).unsqueeze(0).to(self.device)
            
            # Again easy to forget -- no grad this!
            with torch.no_grad():
                q_vals_t = self.policy_net.forward(state_t)

                # Max returns the max value, we select the index of this to correspond to the action
                greedy_action = q_vals_t.max(1).indices.cpu().item()
                return greedy_action

    def train(self, transitions: List[Transition]) -> float:
        batch = Transition(*zip(*transitions))

        state_b_t = torch.stack(batch.state)
        next_state_b_t = torch.stack(batch.next_state)

        action_b_t = torch.cat(batch.action)
        reward_b_t = torch.cat(batch.reward)

        # Invert done_b because we want to ignore incremental reward estimate when the episode had ended
        # Note: danger awaits if this is not a bool! Inversion of True/1 = False/0. Inversion of 1 = -1!
        not_done_b_t = ~torch.cat(batch.done)
        
        # Current q value estimate from online net based on the actions that were taken
        current_qs = self.policy_net.forward(state_b_t).gather(1, action_b_t.unsqueeze(1))

        # In all cases below we detach to ignore the gradient
        # https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
        if self.agent_type == 'dqn':
            target_qs = self.target_net.forward(next_state_b_t).max(1).values.detach()
        # https://arxiv.org/abs/1509.06461
        # Decouple action selection and value estimation by using the online net for action selection
        # Then gather target net's value estimate relative to those selected actions
        elif self.agent_type == 'ddqn':
            policy_actions = self.policy_net.forward(next_state_b_t).max(1).indices.detach()
            target_qs = self.target_net.forward(next_state_b_t).detach().gather(
                1, policy_actions.unsqueeze(1)).squeeze(1)
        else:
            raise AttributeError('Unknown agent type. Expected dqn or ddqn')

        # Future q = reward + discounted future q value estimate based on actions to be taken
        future_qs = reward_b_t + self.gamma * target_qs * not_done_b_t
        
        # Transpose future_qs so dealing with same shape (at this point just q_val x batch_size)
        if self.loss_func == 'huber':
            loss = F.smooth_l1_loss(current_qs, future_qs.unsqueeze(1))
        elif self.loss_func == 'mse':
            loss = F.mse_loss(current_qs, future_qs.unsqueeze(1))
        else:
            raise AttributeError('Unknown loss function. Expected huber or mse')
        
        self.optimizer.zero_grad()
        loss.backward()

        if self.grads_clip_lim:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1 * self.grads_clip_lim, self.grads_clip_lim)

        self.optimizer.step()

        if not self.timestep % self.update_target_steps:
            self._update_target_net()

        return loss.cpu().item()
