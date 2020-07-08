import torch
import torch.nn as nn
import torch.nn.functional as F


class ComaCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(ComaCritic, self).__init__()

        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, args.n_actions)

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        # Note: using leaky RELU to mitigate vanishing gradients
        x = F.leaky_relu(self.fc1(inputs))
        x = F.leaky_relu(self.fc2(x))
        q = self.fc3(x)
        return q


class PolicyRnn(nn.Module):
    def __init__(self, input_shape: int, args):
        super(PolicyRnn, self).__init__()
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

    def init_hidden(self) -> torch.tensor:
        # Note zero initialisation here
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, obs: torch.tensor, hidden_state: torch.tensor) -> (torch.tensor, torch.tensor):
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        x = F.leaky_relu(self.fc1(obs))
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
