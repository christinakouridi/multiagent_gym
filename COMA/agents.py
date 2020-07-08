import numpy as np
import torch
from torch.distributions import Categorical

from config import Episode
from networks import ComaCritic, PolicyRnn


class ComaAgent:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.policy = Coma(args)

        self.args = args

    def act(self, obs: np.ndarray, last_action: np.ndarray, agent_num: int, epsilon: float,
            evaluate: bool = False) -> int:

        # Generate actor inputs >> concatenation of observation, previous action, agent id (one hot)
        agent_ids = np.zeros(self.n_agents)
        agent_ids[agent_num] = 1.

        inputs = np.hstack((obs, last_action, agent_ids))
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.args.device)

        # Select hidden state for relevant agent
        hidden_state = self.policy.eval_hidden[agent_num, :].to(self.args.device)

        # Get Q values and update hidden state for agent
        q_value, self.policy.eval_hidden[agent_num, :] = self.policy.actor.forward(inputs, hidden_state)
        return self._sample_action(q_value.detach(), epsilon, evaluate, self.n_actions)

    def train(self, episode: Episode, train_step: int, epsilon: float = None) -> float:
        return self.policy.learn(episode=episode, train_step=train_step, epsilon=epsilon)

    @staticmethod
    def _sample_action(q_values, epsilon, evaluate, n_actions):
        prob = torch.nn.functional.softmax(q_values, dim=-1)  # generate a probability distribution over q values

        if evaluate:  # if in evaluate mode simply return the max probability
            return torch.argmax(prob).cpu().item()
        else:  # otherwise re-weight probabilities by mixing in a uniform distribution over n_actions equal to epsilon
            prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / n_actions)
            return Categorical(prob).sample().cpu().item()


class Coma:
    def __init__(self, args):
        self.args = args
        actor_input_shape, critic_input_shape = self._get_input_shapes(self.args)

        self.actor = PolicyRnn(actor_input_shape, args).to(self.args.device)

        self.online_critic = ComaCritic(critic_input_shape, self.args).to(self.args.device)
        self.target_critic = ComaCritic(critic_input_shape, self.args).to(self.args.device)
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.online_critic.parameters(), lr=args.lr_critic)

        self.eval_hidden = None

    @staticmethod
    def _get_input_shapes(args) -> (int, int):
        # Actor input >> observation + previous action + agent id
        actor_input_shape = args.obs_shape
        actor_input_shape += args.n_actions
        actor_input_shape += args.n_agents

        # Critic input >> state + agent observation + agent id + other agents' actions + all agents' previous actions
        critic_input_shape = args.state_shape
        critic_input_shape += args.obs_shape
        critic_input_shape += args.n_agents
        critic_input_shape += args.n_actions * args.n_agents * 2

        return actor_input_shape, critic_input_shape

    def init_hidden(self):
        self.eval_hidden = self.actor.init_hidden().expand(self.args.n_agents, -1)

    def learn(self, episode: Episode, train_step: int, epsilon: float) -> float:
        self.init_hidden()

        q_values = self._train_critic(episode, train_step)
        action_prob = self._get_action_prob(episode, epsilon)

        q_taken = torch.gather(q_values, dim=2, index=episode.actions).squeeze(2)
        pi_taken = torch.gather(action_prob, dim=2, index=episode.actions).squeeze(2)
        log_pi_taken = torch.log(pi_taken)

        # Advantage calculation
        baseline = (q_values * action_prob).sum(dim=2, keepdim=True).squeeze(2).detach()
        advantage = (q_taken - baseline).detach()
        loss = - (advantage * log_pi_taken).sum()

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)  # clip gradients
        self.actor_optimizer.step()

        return loss.item()

    def _train_critic(self, episode: Episode, train_step: int):
        # Create an actions_next tensor by concatenating the action history, offset by 1, with a zero action
        # for the episode final timestep
        actions_next_offset = episode.actions[1:]
        padded_next_action = torch.zeros(*actions_next_offset[-1].shape, dtype=torch.long,
                                         device=self.args.device).unsqueeze(0)
        episode_actions_next = torch.cat((actions_next_offset, padded_next_action), dim=0)

        q_evals, q_next_target = self._get_q_values(episode)
        q_values = q_evals.clone()

        q_evals = torch.gather(q_evals, dim=2, index=episode.actions).squeeze(2)
        q_next_target = torch.gather(q_next_target, dim=2, index=episode_actions_next).squeeze(2)
        targets = td_lambda_target(episode, q_next_target.cpu(), self.args).to(self.args.device)

        td_error = targets.detach() - q_evals

        loss = (td_error ** 2).sum()

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), self.args.grad_norm_clip)
        self.critic_optimizer.step()

        if train_step and not train_step % self.args.target_update_cycle:
            self.target_critic.load_state_dict(self.online_critic.state_dict())

        return q_values

    def _get_critic_inputs(self, episode: Episode, transition_idx: int):
        # Replicate the episode action histories for each agent
        episode_actions_onehot_repeated = episode.actions_onehot[transition_idx].view((1, -1)).repeat(
            self.args.n_agents, 1)

        # If the first transition in the episode, create a zero action vector for the previous action
        if transition_idx == 0:
            episode_actions_onehot_last_repeated = torch.zeros_like(episode_actions_onehot_repeated).to(
                self.args.device)
        else:
            episode_actions_onehot_last_repeated = episode.actions_onehot[transition_idx - 1].view((1, -1)).repeat(
                self.args.n_agents, 1)

        # If the last transition in the episode, then create a zero action vector
        if transition_idx != episode.obs.shape[0] - 1:
            episode_actions_onehot_next = episode.actions_onehot[transition_idx + 1]
        else:
            episode_actions_onehot_next = torch.zeros(*episode.actions_onehot[0].shape).to(self.args.device)

        episode_actions_onehot_next_repeated = episode_actions_onehot_next.view((1, -1)).repeat(self.args.n_agents, 1)

        episode_state_expanded = episode.state[transition_idx].unsqueeze(0).expand(self.args.n_agents, -1)
        episode_state_next_expanded = episode.state_next[transition_idx].unsqueeze(0).expand(self.args.n_agents, -1)

        # Invert identity matrix to marginalise out the current agent's actions
        action_mask = torch.tensor(1) - torch.eye(self.args.n_agents)
        # Note: .repeat_interleave() mirrors numpy's .repeat() behaviour (repeats elements of the tensor)
        action_mask = action_mask.view(-1).repeat_interleave(self.args.n_actions).view(
            self.args.n_agents, -1).to(self.args.device)

        inputs, inputs_next = [], []

        inputs.append(episode.obs[transition_idx])
        inputs.append(episode_state_expanded)
        inputs.append(episode_actions_onehot_last_repeated)
        inputs.append(episode_actions_onehot_repeated * action_mask.unsqueeze(0))
        inputs.append(torch.eye(self.args.n_agents).to(self.args.device))
        inputs = torch.cat([X.reshape(self.args.n_agents, -1) for X in inputs], dim=1)

        inputs_next.append(episode.obs_next[transition_idx])
        inputs_next.append(episode_state_next_expanded)
        inputs_next.append(episode_actions_onehot_next_repeated)
        inputs_next.append(episode_actions_onehot_next_repeated * action_mask.unsqueeze(0))
        inputs_next.append(torch.eye(self.args.n_agents).to(self.args.device))
        inputs_next = torch.cat([X.reshape(self.args.n_agents, -1) for X in inputs_next], dim=1)

        return inputs.to(self.args.device), inputs_next.to(self.args.device)

    def _get_q_values(self, episode: Episode) -> torch.tensor:
        q_evals, q_targets = [], []

        for transition_idx in range(episode.obs.shape[0]):
            inputs, inputs_next = self._get_critic_inputs(episode, transition_idx)

            # Online net generates q values against the current state, target net against next state
            q_eval = self.online_critic(inputs).view(self.args.n_agents, -1)
            q_target = self.target_critic(inputs_next).view(self.args.n_agents, -1)

            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals_s = torch.stack(q_evals, dim=0)
        q_targets_s = torch.stack(q_targets, dim=0)

        return q_evals_s.to(self.args.device), q_targets_s.to(self.args.device)

    def _get_actor_inputs(self, episode: Episode, transition_idx: int):
        inputs = list()

        inputs.append(episode.obs[transition_idx])

        if transition_idx == 0:
            inputs.append(torch.zeros_like(episode.actions_onehot[transition_idx]).to(self.args.device))
        else:
            inputs.append(episode.actions_onehot[transition_idx - 1])

        inputs.append(torch.eye(self.args.n_agents).to(self.args.device))

        return torch.cat([X.reshape(self.args.n_agents, -1) for X in inputs], dim=1)

    def _get_action_prob(self, episode: Episode, epsilon: float) -> torch.tensor:
        transitions_action_prob = []

        for transition_idx in range(episode.obs.shape[0]):
            inputs = self._get_actor_inputs(episode, transition_idx)
            outputs, self.eval_hidden = self.actor(inputs, self.eval_hidden)

            outputs = outputs.view(self.args.n_agents, -1)
            transition_action_prob = torch.nn.functional.softmax(outputs, dim=-1)
            transitions_action_prob.append(transition_action_prob)

        transitions_action_prob_s = torch.stack(transitions_action_prob, dim=0).cpu()
        action_probs = ((1 - epsilon) * transitions_action_prob_s) + \
                       torch.ones_like(transitions_action_prob_s) * epsilon / self.args.n_actions
        return action_probs.to(self.args.device)


def td_lambda_target(episode: Episode, q_targets: torch.tensor, args) -> torch.tensor:
    episode_len = episode.obs.shape[0]

    terminated = ~episode.terminated.repeat(1, args.n_agents).cpu()
    reward_repeated = episode.rewards.repeat((1, args.n_agents)).cpu()  # expand central episode reward for each agent

    n_step_return = torch.zeros((episode_len, args.n_agents, episode_len))
    for transition_idx in range(episode_len-1, -1, -1):  # stepping backwards through episode
        # First n_step_return update initialised with the q_target estimate at that timestep
        n_step_return[transition_idx, :, 0] = reward_repeated[transition_idx] + \
                                              args.gamma * q_targets[transition_idx] * terminated[transition_idx]
        for n in range(1, episode_len - transition_idx):  # and then discounts this by gamma at each preceding timestep
            n_step_return[transition_idx, :, n] = reward_repeated[transition_idx] + \
                                                  args.gamma * n_step_return[transition_idx+1, :, n-1]

    lambda_return = torch.zeros((episode_len, args.n_agents))
    for transition_idx in range(episode_len):
        returns = torch.zeros(args.n_agents)
        for n in range(1, episode_len - transition_idx):
            returns += args.td_lambda**(n-1) * n_step_return[transition_idx, :, n-1]
        lambda_return[transition_idx] = (1-args.td_lambda)*returns + args.td_lambda**(episode_len-transition_idx-1) *\
            n_step_return[transition_idx, :, episode_len-transition_idx-1]
    return lambda_return
