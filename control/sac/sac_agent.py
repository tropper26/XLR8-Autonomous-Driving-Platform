import numpy as np
import torch as T
import torch.nn.functional as F

from control.reinforcement_learning.agents.base_agent import BaseAgent
from control.reinforcement_learning.networks.actor_network import SACActorNetwork
from control.reinforcement_learning.networks.critic_network import CriticNetwork
from control.reinforcement_learning.networks.value_network import ValueNetwork


class SACAgent(BaseAgent):
    def __init__(
        self,
        actor_lr: float,
        critic_lr: float,
        tau: float,
        state_dim: int,
        actions_dim: int,
        max_action_bounds: list[float],
        hidden_layer_dims: tuple[int, int] = (256, 256),
        gamma=0.99,
        batch_size=100,
        replay_buffer_max_size=1000000,
        reward_scale=5,
    ):
        super().__init__(
            state_dim=state_dim,
            actions_dim=actions_dim,
            gamma=gamma,
            tau=tau,
            replay_buffer_max_size=replay_buffer_max_size,
            batch_size=batch_size,
        )

        self.std_min = 1e-6

        self.actor = SACActorNetwork(
            name="actor",
            lr=actor_lr,
            layer_dims=(state_dim, hidden_layer_dims[0], hidden_layer_dims[1]),
            output_dim=actions_dim,
            std_min=self.std_min,
            norm_layers=False,
        )
        self.critic_1 = CriticNetwork(
            name="critic_1",
            lr=critic_lr,
            layer_dims=(
                state_dim + actions_dim,
                hidden_layer_dims[0],
                hidden_layer_dims[1],
            ),
            output_dim=1,
        )
        self.critic_2 = self.critic_1.full_copy("critic_2")

        self.value = ValueNetwork(
            name="value",
            lr=critic_lr,
            layer_dims=(state_dim, hidden_layer_dims[0], hidden_layer_dims[1]),
            output_dim=1,
        )
        self.target_value = self.value.full_copy("target_value")

        self.networks = [
            self.actor,
            self.critic_1,
            self.critic_2,
            self.value,
            self.target_value,
        ]

        self.reward_scale = reward_scale
        self.max_action_bounds = T.tensor(max_action_bounds, dtype=T.float).to(
            self.actor.device
        )
        self.update_network_parameters(self.value, self.target_value, 1.0)

    def choose_action(self, observation):
        observation = np.array([observation])
        state = T.Tensor(observation).to(self.actor.device)
        mu, sigma = self.actor.forward(state)
        actions, _ = self.sample_normal(mu=mu, sigma=sigma, reparameterize=False)
        # actions is an array of arrays due to the added dimension in state
        return actions.cpu().detach().numpy()[0]

    def sample_normal(self, mu, sigma, reparameterize=True):
        probabilities = T.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()  # reparameterizes the policy
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * self.max_action_bounds
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.actor.std_min)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        value = self.value.forward(state).view(-1)
        value_ = self.target_value.forward(state_).view(-1)
        value_[done] = 0.0

        mu, sigma = self.actor.forward(state)
        actions, log_probs = self.sample_normal(mu, sigma, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.sample_normal(mu, sigma, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.reward_scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters(self.value, self.target_value, self.tau)