import torch as T
import torch.nn.functional as F
import numpy as np

from control.reinforcement_learning.agents.base_agent import BaseAgent
from control.reinforcement_learning.networks.actor_network import ActorNetwork
from control.reinforcement_learning.networks.critic_network import CriticNetwork


class TD3Agent(BaseAgent):
    def __init__(
        self,
        actor_lr: float,
        critic_lr: float,
        tau: float,
        state_dim: int,
        actions_dim: int,
        max_action_bounds: list[float],
        min_action_bounds: list[float],
        hidden_layer_dims: tuple[int, int] = (400, 300),
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        batch_size=100,
        replay_buffer_max_size=1000000,
        noise=0.1,
    ):
        super().__init__(
            state_dim=state_dim,
            actions_dim=actions_dim,
            gamma=gamma,
            tau=tau,
            replay_buffer_max_size=replay_buffer_max_size,
            batch_size=batch_size,
        )

        self.max_action_bounds = max_action_bounds  # list of size action_dim representing the max value for each action
        self.min_action_bounds = min_action_bounds  # -||- min value -||-
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.actions_dim = actions_dim
        self.update_actor_interval = update_actor_interval
        self.noise = noise

        self.actor = ActorNetwork(
            name="actor",
            lr=actor_lr,
            layer_dims=(state_dim, hidden_layer_dims[0], hidden_layer_dims[1]),
            output_dim=actions_dim,
            norm_layers=False,
        )
        self.target_actor = self.actor.full_copy("target_actor")

        self.critic_1 = CriticNetwork(
            name="critic_1",
            lr=critic_lr,
            layer_dims=(
                state_dim + actions_dim,
                hidden_layer_dims[0],
                hidden_layer_dims[1],
            ),
            output_dim=1,
            norm_layers=False,
        )
        self.target_critic_1 = self.critic_1.full_copy("target_critic_1")
        self.critic_2 = self.critic_1.full_copy("critic_2")
        self.target_critic_2 = self.critic_1.full_copy("target_critic_2")

        self.networks = [
            self.actor,
            self.critic_1,
            self.critic_2,
            self.target_actor,
            self.target_critic_1,
            self.target_critic_2,
        ]

        self.max_action_bounds = T.tensor(max_action_bounds, dtype=T.float).to(
            self.actor.device
        )
        self.min_action_bounds = T.tensor(min_action_bounds, dtype=T.float).to(
            self.actor.device
        )

        self.update_network_parameters(self.actor, self.target_actor, 1)
        self.update_network_parameters(self.critic_1, self.target_critic_1, 1)
        self.update_network_parameters(self.critic_2, self.target_critic_2, 1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(
                np.random.normal(scale=self.noise, size=(self.actions_dim,)),
                device=self.actor.device,
            )
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(
            self.actor.device
        )

        # mu_prime = T.clamp(mu_prime, self.min_action_bounds, self.max_action_bounds)
        # Element-wise clamping
        mu_prime = T.max(
            T.min(mu_prime, self.max_action_bounds), self.min_action_bounds
        )
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

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

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(
            T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5
        )
        target_actions = T.max(
            T.min(target_actions, self.max_action_bounds), self.min_action_bounds
        )

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # self.update_network_parameters()
        self.update_network_parameters(self.actor, self.target_actor, self.tau)
        self.update_network_parameters(self.critic_1, self.target_critic_1, self.tau)
        self.update_network_parameters(self.critic_2, self.target_critic_2, self.tau)