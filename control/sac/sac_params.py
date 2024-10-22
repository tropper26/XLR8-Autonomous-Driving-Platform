from control.base_params import BaseControllerParams
from dto.form_dto import FormDTO


class SACParams(BaseControllerParams, FormDTO):
    def __init__(
        self,
        actor_lr=0.0003,
        critic_lr=0.0003,
        tau=0.005,
        hidden_layer_dims=(256, 256),
        gamma=0.99,
        batch_size=256,
        replay_buffer_max_size=1000000,
        reward_scale=2,
        tuning_params=None,
    ):
        super().__init__(tuning_params=tuning_params)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.hidden_layer_dims = hidden_layer_dims
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer_max_size = replay_buffer_max_size
        self.reward_scale = reward_scale

    def attributes_to_ignore(self):
        return ["tuning_params"]

    @classmethod
    def from_dict(cls, saved_dict: dict):
        return cls(
            actor_lr=saved_dict["actor_lr"],
            critic_lr=saved_dict["critic_lr"],
            tau=saved_dict["tau"],
            hidden_layer_dims=saved_dict["hidden_layer_dims"],
            gamma=saved_dict["gamma"],
            batch_size=saved_dict["batch_size"],
            replay_buffer_max_size=saved_dict["replay_buffer_max_size"],
            reward_scale=saved_dict["reward_scale"],
        )