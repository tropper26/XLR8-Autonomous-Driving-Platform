from control.base_params import BaseControllerParams
from dto.form_dto import FormDTO


class TD3Params(BaseControllerParams, FormDTO):
    def __init__(
        self,
        actor_lr=0.001,
        critic_lr=0.001,
        tau=0.005,
        hidden_layer_dims=(400, 300),
        gamma=0.99,
        batch_size=100,
        replay_buffer_max_size=1000000,
        noise=0.1,
        update_actor_interval=2,
        warmup=1000,
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
        self.noise = noise
        self.update_actor_interval = update_actor_interval
        self.warmup = warmup

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
            noise=saved_dict["noise"],
            update_actor_interval=saved_dict["update_actor_interval"],
            warmup=saved_dict["warmup"],
        )