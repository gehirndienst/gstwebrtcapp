import gymnasium
import json
from inspect import signature
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC, DQN
from sb3_contrib import QRDQN, RecurrentPPO
import torch
from typing import Any, Optional, Dict, Type


SB3_MODEL_CLASSES = {
    'ppo': PPO,
    'ppo_lstm': RecurrentPPO,
    'a2c': A2C,
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
    'qrdqn': QRDQN,
    'dqn': DQN,
}

ACTIVATION_FUNCTIONS = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'elu': torch.nn.ELU,
    'gelu': torch.nn.GELU,
    'glu': torch.nn.GLU,
}


class DrlModelConfigurator:
    """
    A class which configures SB3 DRL model. It could take config from a file or a dict or merge them together.
    It uses introspection to validate given params against class params of a selected model.

    :model_name: SB3 model name, should be one of SB3_MODEL_CLASSES keys
    :param model_hyperparams_file: The path to the file containing model hyperparameters in JSON format (nullalble)
    :param kwargs: Additional keyword arguments representing hyperparameters
    """

    def __init__(
        self,
        model_name: str = "ppo",
        model_hyperparams_file: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name.lower()
        self.model_class = self.get_model_class()
        self.model_class_args = list(signature(self.model_class.__init__).parameters.keys())

        if model_hyperparams_file is not None:
            with open(model_hyperparams_file, "r") as f:
                model_params = self._parse(json.load(f))
                for key, value in kwargs.items():
                    if key in self.model_class_args:
                        model_params[key] = value
        else:
            model_params = self._parse(kwargs)
            for key, value in model_params.items():
                if key in self.model_class_args:
                    model_params[key] = value
        self.model_params = model_params

    def make_model(self, env: gymnasium.Env, default_policy: str = 'MultiInputPolicy') -> BaseAlgorithm:
        if 'policy' not in self.model_params:
            self.model_params['policy'] = default_policy
        self.model = self.model_class(env=env, **self.model_params)
        return self.model

    def get_model_class(self) -> Type[BaseAlgorithm]:
        if self.model_name not in SB3_MODEL_CLASSES:
            raise ValueError(f"Model {self.model_name} is not found!")
        else:
            return SB3_MODEL_CLASSES[self.model_name]

    def _parse(self, dict_params: Dict[str, Any]) -> Dict[str, Any]:
        # typecast str-like activation function to torch.nn.Module object
        if 'activation_fn' in dict_params["policy_kwargs"]:
            activation_fn_name = dict_params["policy_kwargs"]["activation_fn"].lower()
            if activation_fn_name in ACTIVATION_FUNCTIONS:
                dict_params["policy_kwargs"]["activation_fn"] = ACTIVATION_FUNCTIONS[activation_fn_name]
            else:
                raise ValueError(f"Activation function {activation_fn_name} is not found!")
        # other typecasting/parsing...
        return dict_params
