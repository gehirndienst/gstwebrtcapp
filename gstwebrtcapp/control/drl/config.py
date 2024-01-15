from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DrlConfig:
    """
    A data class to hold DRL config.

    :param str mode: Either 'train' or 'eval'
    :param model_file: The file containing the DRL model. Nullable
    :param model_name: The name of the DRL model
    :param episodes: The number of episodes to run
    :param episode_length: The number of steps per episode
    :param state_update_interval: The interval between the state updates in seconds
    :param hyperparams_cfg: Hyperparameters configuration: either a path to a json gile or a dictionary. Nullable
    :param deterministic: Whether the DRL model should be deterministic
    :param callbacks: Optional list of callbacks for SB3 model given as string aliases. One of 'save_model', 'save_step', 'print_step'. Nullable
    :param save_model_path: The path to save the DRL model
    :param save_log_path: The path to save the DRL logs
    :param verbose: The verbosity level. One of 0, 1, 2
    """

    mode: str = 'train'
    model_file: str = None
    model_name: str = 'sac'
    episodes: int = 30
    episode_length: int = 512
    state_update_interval: float = 1.0
    hyperparams_cfg: str | Dict[str, Any] | None = None
    deterministic: bool = False
    callbacks: List[str] | None = None
    save_model_path: str = './models'
    save_log_path: str = './logs'
    verbose: int = 1
