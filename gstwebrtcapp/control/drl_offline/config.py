from dataclasses import dataclass


@dataclass
class DrlOfflineConfig:
    """
    A data class to hold DRL config.

    :param model_file: The file containing the D3RLPY model. Nullable
    :param episodes: The number of episodes to run. -1 means run indefinitely
    :param episode_length: The number of steps per episode
    :param state_update_interval: The interval between the state updates in seconds
    :param state_max_inactivity_time: The maximum time in seconds to wait for the state update. If exceeded, the episode is terminated
    :param save_log_path: The path to save the logs
    :param device: The device to run the model on. Nullable
    """

    model_file: str | None = None
    episodes: int = -1
    episode_length: int = 256
    state_update_interval: float = 3.0
    state_max_inactivity_time: float = 60.0
    save_log_path: str = './logs'
    device: str | None = None
