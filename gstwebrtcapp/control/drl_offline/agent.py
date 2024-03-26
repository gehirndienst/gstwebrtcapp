import os
import time
from typing import Any, OrderedDict

import d3rlpy
import numpy as np
import torch

from control.agent import Agent, AgentType
from control.drl.mdp import MDP
from control.drl.env import DrlEnv
from control.drl_offline.config import DrlOfflineConfig
from message.client import MqttConfig
from utils.base import LOGGER


class DrlOfflineAgent(Agent):
    def __init__(
        self,
        drl_offline_config: DrlOfflineConfig,
        mdp: MDP,
        mqtt_config: MqttConfig,
        warmup: float = 20.0,
    ) -> None:
        super().__init__(mqtt_config)

        self.drl_offline_config = drl_offline_config
        self.mdp = mdp
        self.warmup = warmup
        self.type = AgentType.DRL_OFFLINE

        self.model = None
        self.env = None
        self.is_episode_done = False
        self.is_running = False

    def run(self, _) -> None:
        super().run()
        time.sleep(self.warmup)
        LOGGER.info(f"INFO: DrlOfflineAgent warmup {self.warmup} sec is finished, starting...")

        self._setup()

        while self.is_running:
            for _ in (
                range(self.drl_offline_config.episodes) if self.drl_offline_config.episodes != -1 else iter(int, 1)
            ):
                state = self._to_d3rlpy_state(self.env.reset()[0])
                reward = 0.0
                # TODO: add logging
                while not self.is_episode_done:
                    action = self.model.predict(state, reward)
                    state_gym, reward, term, trunc, _ = self.env.step(action)
                    state = self._to_d3rlpy_state(state_gym)
                    self.is_episode_done = term or trunc

    def stop(self) -> None:
        super().stop()
        self.is_episode_done = True
        self.is_running = False
        LOGGER.info("INFO: stopping DrlOffline agent...")

    def _setup(self) -> None:
        if not os.path.isfile(self.drl_offline_config.model_file):
            raise FileNotFoundError(f"DrlOfflineAgent: Model file {self.drl_offline_config.model_file} not found!")
        else:
            m = d3rlpy.load_learnable(
                self.drl_offline_config.model_file,
                (
                    self.drl_offline_config.device
                    if self.drl_offline_config.device
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                ),
            )
            self.model = m.as_stateful_wrapper(target_return=0)

        self.env = DrlEnv(
            mdp=self.mdp,
            mqtts=self.mqtts,
            max_episodes=self.drl_offline_config.episodes,
            state_update_interval=self.drl_offline_config.state_update_interval,
            max_inactivity_time=self.drl_offline_config.state_max_inactivity_time,
        )

        self.is_running = True

    def _to_d3rlpy_state(self, state: OrderedDict[str, Any]) -> np.ndarray:
        concatenated_values = np.array([v for values_list in state.values() for v in values_list])
        # return np.expand_dims(concatenated_values, axis=0).astype(np.float32)
        return concatenated_values.astype(np.float32)
