import time
from typing import Tuple

from control.agent import Agent, AgentType
from control.drl.config import DrlConfig
from control.drl.manager import DrlManager
from control.drl.mdp import MDP
from message.client import MqttConfig
from utils.base import LOGGER


class DrlAgent(Agent):
    def __init__(
        self,
        drl_config: DrlConfig,
        mdp: MDP,
        mqtt_config: MqttConfig,
        warmup: float = 10.0,
    ) -> None:
        super().__init__(mqtt_config)
        self.warmup = warmup
        self.type = AgentType.DRL
        self.manager = DrlManager(drl_config, mdp, self.mqtts)

    def run(self, is_load_last_model: bool = False) -> None:
        super().run()
        time.sleep(self.warmup)
        LOGGER.info(f"INFO: DRL Agent warmup {self.warmup} sec is finished, starting...")

        self.manager.reset(is_load_last_model)
        if self.manager.config.mode == "train":
            self.manager.train()
        elif self.manager.config.mode == "eval":
            self.manager.eval()
        else:
            raise Exception(f"Unknown DRL mode {self.manager.config.mode}")

    def stop(self) -> None:
        super().stop()
        LOGGER.info("INFO: stopping DRL agent...")
        self.manager.stop()
