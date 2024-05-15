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
        id: str = "drl",
        warmup: float = 10.0,
    ) -> None:
        super().__init__(mqtt_config, id, warmup)
        self.type = AgentType.DRL
        self.manager = DrlManager(drl_config, mdp, self.mqtts)

    def run(self, is_load_last_model: bool = False) -> None:
        super().run()
        time.sleep(self.warmup)
        self.mqtts.subscriber.subscribe([self.mqtts.subscriber.topics.actions])
        self.mqtts.subscriber.clean_message_queue(self.mqtts.subscriber.topics.gcc)
        self.is_running = True
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
