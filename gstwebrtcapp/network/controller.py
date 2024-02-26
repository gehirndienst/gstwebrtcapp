import asyncio
import enum
import subprocess
import shlex
import random
from typing import List

from utils.base import LOGGER


class NetworkScenario(enum.Enum):
    GOOD = "good"
    OK = "ok"
    BAD = "bad"


class NetworkController:
    '''
    NetworkController class is responsible for controlling the network interface.
    It can apply rules to the network interface to simulate different network scenarios.
    So far, it can simulate good, ok, and bad network scenarios by restricting
    the bandwidth of the network interface accordingly.
    '''

    def __init__(
        self,
        gt_bandwidth: float,  # mbps
        interval: float = 10.0,  # sec
        interface: str = "eth0",
        scenario_weights: List[float] | None = None,
    ) -> None:
        self.interval = interval
        self.gt_bandwidth = gt_bandwidth
        self.interface = interface
        self._update_weights(scenario_weights)

        self.rules = []
        self.current_rule = ""
        self.is_fix_current_rule = False

    async def update_network_rule(self) -> None:
        while True:
            if not self.is_fix_current_rule:
                if self.rules:
                    rule = self.rules.pop(0)
                    self._apply_rule(rule)
                else:
                    self._apply_rule(self._generate_rule(self._get_scenario()))
            await asyncio.sleep(self.interval)

    def set_rule(self, rule: str, is_fix: bool = True) -> None:
        self._apply_rule(rule)
        self.is_fix_current_rule = is_fix

    def reset_rule(self) -> None:
        self._delete_rules()
        self.current_rule = ""
        self.is_fix_current_rule = False

    def generate_rules(self, count: int, weights: List[float] | None) -> None:
        if weights:
            self._update_weights(weights)
        self.rules = []
        for _ in range(count):
            self.rules.append(self._generate_rule())
        LOGGER.info(f"NetworkController: {count} rules with weights {weights} generated")

    def _apply_rule(self, rule: str) -> None:
        self._delete_rules()
        subprocess.run(shlex.split(self._make_tcset_cmd(rule)))
        LOGGER.info(f"NetworkController: Rule applied, tcset cmd: --{rule}")

    def _delete_rules(self) -> None:
        subprocess.run(shlex.split(f"tcdel {self.interface} --all"))

    def _generate_rule(self, scenario: NetworkScenario | None = None) -> str:
        if scenario is None:
            scenario = self._get_scenario()
        if scenario == NetworkScenario.GOOD:
            rate_range = (self.gt_bandwidth * 0.8, self.gt_bandwidth * 0.99)
        elif scenario == NetworkScenario.OK:
            rate_range = (self.gt_bandwidth * 0.4, self.gt_bandwidth * 0.8)
        elif scenario == NetworkScenario.BAD:
            rate_range = (self.gt_bandwidth * 0.05, self.gt_bandwidth * 0.2)
        rate_value = random.uniform(rate_range[0], rate_range[1])
        return f"rate {rate_value}Mbps"

    def _get_scenario(self) -> NetworkScenario:
        return (
            random.choices(list(NetworkScenario), weights=self.scenario_weights)[0]
            if self.scenario_weights is not None
            else random.choice(list(NetworkScenario))
        )

    def _update_weights(self, new_weights: List[float] | None) -> None:
        if new_weights is None:
            self.scenario_weights = None
            return

        if abs(sum(new_weights) - 1.0) < 1e-6 and len(new_weights) == 3:
            self.scenario_weights = new_weights
        else:
            LOGGER.warning("NetworkController: Given scenario weights are not valid, ignore given weights.")

    def _make_tcset_cmd(self, rule: str) -> str:
        self.current_rule = rule
        return f"tcset {self.interface} --{self.current_rule}"
