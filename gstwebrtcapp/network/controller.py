import asyncio
import csv
from datetime import datetime
import enum
import os
import subprocess
import shlex
import random
from typing import List, Tuple

from gstwebrtcapp.network.trace import NetworkTrace
from gstwebrtcapp.utils.base import LOGGER, extract_network_traces_from_csv


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
        interval: float | Tuple[float, float] = (10.0, 60.0),  # sec
        interface: str = "eth0",
        scenario_weights: List[float] | None = None,
        additional_rule_str: str = "",  # either --delay ..., or --loss ...
        is_stop_after_no_rule: bool = False,
        log_path: str | None = None,
        warmup: float = 10.0,
    ) -> None:
        self.interval = (interval, interval) if isinstance(interval, float) else interval
        self.gt_bandwidth = gt_bandwidth
        self.interface = interface
        self._update_weights(scenario_weights)
        self.is_stop_after_no_rule = is_stop_after_no_rule
        self.additional_rule_str = additional_rule_str
        self.warmup = warmup

        self.log_path = log_path
        self.csv_file = None
        self.csv_handler = None
        self.csv_writer = None

        self.rules = []
        self.current_rule = ""
        self.current_cmd = ""
        self.is_fix_current_rule = False

    async def update_network_rule(self) -> None:
        await asyncio.sleep(self.warmup)
        cancelled = False
        while not cancelled:
            try:
                if not self.is_fix_current_rule:
                    if self.rules:
                        rule = self.rules.pop(0)
                        self._apply_rule(rule)
                    else:
                        if self.is_stop_after_no_rule:
                            raise asyncio.CancelledError
                        else:
                            self._apply_rule(self._generate_rule(self._get_scenario()))
                if self.log_path is not None:
                    self._save_rule_to_csv()
                await asyncio.sleep(random.uniform(*self.interval))
            except asyncio.CancelledError:
                cancelled = True
                self.reset_rule()
                if self.csv_handler is not None:
                    self.csv_handler.close()
                    self.csv_handler = None
                    self.csv_writer = None

    def set_rule(self, rule: str, is_fix: bool = True) -> None:
        self._apply_rule(rule)
        self.is_fix_current_rule = is_fix

    def reset_rule(self) -> None:
        self._delete_rules()
        self.current_rule = ""
        self.current_cmd = ""
        self.is_fix_current_rule = False

    def generate_rules(self, count: int, weights: List[float] | None = None) -> None:
        if weights is not None:
            self._update_weights(weights)
        self.rules = []
        for _ in range(count):
            self.rules.append(self._generate_rule())
        LOGGER.info(f"NetworkController: {count} rules with weights {weights} generated")

    def generate_rules_from_traces(self, trace_folder: str, is_curriculum_learning: bool = False) -> None:
        network_traces: List[NetworkTrace] = []
        for filename in os.listdir(trace_folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(trace_folder, filename)
                bw_values, ooc_rate = extract_network_traces_from_csv(filepath)
                size = len(bw_values)
                av_value = sum(bw_values) / size if size > 0 else 0
                network_trace = NetworkTrace(size=size, av_value=av_value, ooc_rate=ooc_rate, values=bw_values)
                network_traces.append(network_trace)

        if is_curriculum_learning:
            # sort by complexity (ooc_rate is when the bw lower than 1 mbps)
            network_traces = sorted(network_traces, key=lambda x: x.ooc_rate)

        self.rules = []
        for network_trace in network_traces:
            for bw_value in network_trace.values:
                bw_value = max(bw_value, 0.1)
                self.rules.append(f"rate {bw_value}Mbps")

    def _apply_rule(self, rule: str) -> None:
        self._delete_rules()
        self._make_tcset_cmd(rule)
        subprocess.run(shlex.split(self.current_cmd))
        LOGGER.info(f"NetworkController: Rule applied, tcset cmd: {self.current_cmd}")

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

    def _make_tcset_cmd(self, rule: str) -> None:
        self.current_rule = rule
        self.current_cmd = f"tcset {self.interface} --{self.current_rule} {self.additional_rule_str}"

    def _save_rule_to_csv(self) -> None:
        datetime_now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")[:-3]
        if self.csv_handler is None:
            os.makedirs(self.log_path, exist_ok=True)
            if self.csv_file is None:
                self.csv_file = os.path.join(self.log_path, f"network_rules_{datetime_now}.csv")
            header = ["timestamp", "rule", "additional_rule"]
            self.csv_handler = open(self.csv_file, mode="a", newline="\n")
            self.csv_writer = csv.DictWriter(self.csv_handler, fieldnames=header)
            if os.stat(self.csv_file).st_size == 0:
                self.csv_writer.writeheader()
            self.csv_handler.flush()
        else:
            row = {
                "timestamp": datetime_now,
                "rule": f"--{self.current_rule}",
                "additional_rule": self.additional_rule_str,
            }
            self.csv_writer.writerow(row)
