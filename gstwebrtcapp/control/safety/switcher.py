import collections
from dataclasses import dataclass

from control.agent import Agent


@dataclass
class SwitcherConfig:
    k: int = 0.05
    window_size: int = 10
    recover_iterations: int = 8
    switch_forgives: int | None = None


class SwitchingPair:
    def __init__(self, safe_id: str, unsafe_id: str) -> None:
        self.safe_id = safe_id
        self.unsafe_id = unsafe_id
        self.is_warmups_resetted = False


class Switcher:
    def __init__(
        self,
        config: SwitcherConfig = SwitcherConfig(),
    ) -> None:
        self.window_size = 0
        self.max_window_size = config.window_size

        self.values = collections.deque(maxlen=self.max_window_size)
        self.threshold = 0.0
        self.k = config.k

        self.recover_iterations = 0
        self.max_recover_iterations = config.recover_iterations

        self.switch_forgives = 0
        self.max_switch_forgives = config.switch_forgives

        self.algo = 1  # default "unsafe"

    def switch(self, val: float) -> int:
        self.values.append(val)
        weighted_average = self._get_weighted_average_over_window()
        self._update_threshold(weighted_average)
        is_exceed = weighted_average > self.threshold
        if self._should_switch(is_exceed):
            self.algo = 1 if self.algo == 0 else 0
        return self.algo

    def _should_switch(self, is_exceed: bool = False) -> bool:
        # too few collected to switch
        if len(self.values) < self.max_window_size:
            return False

        if self.algo == 0:
            # if safe is on check whether there is enough consecutive "recovering" iterations
            if self.recover_iterations < self.max_recover_iterations:
                if not is_exceed:
                    self.recover_iterations += 1
                else:
                    self.recover_iterations = 0
                return False
            else:
                self.recover_iterations = 0
                return True
        else:
            # if safe is off check whether the threshold is exceed
            if not is_exceed:
                return False
            else:
                # if max_switch_forgives is set, then check whether there is enough "forgiving" iterations
                if self.max_switch_forgives is not None:
                    self.switch_forgives += 1
                    if self.switch_forgives < self.max_switch_forgives:
                        return False
                    else:
                        self.switch_forgives = 0
                        return True
                else:
                    return True

    def _get_weighted_average_over_window(self) -> float:
        if self.window_size < self.max_window_size:
            self.window_size = len(self.values)
        weighted_sum = 0.0
        for i in range(1, self.window_size):
            weight = 2**-i
            weighted_sum += weight * (self.values[-i] - self.values[-i - 1])
        return weighted_sum

    def _update_threshold(self, weighted_average: float) -> None:
        self.threshold = self.threshold + self.k * (abs(weighted_average) - self.threshold)
