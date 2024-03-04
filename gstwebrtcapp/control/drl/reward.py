from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any, Deque, Dict, OrderedDict, Tuple

from utils.base import get_list_average


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        self.state = None
        self.prev_state = None
        self.reward_parts = None

    @abstractmethod
    def calculate_reward(self, states: Deque[OrderedDict[str, Any]]) -> Tuple[float, Dict[str, Any | float] | None]:
        if len(states) == 0:
            return 0.0, dict(zip(self.reward_parts, [0.0] * len(self.reward_parts)))
        elif len(states) == 1:
            self.state = states[-1]
            self.prev_state = None
        else:
            self.state = states[-1]
            self.prev_state = states[-2]


class QoePaper(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_parts = ["rew", "rate", "rtt", "plr", "jit", "smt", "pli", "nack"]

    def calculate_reward(self, states: Deque[OrderedDict[str, Any]]) -> Tuple[float, Dict[str, Any | float] | None]:
        super().calculate_reward(states)

        # 1. rate
        reward_rate = np.log((np.exp(1) - 1) * (self.state["rxRate"]) + 1)

        # 2. rtt
        rtts = [s["fractionRtt"] for s in states]
        rtt_max = max(rtts) if len(rtts) > 0 else 0.0
        rtt_min = min(rtts) if len(rtts) > 0 else 0.0
        reward_rtt = (rtt_max - self.state["fractionRtt"]) / (rtt_max - rtt_min) if rtt_max - rtt_min > 0 else 0.0

        # 3. plr
        reward_plr = 1 - self.state["fractionLossRate"]

        # 4. jitter
        jitter_scaled = min(self.state["interarrivalJitter"] * 1000, 400) / 16
        reward_jitter = -0.2 * np.sqrt(jitter_scaled) + 1

        # 5. smooth: take rate of change
        last_rate = self.prev_state["rxRate"] if self.prev_state is not None else 0.0
        rate_of_change = abs(self.state["rxRate"] - last_rate)
        # don't penalize if bitrate changes less than 10% or if it's the first state
        reward_smooth = 1 if rate_of_change <= 0.1 or last_rate == 0.0 else 1 - rate_of_change

        # 6. penalties for NACK and PLI rates
        penalty_nack = min(0.5, self.state["fractionNackRate"] * 2)
        penalty_pli = min(0.5, self.state["fractionPliRate"] * 20)

        # rewards/penalties coefficients
        a = 0.3
        b = 0.2
        c = 0.3
        d = 0.12
        e = 0.08
        f = -1
        e = -1
        reward = (
            a * reward_rate
            + b * reward_rtt
            + c * reward_plr
            + d * reward_jitter
            + e * reward_smooth
            + f * penalty_nack
            + e * penalty_pli
        )
        return reward, dict(
            zip(
                self.reward_parts,
                [
                    reward,
                    a * reward_rate,
                    b * reward_rtt,
                    c * reward_plr,
                    d * reward_jitter,
                    e * reward_smooth,
                    f * penalty_nack,
                    e * penalty_pli,
                ],
            )
        )


class QoeAhoy(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_parts = ["rew", "rate", "rtt", "plr", "jit", "smt", "pli", "nack"]

    def calculate_reward(self, states: Deque[OrderedDict[str, Any]]) -> Tuple[float, Dict[str, Any | float] | None]:
        super().calculate_reward(states)

        # 1. rate: 0...0.2
        reward_rate = np.log((np.exp(1) - 1) * (self.state["rxGoodput"]) + 1)
        reward_rate *= 0.2

        # 2. rtt: 0...0.2
        # 2.1. mean for the last N states - current rtt
        # calculate mean rtt for the last states except the current one
        rtt_sum = 0.0
        for i in range(len(states) - 1):
            rtt_sum += states[i]["fractionRtt"]
        rtt_avg = rtt_sum / (len(states) - 1) if len(states) > 1 else self.state["fractionRtt"]
        sub_reward_avg_curr_diff_rtt = rtt_avg - self.state["fractionRtt"]
        # 2.2. prev - current rtt
        sub_reward_prev_curr_diff_rtt = 2 * (
            self.prev_state["fractionRtt"] - self.state["fractionRtt"] if self.prev_state is not None else 0.0
        )
        # final
        sub_sum_rtt = sub_reward_avg_curr_diff_rtt + sub_reward_prev_curr_diff_rtt
        # if >= 0 then it is perfect and give the max reward 0, if less then penalize until -0.4.
        # final reward is bw 0..0.2
        final_sum_rtt = 0 if sub_sum_rtt >= 0 else max(-0.4, sub_sum_rtt)
        reward_rtt = 0.4 + final_sum_rtt
        reward_rtt *= 0.5

        # 3. plr: 0...0.2
        # plr is not so often but very deadly, so penalize more. Set 20% to be the most critical
        reward_plr = max(0, 1 - 5 * self.state["fractionLossRate"])
        reward_plr *= 0.2

        # 4. jitter: 0...0.1
        # max 250 ms, more than that is very bad, 10 ms jitter is considered to be acceptable
        thresholded_jitter = max(0, self.state["interarrivalRttJitter"] - 0.01)
        reward_jitter = max(0, 0.5 - np.sqrt(thresholded_jitter))
        reward_jitter *= 0.2

        # 5. smooth: take rate of change: 0...0.1
        rate_prev = self.prev_state["rxGoodput"] if self.prev_state is not None else 0.0
        rate_of_change = abs(self.state["rxGoodput"] - rate_prev)
        # don't penalize if bitrate changes less than 10% or if it's the first state
        reward_smooth = 1 if rate_of_change <= 0.1 or rate_prev == 0.0 else 1 - rate_of_change
        reward_smooth *= 0.1

        # 6. pli rate should not be higher than 0.1%: 0..0.05
        reward_pli = max(0, 1 - (self.state["fractionPliRate"] * 1000))
        reward_pli *= 0.05

        # 7. nack rate should not be higher than 5%: 0..0.05
        reward_nack = max(0, 1 - (self.state["fractionNackRate"] * 20))
        reward_nack *= 0.05

        # final
        reward = reward_rate + reward_rtt + reward_plr + reward_jitter + reward_smooth + reward_pli + reward_nack
        reward = np.clip(reward, 0, 1)
        # ! extra cases:
        # 1. if plr > 20% then reward = 0
        # 2. if rtt > 500ms then reward = 0
        # 3. if rxRate / txRate < 0.2 then reward = 0
        # 4. if jitter > 250ms then reward = 0
        # 5. if plir > 1% then reward = 0
        if (
            self.state["fractionLossRate"] > 0.2
            or self.state["fractionRtt"] > 0.5
            or (self.state["txGoodput"] > 0 and self.state["rxGoodput"] / self.state["txGoodput"] < 0.2)
            or self.state["interarrivalRttJitter"] > 0.25
            or self.state["fractionPliRate"] > 0.01
        ):
            reward = 0.0

        return reward, dict(
            zip(
                self.reward_parts,
                [
                    reward,
                    reward_rate,
                    reward_rtt,
                    reward_plr,
                    reward_jitter,
                    reward_smooth,
                    reward_pli,
                    reward_nack,
                ],
            )
        )


class QoeAhoySeq(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_parts = ["rew", "rate", "rtt", "plr", "jit", "smt", "pli", "nack"]

    def calculate_reward(self, states: Deque[OrderedDict[str, Any]]) -> Tuple[float, Dict[str, Any | float] | None]:
        super().calculate_reward(states)

        # 1. rate: 0...0.2
        reward_rate = np.log((np.exp(1) - 1) * (get_list_average(self.state["rxGoodput"])) + 1)
        reward_rate *= 0.2

        # 2. rtt: 0...0.2
        # 2.1. mean for the last N states - current rtt
        # calculate mean rtt for the last states except the current one
        fraction_rtt = get_list_average(self.state["fractionRtt"])
        sub_reward_avg_curr_diff_rtt = self.state["rttMean"] - fraction_rtt
        # 2.2. prev - current rtt
        sub_reward_prev_curr_diff_rtt = 2 * (
            get_list_average(self.prev_state["fractionRtt"]) - fraction_rtt if self.prev_state is not None else 0.0
        )
        # final
        sub_sum_rtt = sub_reward_avg_curr_diff_rtt + sub_reward_prev_curr_diff_rtt
        # if >= 0 then it is perfect and give the max reward 0, if less then penalize until -0.4.
        # final reward is bw 0..0.2
        final_sum_rtt = 0 if sub_sum_rtt >= 0 else max(-0.4, sub_sum_rtt)
        reward_rtt = 0.4 + final_sum_rtt
        reward_rtt *= 0.5

        # 3. plr: 0...0.2
        # plr is not so often but very deadly, so penalize more. Set 20% to be the most critical
        fraction_loss_rate = get_list_average(self.state["fractionLossRate"])
        reward_plr = max(0, 1 - 5 * fraction_loss_rate)
        reward_plr *= 0.2

        # 4. jitter: 0...0.1
        # max 250 ms, more than that is very bad, 10 ms jitter is considered to be acceptable
        jitter = max(self.state["interarrivalRttJitter"])
        thresholded_jitter = max(0, jitter - 0.01)
        reward_jitter = max(0, 0.5 - np.sqrt(thresholded_jitter))
        reward_jitter *= 0.2

        # 5. smooth: take rate of change: 0...0.1
        rx_rate_prev = get_list_average(self.prev_state["rxGoodput"]) if self.prev_state is not None else 0.0
        rx_rate = get_list_average(self.state["rxGoodput"])
        rate_of_change = abs(rx_rate - rx_rate_prev)
        # don't penalize if bitrate changes less than 10% or if it's the first state
        reward_smooth = 1 if rate_of_change <= 0.1 or rx_rate_prev == 0.0 else 1 - rate_of_change
        reward_smooth *= 0.1

        # 6. pli rate should not be higher than 0.1%: 0..0.05
        pli_rate = get_list_average(self.state["fractionPliRate"])
        reward_pli = max(0, 1 - (pli_rate * 1000))
        reward_pli *= 0.05

        # 7. nack rate should not be higher than 5%: 0..0.05
        nack_rate = get_list_average(self.state["fractionNackRate"])
        reward_nack = max(0, 1 - (nack_rate * 20))
        reward_nack *= 0.05

        # final
        reward = reward_rate + reward_rtt + reward_plr + reward_jitter + reward_smooth + reward_pli + reward_nack
        reward = np.clip(reward, 0, 1)
        # ! extra cases:
        # 1. if plr > 20% then reward = 0
        # 2. if rtt > 500ms then reward = 0
        # 3. if rxRate / txRate < 0.2 then reward = 0
        # 4. if jitter > 250ms then reward = 0
        # 5. if plir > 1% then reward = 0
        if (
            fraction_loss_rate > 0.2
            or fraction_rtt > 0.5
            or (
                get_list_average(self.state["txGoodput"]) > 0
                and rx_rate / get_list_average(self.state["txGoodput"]) < 0.2
            )
            or jitter > 0.25
            or pli_rate > 0.01
        ):
            reward = 0.0

        return reward, dict(
            zip(
                self.reward_parts,
                [
                    reward,
                    reward_rate,
                    reward_rtt,
                    reward_plr,
                    reward_jitter,
                    reward_smooth,
                    reward_pli,
                    reward_nack,
                ],
            )
        )


class RewardFunctionFactory:
    reward_functions = {
        "qoe_paper": QoePaper,
        "qoe_ahoy": QoeAhoy,
        "qoe_ahoy_seq": QoeAhoySeq,
        # add more reward function classes as needed
    }

    @classmethod
    def create_reward_function(cls, name: str, *args, **kwargs) -> RewardFunction:
        if name in cls.reward_functions:
            return cls.reward_functions[name](*args, **kwargs)
        else:
            raise ValueError(f"Reward function '{name}' not found in the factory dictionary: {cls.reward_functions}")
