import asyncio
import logging
import time
from typing import Any, Callable, Dict, List

# logger
LOGGER = logging
LOGGER.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# exceptions
class GSTWEBRTCAPP_EXCEPTION(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self):
        return f"{self.args[0]}"


# general utils
def wait_for_condition(
    condition_func: Callable[[], bool],
    timeout_sec: int,
    sleeping_time_sec: float = 0.1,
) -> bool:
    """
    Wait for condition_func to return True or timeout_sec is reached

    :param condition_func: callable that returns bool
    :param timeout_sec: timeout in seconds
    :param sleeping_time_sec: meanwhile sleeping time in seconds
    :return: True if condition_func returned True, False otherwise
    :raises TimeoutError: if timeout_sec is reached
    """
    start_time = time.time()
    while not condition_func():
        if time.time() - start_time >= float(timeout_sec) and timeout_sec >= 0:
            raise TimeoutError(f"Timeout {timeout_sec} sec is reached for condition {condition_func.__name__}")
        time.sleep(sleeping_time_sec)
    return True


async def async_wait_for_condition(
    condition_func: Callable[[], bool],
    timeout_sec: int,
    sleeping_time_sec: float = 0.1,
) -> bool:
    """
    Asynchronously wait for condition_func to return True or timeout_sec is reached

    :param condition_func: callable that returns bool
    :param timeout_sec: timeout in seconds
    :param sleeping_time_sec: meanwhile sleeping time in seconds
    :return: True if condition_func returned True, False otherwise
    :raises TimeoutError: if timeout_sec is reached
    """
    start_time = time.time()
    while not condition_func():
        if time.time() - start_time >= float(timeout_sec) and timeout_sec >= 0:
            raise TimeoutError(f"Timeout {timeout_sec} sec is reached for condition {condition_func.__name__}")
        await asyncio.sleep(sleeping_time_sec)
    return True


def scale(val: int | float, min: int | float, max: int | float) -> int | float:
    """
    Scale value to 0,1 range

    :param val: value to scale
    :param min: minimum value
    :param max: maximum value
    :return: scaled value
    """
    if val < min:
        return 0
    elif val > max:
        return 1
    else:
        return (val - min) / (max - min) if min < max or (max - min) != 0 else 0.0


def unscale(scaled_val: int | float, min: int | float, max: int | float) -> int | float:
    """
    Unscale value from 0,1 range to original range

    :param scaled_val: scaled value
    :param min: minimum value
    :param max: maximum value
    :return: unscaled value
    """
    if scaled_val < 0:
        return min
    elif scaled_val > 1:
        return max
    else:
        return scaled_val * (max - min) + min if min < max else min


def merge_observations(observations: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, List[Any]]]:
    """
    Merge observations from different agents or several observations from one agent

    :param observations: list of observations in form of a dict containing also dicts
    :return: dict of merged observations
    """
    merged_observations = dict(dict())
    for observation in observations:
        for stats_key, stats in observation.items():
            if stats_key not in merged_observations:
                merged_observations[stats_key] = {}
            if isinstance(stats, dict):
                for param_key, param_value in stats.items():
                    if param_key not in merged_observations[stats_key]:
                        merged_observations[stats_key][param_key] = []
                    merged_observations[stats_key][param_key].append(param_value)
            else:
                merged_observations[stats_key].append(stats)
    return merged_observations


def get_list_average(input_list: List[int | float]) -> int | float:
    """
    Get average value from list of values

    :param input_list: list of values
    :return: average value
    """
    return sum(input_list) / len(input_list) if len(input_list) > 0 else 0.0


def select_n_equidistant_elements_from_list(input_list: List[Any], n: int, cut_percent: int = 0) -> List[Any]:
    """
    Select n equidistant elements from the input list. E.g., n = 5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = [1, 3, 5, 8, 10]

    :param input_list: list of values
    :param n: number of elements to select
    :param cut_percent: percentage of elements to cut from the beginning (default is 0)
    :return: list of selected elements
    """
    if len(input_list) < n:
        raise ValueError(
            f"select_n_equidistant_elements_from_list: Input list length {len(input_list)} is less than n {n}"
        )
    elif n == 1:
        return [input_list[-1]]
    elif n == 2:
        return [input_list[0], input_list[-1]]
    else:
        cut_count = round(len(input_list) * cut_percent / 100)
        cut_count = min(cut_count, len(input_list) - n)
        interval = (len(input_list) - 1 - cut_count) / (n - 1)
        selected_indices = [0 + cut_count]

        for i in range(1, n - 1):
            index = round(i * interval) + cut_count
            selected_indices.append(index)

        selected_indices.append(len(input_list) - 1)
        return [input_list[i] for i in sorted(selected_indices)]
