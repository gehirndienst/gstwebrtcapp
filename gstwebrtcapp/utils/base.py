import asyncio
import logging
import numpy as np
import time
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

# logger
LOGGER = logging
LOGGER.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# exceptions
class GSTWEBRTCAPP_EXCEPTION(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self):
        return f"{self.args[0]}"


# TIME
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


def sleep_until_condition_with_intervals(
    num_intervals: int,
    sleeping_time_sec: float,
    condition_func: Callable[[], bool],
) -> bool:
    """
    Sleep until condition_func returns True or num_intervals is reached
    :param num_intervals: number of intervals
    :param sleeping_time_sec: sleeping time in seconds
    :param condition_func: callable that returns bool
    :return: True if condition_func returned True before num_intervals is reached, False otherwise
    """
    tick_interval_sec = sleeping_time_sec / num_intervals
    for _ in range(num_intervals):
        time.sleep(tick_interval_sec)
        if condition_func():
            return True
    return False


# SCALING
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


# LIST OPERATIONS
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


def get_list_average(input_list: List[int | float], is_skip_zeroes: bool = False) -> int | float:
    """
    Get average value from list of values

    :param input_list: list of values
    :param is_skip_zeroes: skip zeroes in the list
    :return: average value
    """
    if is_skip_zeroes:
        input_list = [value for value in input_list if value != 0.0]
    return sum(input_list) / len(input_list) if len(input_list) > 0 else 0


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


def slice_list_in_intervals(
    input_list: List[Any],
    num_intervals: int,
    intervals_type: str = 'equidistant',
) -> List[List[Any]]:
    """
    Get equidistant or sliding intervals from the input list. Regulated by intervals_type.
    E.g., [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] with num_intervals = 3

    equidistant -> [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]

    sliding -> [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    :param input_list: list of values
    :param num_intervals: number of intervals
    :param intervals_type: type of intervals. One of 'equidistant', 'sliding'
    :return: list of intervals
    """

    intervals_type = intervals_type.lower() if intervals_type in ['equidistant', 'sliding'] else 'equidistant'
    intervals = []
    interval_length = len(input_list) // num_intervals
    remainder = len(input_list) % num_intervals

    start_index = 0
    for i in range(num_intervals):
        end_index = start_index + interval_length + (1 if i < remainder else 0)
        if intervals_type == 'equidistant':
            interval_values = input_list[start_index:end_index]
        else:
            interval_values = input_list[:end_index]
        intervals.append(interval_values)
        start_index = end_index

    return intervals


def get_decay_weights(num_weights: int, start_weight: float = 0.4, ratio: float = 0.5) -> np.ndarray:
    """
    Get decay weights for the given number of weights, start weight and ratio. Sum of weights is 1.

    :param num_weights: number of weights
    :param start_weight: start weight
    :param ratio: ratio
    :return: decay weights
    """
    weights = start_weight * np.power(ratio, np.arange(num_weights))
    weights /= np.sum(weights)
    return weights


def cut_first_elements_in_list(
    input_list: List[Any],
    cut_percent: int = 0,
    min_remaining_elements: int = -1,
) -> List[Any]:
    """
    Cuts elements from the beginning of the list by the given cut percent.
    :param input_list: list of values
    :param cut_percent: percentage of elements to delete from the beginning
    :param min_remaining_elements: minimum number of elements to remain in the list. Default is -1 (no minimum)
    :return: list of values with deleted elements
    """
    num_elements_to_delete = int(len(input_list) * cut_percent / 100)
    if min_remaining_elements > 0:
        num_elements_to_delete = min(num_elements_to_delete, len(input_list) - min_remaining_elements)
    del input_list[:num_elements_to_delete]
    return input_list


def extract_network_traces_from_csv(csv_file: str, aggregation_interval: int = 1) -> Tuple[List[float], float]:
    """
    Extract aggregated bandwidth values from the csv file. The original interval is assumed to be 1 second.
    Csv file should contain two columns: bandwidth values and units.

    :param csv_file: csv file with bandwidth values
    :param aggregation_interval: aggregation interval
    :return: aggregated bandwidth values and out-of-coverage rate
    """
    df = pd.read_csv(csv_file, header=None, delimiter=',')
    bws = []
    curr_bandwidth, curr_count = 0, 0
    ooc_count = 0

    for _, row in df.iterrows():
        if row[1] == 'Kbits/sec':
            bandwidth = float(row[0]) / 1e3
        elif row[1] == 'bits/sec':
            bandwidth = float(row[0]) / 1e6
        else:
            bandwidth = float(row[0])

        if bandwidth < 1:
            ooc_count += 1

        curr_bandwidth += bandwidth
        curr_count += 1

        if curr_count == aggregation_interval:
            average_bandwidth = curr_bandwidth / aggregation_interval
            bws.append(average_bandwidth)
            curr_bandwidth, curr_count = 0, 0

    size = len(bws)
    ooc_rate = ooc_count / size / aggregation_interval if size > 0 else 0
    return bws, ooc_rate
