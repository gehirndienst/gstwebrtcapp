import asyncio
import logging
import time
from typing import Callable

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
