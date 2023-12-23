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


class APPBIN_EXCEPTION(GSTWEBRTCAPP_EXCEPTION):
    pass


# general utils
def wait_for_condition(condition_func: Callable[[], bool], timeout_sec: int) -> bool:
    start_time = time.time()
    while not condition_func():
        if time.time() - start_time >= float(timeout_sec):
            raise APPBIN_EXCEPTION(f"Timeout {timeout_sec} sec is reached for condition {condition_func.__name__}")
    return True


async def async_wait_for_condition(condition_func: Callable[[], bool], timeout_sec: int) -> bool:
    start_time = time.time()
    while not condition_func():
        if time.time() - start_time >= float(timeout_sec):
            raise APPBIN_EXCEPTION(f"Timeout {timeout_sec} sec is reached for condition {condition_func.__name__}")
        await asyncio.sleep(0.5)
    return True
