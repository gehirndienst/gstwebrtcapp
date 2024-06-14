import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from typing import Any, Callable

from utils.base import LOGGER


async def restart_wrapper(
    coro: Callable[..., asyncio.Future],
    *args: Any,
    condition_cb: Callable[..., bool] = lambda: True,
    restart_cb: Callable[..., asyncio.Future] = lambda: asyncio.sleep(1),
    is_raise_exception: bool = False,
) -> None:
    """
    An async wrapper that restarts a coroutine if it fails or got cancelled. It does not handle the runtime backend errors if given.
    :param coro: Coroutine
    :param args: Arguments to pass to the coroutine
    :param condition_cb: A callback that controls the execution of the coroutine's loop. By default, it is always True
    :param restart_cb: A callback that is awaited when the coroutine is cancelled. By default, it is just a 1 second sleep
    :param is_raise_exception: If True, the coroutine will raise an exception and stop if it fails
    """
    while condition_cb():
        try:
            await coro(*args)
        except asyncio.CancelledError:
            await restart_cb()
            continue
        except Exception as e:
            LOGGER.error(f"ERROR: Coroutine {coro.__name__} has encountered an exception:\n {e}")
            if is_raise_exception:
                raise e
            await asyncio.sleep(1)  # FIXME: ignore the restart_cb if exception is raised?
            continue


async def executor_wrapper(
    coro: Callable[..., asyncio.Future],
    *args: Any,
    condition_cb: Callable[..., bool] = lambda: True,
    restart_cb: Callable[..., asyncio.Future] = lambda: asyncio.sleep(1),
    is_raise_exception: bool = False,
    loop: asyncio.AbstractEventLoop = None,
) -> None:
    """
    An async wrapper that runs a coroutine in another thread managed by an executor and restarts it if it fails or gets cancelled.
    :param coro: Coroutine
    :param args: Arguments to pass to the coroutine
    :param condition_cb: A callback that controls the execution of the coroutine's loop. By default, it is always True
    :param restart_cb: A callback that is awaited when the coroutine is cancelled. By default, it is just a 1 second sleep
    :param is_raise_exception: If True, the coroutine will raise an exception and stop if it fails
    :param loop: The asyncio event loop to run the coroutine. If None, the current loop is used
    """
    if loop is None:
        loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    def _run_coroutine_threadsafe(loop, coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    try:
        while condition_cb():
            try:
                await loop.run_in_executor(executor, _run_coroutine_threadsafe, loop, coro(*args))
            except asyncio.CancelledError:
                await restart_cb()
                continue
            except Exception as e:
                LOGGER.error(f"ERROR: Coroutine {coro.__name__} has encountered an exception:\n {e}")
                if is_raise_exception:
                    raise e
                await asyncio.sleep(1)
                continue
    finally:
        executor.shutdown(wait=False)


try:
    import uvloop
except ImportError:
    uvloop = None


def threaded_wrapper(
    coro: Callable[..., asyncio.Future],
    *args: Any,
    condition_cb: Callable[..., bool] = lambda: True,
    restart_cb: Callable[..., asyncio.Future] = lambda: asyncio.sleep(1),
    is_raise_exception: bool = False,
) -> asyncio.Future:
    """
    A sync wrapper that runs a coroutine in another thread and restarts it if it fails or gets cancelled.
    :param coro: Coroutine to monitor
    :param args: Arguments to pass to the coroutine
    :param condition_cb: A callback that controls the execution of the coroutine's loop. By default, it is always True
    :param restart_cb: A callback that is awaited when the coroutine is cancelled. By default, it is just a 1 second sleep
    :param is_raise_exception: If True, the coroutine will raise an exception and stop if it fails
    :return: A Future that completes when the threaded task is done
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def thread_func():
        if uvloop:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        while condition_cb():
            try:
                thread_loop.run_until_complete(coro(*args))
            except asyncio.CancelledError:
                thread_loop.run_until_complete(restart_cb())
                continue
            except Exception as e:
                LOGGER.error(f"ERROR: Coroutine {coro.__name__} has encountered an exception:\n {e}")
                if is_raise_exception:
                    future.set_exception(e)
                    return
                thread_loop.run_until_complete(asyncio.sleep(1))
                continue
        thread_loop.close()
        future.set_result(None)

    thread = threading.Thread(target=thread_func)
    thread.start()

    def cleanup_thread():
        if thread.is_alive():
            thread.join()

    future.add_done_callback(lambda _: cleanup_thread())
    return future
