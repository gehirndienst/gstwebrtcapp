"""
controller.py

Description:
    Controller class is responsible for applying actions and collecting observations (webrtc metrics).

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

import asyncio
from typing import Any, Dict, Optional

from ahoyapp.app import GstWebRTCBinApp
from utils.base import LOGGER


class Controller:
    '''
    Controller class is responsible for applying actions and collecting observations (webrtc metrics)
    '''

    def __init__(self):
        self.action_queue = asyncio.Queue()
        self.observation_queue = asyncio.Queue()

        self.is_started = False

    async def handle_actions(self, app: GstWebRTCBinApp) -> None:
        while True:
            action_msg: Dict[str, Any] = await self.action_queue.get()
            if app is not None and len(action_msg) > 0:
                for action in action_msg:
                    if action_msg.get(action) is None:
                        LOGGER.error(f"ERROR: Action {action} has no value!")
                        continue
                    else:
                        match action:
                            case "bitrate":
                                app.set_bitrate(action_msg[action])
                            case "resolution":
                                app.set_resolution(action_msg[action])
                            case "framerate":
                                app.set_framerate(action_msg[action])
                            case _:
                                LOGGER.error(f"ERROR: Unknown action in the message: {action_msg}")

    def push_action(self, action_message: Dict[str, Any]) -> None:
        self.action_queue.put_nowait(action_message)

    def push_observation(self, obs: Dict[str, Any]) -> None:
        self.observation_queue.put_nowait(obs)

    def get_observation(self) -> Optional[Dict[str, Any]]:
        return self.observation_queue.get_nowait() if not self.observation_queue.empty() else None

    def clean_observation_queue(self) -> None:
        while not self.observation_queue.empty():
            _ = self.observation_queue.get_nowait()
