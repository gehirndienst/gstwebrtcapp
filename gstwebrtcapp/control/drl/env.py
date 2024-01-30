import collections
from gymnasium.core import Env
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import time
from typing import Any, Dict, List, Optional, OrderedDict

from control.controller import Controller
from control.drl.mdp import MDP
from utils.base import LOGGER


class DrlEnv(Env):
    def __init__(
        self,
        controller: Controller,
        mdp: MDP,
        max_episodes: int = -1,
        state_update_interval: float = 1.0,
    ):
        self.controller = controller
        self.mdp = mdp
        self.max_episodes = max_episodes
        self.state_update_interval = state_update_interval

        self.episodes = 1
        self.steps = 0
        self.last_action = None
        self.state = collections.OrderedDict()
        self.reward = 0.0
        self.reward_parts = {}
        self.is_finished = False

        self.observation_space = self.mdp.create_observation_space()
        self.action_space = self.mdp.create_action_space()

    def step(self, action):
        self.steps += 1
        self.last_action = action
        self.controller.push_action(self.mdp.pack_action_for_controller(action))

        # get observation (webrtc stats) from the controller
        stats = self._get_observation()
        if stats is None:
            return self.state, self.reward, True, True, {}

        # make state from the observation
        state_dict = self.mdp.make_state(stats)
        self.state = self._dict_to_gym_space_sample(state_dict)

        self.reward, self.reward_parts = self.mdp.calculate_reward()

        terminated = self.mdp.is_terminated(self.steps)
        truncated = self.mdp.is_truncated(self.steps)
        if terminated or truncated:
            self.episodes += 1

        return self.state, self.reward, terminated, truncated, {}

    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)
        self.steps = 0

        self.mdp.reset()

        if self.max_episodes > 0 and self.episodes > self.max_episodes:
            self.is_finished = True
        elif options.get("reset_after_break") is not None:
            self.is_finished = False

        self.state = self._get_initial_state()

        # to save the reward fields in a dict
        self.reward_parts = self.mdp.get_default_reward_parts_dict()

        LOGGER.info(f"INFO: resetting the DRL env: episodes {self.episodes}, is finished {self.is_finished}")
        return self.state, {}

    def _get_observation(self) -> Optional[Dict[str, Any]]:
        start_fetch_time = time.time()
        while (
            self.controller.observation_queue.empty()
            and time.time() - start_fetch_time < self.state_update_interval * 2
        ):
            time.sleep(0.1)
            if self.is_finished:
                # this could be triggered e.g., by agent.stop() call or by DrlBreakCallback
                LOGGER.info("WARNING: Interrupted by a finish signal, closing the env...")
                self.state = self._get_initial_state()
                self.reward = 0.0
                return None

        is_obs = False
        time_inactivity_starts = time.time()
        while not is_obs:
            stats = self.controller.get_observation()
            if stats is None:
                # this could be triggered if you pulled all queue elements
                # but none of them passed the check after max timeout defined in controller
                if time.time() - time_inactivity_starts > self.controller.max_inactivity_time:
                    LOGGER.warning(
                        "WARNING: No stats were pulled from the observation queue after"
                        f" {self.controller.max_inactivity_time} sec, closing the env..."
                    )
                    self.is_finished = True
                    self.state = self._get_initial_state()
                    self.reward = 0.0
                    return None
                else:
                    LOGGER.warning("WARNING: No stats were pulled from the observation queue, waiting...")
                    continue
            else:
                is_obs = self.mdp.check_observation(stats)
        # do this to avoid stuck obs in the queue to take only the most recent one
        self.controller.clean_observation_queue()
        return stats

    def _dict_to_gym_space_sample(self, state_dict: dict) -> OrderedDict[str, Any]:
        tuples = []
        for key, space in self.observation_space.items():
            if isinstance(space, Box):
                if isinstance(state_dict[key], List) or isinstance(state_dict[key], np.ndarray):
                    value = np.array(state_dict[key], dtype=space.dtype).reshape(space.shape)
                else:
                    value = np.array([state_dict[key]], dtype=space.dtype).reshape(space.shape)
            elif isinstance(space, MultiDiscrete):
                if isinstance(state_dict[key], List) or isinstance(state_dict[key], np.ndarray):
                    value = np.array(state_dict[key], dtype=space.dtype).reshape(space.nvec.shape)
                else:
                    value = np.array([state_dict[key]], dtype=space.dtype).reshape(space.nvec.shape)
            else:
                value = state_dict[key]
            tuples.append((key, value))
        return collections.OrderedDict(tuples)

    def _get_initial_state(self) -> OrderedDict[str, Any]:
        return self._dict_to_gym_space_sample(self.mdp.make_default_state())

    def get_step_info(self) -> Dict[str, Any]:
        # get all env variables after step execution in a pretty-printing way. Called externally by the callback.
        general_dict = {
            "step": self.steps,
            "episode": self.episodes,
            "action": (
                self.last_action[0]
                if self.last_action is not None
                else (
                    None
                    if not self.mdp.is_scaled
                    else (
                        self.mdp.convert_to_unscaled_action(self.last_action)[0]
                        if self.last_action is not None
                        else None
                    )
                )
            ),
        }
        state_unscaled = self.state if not self.mdp.is_scaled else self.mdp.convert_to_unscaled_state(self.state)
        state_dict = {
            f"state/{k}": (
                v[0] if (isinstance(v, List) and len(v) == 1) or (isinstance(v, np.ndarray) and v.size == 1) else v
            )
            for k, v in state_unscaled.items()
        }
        rewards_dict = {f"reward/{k}": v for k, v in self.reward_parts.items()}
        return general_dict | state_dict | rewards_dict
