import csv
import os
import time
import numpy as np
import torch
from typing import Dict, List, Union

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from control.drl.config import DrlConfig
from control.drl.callbacks import (
    DrlCheckpointCallback,
    DrlEvaluatingCallback,
    DrlPrintStepCallback,
    DrlSaveStepCallback,
    DrlBreakCallback,
)
from control.drl.env import DrlEnv
from control.drl.mconfigurator import DrlModelConfigurator
from control.drl.mdp import MDP
from message.client import MqttPair
from utils.base import LOGGER


class DrlManager:
    """
    A manager for preprocessing, running and overall controlling of DRL training/evaluation process, namely it:
        1) instantiates a Gymnasium environment,\n
        2) configures SB3 DRL model according to the given hyperparameters and configuration settings,\n
        3) attaches loggers and callbacks and controls the output directories,\n
        4) performs a training or evaluation process for the DRL model using SB3 backend.\n

    :param config: DRL model params, look into ``control/drl/config.py``.
    :param mdp: The MDP instance, look into ``control/drl/mdp.py``.
    :param mqtts: MQTT instances`.
    """

    def __init__(self, config: DrlConfig, mdp: MDP, mqtts: MqttPair):
        self.config = config
        self.mdp = mdp
        self.mqtts = mqtts

        self._setup()

    def _setup(self) -> None:
        # set mqqts for the mdp
        self.mdp.mqtts = self.mqtts

        # set paths
        self.log_path, self.model_path = self._set_save_paths(self.config.save_log_path, self.config.save_model_path)

        # set callbacks
        self.callbacks = self._set_callbacks(self.config.callbacks)

        # set episodes and total (max) timesteps for the process
        self.episodes = self.config.episodes
        if self.episodes < 0 and self.config.mode == "train":
            raise Exception(f"ERROR: DrlManager: episodes must be > 0 for training mode!")
        elif self.episodes == 0:
            LOGGER.error(f"ERROR: DrlManager: episodes can't be 0, setting to 1 (default)")
            self.episodes = 1
        self.episode_length = self.config.episode_length
        self.total_timesteps = self.episodes * self.episode_length
        LOGGER.info(
            f"OK: Episodes: {self.episodes}, episode length: {self.episode_length}, timesteps: {self.total_timesteps}"
        )

        # check for CUDA and set up the device
        if self.config.device is not None and (
            self.config.device.startswith("cpu") or self.config.device.startswith("cuda")
        ):
            self.device = self.config.device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
                curr_dev = torch.cuda.current_device()
                gpu = torch.cuda.get_device_properties(curr_dev)
                LOGGER.info(
                    f"INFO: Cuda is ON: found {torch.cuda.device_count()} GPUs available. Using the following GPU"
                    f" {curr_dev} {gpu.name} with {gpu.total_memory / 1e9}Gb of total memory"
                )
            else:
                self.device = "cpu"
                LOGGER.info("INFO: Cuda is OFF: using cpu only\n")

        # initialize env
        self.env = DrlEnv(
            mdp=self.mdp,
            mqtts=self.mqtts,
            max_episodes=self.episodes,
            state_update_interval=self.config.state_update_interval,
            max_inactivity_time=self.config.state_max_inactivity_time,
        )

        # initialize SB3 DRL model
        if isinstance(self.config.hyperparams_cfg, Dict):
            self.model_cfg = DrlModelConfigurator(
                model_name=self.config.model_name,
                tensorboard_log=self.log_path if self.config.verbose == 2 else None,
                device=self.device,
                verbose=self.config.verbose,
                **self.config.hyperparams_cfg,
            )
        else:
            self.model_cfg = DrlModelConfigurator(
                model_name=self.config.model_name,
                model_hyperparams_file=self.config.hyperparams_cfg,
                n_steps=self.episode_length,
                tensorboard_log=self.log_path if self.config.verbose == 2 else None,
                device=self.device,
                verbose=self.config.verbose,
            )

        # a path to a saved model file with extension if exists
        try:
            self.model_file = (
                os.path.splitext(self.config.model_file)[0] if self.config.model_file is not None else None
            )
        except OSError:
            raise Exception(f"There is no valid DRL model on this path : {self.model_file}")

        # make model
        if self.model_file is None:
            # make a fresh one
            self.model = self.model_cfg.make_model(self.env)
            self.is_reset_timesteps = True
            LOGGER.info(
                f"OK: Successfully created a new {self.config.model_name} model with the given hyperparameters!\n"
            )
        else:
            # load the model from the given file
            self.model = self.model_cfg.get_model_class().load(self.model_file, env=self.env, device=self.device)
            self.is_reset_timesteps = False
            LOGGER.info(
                f"OK: Successfully loaded {self.config.model_name} model from the given file {self.model_file}!\n"
            )

        # set logs
        self.model.verbose = self.config.verbose
        loggers = []
        if self.config.verbose >= 1:
            loggers.append("stdout")
            if self.config.verbose == 2:
                os.makedirs(self.log_path, exist_ok=True)
                if self.config.verbose == 2 and self.config.mode == 'train':
                    loggers += ["log", "csv", "tensorboard"]
        self.logger = configure(self.log_path if self.config.verbose == 2 else None, loggers)
        self.model.set_logger(self.logger)

        # set deterministic flag for evaluation
        self.deterministic = self.config.deterministic if self.config.mode == 'eval' else True

        # finish all setup steps...

    def reset(self, is_load_last_model: bool = False) -> None:
        """reset the manager after breaking the training"""

        self.env.reset(options={"reset_after_break": True})

        if (
            is_load_last_model
            and self.model.num_timesteps > 0
            and self.model_file is None
            and self.config.mode == "train"
        ):
            # we trained model from scratch and want to continue training after saving it on DrlBreakCallback event
            if not any(isinstance(cb, DrlCheckpointCallback) for cb in self.callbacks):
                LOGGER.warning("WARNING: No DrlCheckpointCallback is set, cannot load the last trained model!")
                return
            LOGGER.info(f"OK: Loading the last trained model on the DRL manager reset...")
            # NOTE: this is a fixed filename. "last_full" contains replay buffer and optimizer state, "last_default" does not
            model_file = os.path.join(self.model_path, "last_full")
            self.model = self.model_cfg.get_model_class().load(model_file, env=self.env, device=self.device)
            self.is_reset_timesteps = False
            LOGGER.info(f"OK: Successfully loaded {self.config.model_name} model from the given file {model_file}!")

    def stop(self) -> None:
        """stop the manager and the env"""

        self.env.is_finished = True

    def train(self) -> None:
        """train the model"""

        LOGGER.info(f"OK: Training {self.config.model_name} model for {self.total_timesteps} steps...\n")
        self.model = self.model.learn(
            total_timesteps=self.total_timesteps,
            reset_num_timesteps=self.is_reset_timesteps,
            callback=self.callbacks,
        )

    def eval(self, eval_callback: DrlEvaluatingCallback | str | None = "default") -> None:
        """evaluate the model"""

        if eval_callback is not None:
            if isinstance(eval_callback, str) and eval_callback == "default":
                callback_step = DrlEvaluatingCallback(
                    self.env,
                    save_path=self.log_path,
                    model_name=self.config.model_name + "_eval" + ("_det" if self.deterministic else "_ndet"),
                    verbose=self.config.verbose,
                )
            elif isinstance(eval_callback, DrlEvaluatingCallback):
                callback_step = eval_callback
            else:
                raise Exception(f"Unknown eval callback {eval_callback}")
        else:
            callback_step = None

        callback = lambda locals, globals: callback_step.on_step() if callback_step is not None else None

        LOGGER.info(
            f"OK: Evaluating {self.config.model_name} model with det={self.deterministic} for"
            f" {self.total_timesteps} steps...\n"
        )

        # based on stable_baselines3.common.evaluation.evaluate_policy
        if not isinstance(self.env, VecEnv):
            # num_envs is always 1
            self.env = DummyVecEnv([lambda: self.env])  # type: ignore[list-item, return-value]
            assert self.env.num_envs == 1, "Assertion FAILED: DrlManager.eval: only single env is allowed"

        is_infinite_episodes = False
        episodes = self.episodes
        if episodes < 0:
            # infinite episodes but the env will be resetted after each episode_length steps or on terminal state
            episodes = 1
            is_infinite_episodes = True
        episodes_passed = 0
        episode_rewards = []
        episode_lengths = []
        current_reward = 0.0
        current_length = 0

        observations = self.env.reset()
        hidden_states = None
        episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        while episodes_passed < episodes:
            actions, hidden_states = self.model.predict(
                observations,  # type: ignore[arg-type]
                state=hidden_states,
                episode_start=episode_starts,
                deterministic=self.deterministic,
            )

            new_observations, rewards, dones, _ = self.env.step(actions)

            reward = rewards[0]
            done = dones[0]
            episode_starts[0] = done
            current_reward += reward
            current_length += 1

            if callback is not None:
                callback(locals(), globals())

            if done:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                if not is_infinite_episodes:
                    episodes_passed += 1
                current_reward = 0.0
                current_length = 0

            observations = new_observations

        LOGGER.info(
            f"OK: Evaluation is finished for {len(episode_rewards)} episodes! min_reward: "
            f"{min(episode_rewards)}, max_reward: {max(episode_rewards)}, "
            f"mean_reward: {np.mean(episode_rewards)}, std_reward: {np.std(episode_rewards)}"
        )

        if self.config.verbose == 2:
            eval_path = os.path.join(
                self.log_path, f'{self.config.model_name}_eval_output_{time.strftime("%Y%m%d-%H%M%S")}.csv'
            )
            with open(eval_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode_rewards", "episode_lengths"])
                writer.writeheader()
                writer.writerows(
                    [{"episode_rewards": i, "episode_lengths": j} for i, j in zip(episode_rewards, episode_lengths)]
                )

    def _set_save_paths(self, save_log_path: str, save_model_path: str) -> None:
        assert save_log_path is not None and save_model_path is not None, "ERROR: save paths are not set!"
        timestamp = time.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        log_path = os.path.join(save_log_path, f"log_{timestamp}")
        model_path = os.path.join(save_model_path, f"model_{timestamp}")
        return log_path, model_path

    def _set_callbacks(self, callbacks: Union[None, List[BaseCallback], List[str]]) -> List[BaseCallback]:
        cbs: List[BaseCallback] = []
        if callbacks is not None:
            if isinstance(callbacks, list) and all(isinstance(cb, str) for cb in callbacks):
                for callback_name in callbacks:
                    match callback_name:
                        case "save_model":
                            os.makedirs(self.model_path, exist_ok=True)
                            cbs.append(DrlCheckpointCallback(save_path=self.model_path, verbose=self.config.verbose))
                        case "save_step":
                            os.makedirs(self.log_path, exist_ok=True)
                            cbs.append(
                                DrlSaveStepCallback(
                                    save_path=self.log_path,
                                    model_name=self.config.model_name,
                                    verbose=self.config.verbose,
                                )
                            )
                        case "print_step":
                            cbs.append(DrlPrintStepCallback(verbose=self.config.verbose))
                        case _:
                            raise Exception(f"GymirDrlManager: unknown callback {callback_name}")
            else:
                cbs = callbacks

        # always append this callback (regulates finishing)
        cbs.append(DrlBreakCallback(verbose=self.config.verbose))

        return cbs
