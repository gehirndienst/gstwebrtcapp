import csv
import datetime
import glob
import gymnasium
import os

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from utils.base import LOGGER


class DrlCheckpointCallback(CheckpointCallback):
    """
    Save model with the given frequency and always at the end of training

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level (0 -- 2)
    """

    def __init__(
        self,
        save_freq: int = 1000,
        save_path: str = "./models",
        name_prefix: str = "drl_model",
        save_replay_buffer: bool = True,
        save_vecnormalize: bool = True,
        verbose: int = 0,
    ):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _on_training_end(self):
        include = [
            "policy",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage_logger",
            "_custom_logger",
        ]

        dt = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        model_path_full = os.path.join(self.save_path, f"{self.name_prefix}_{dt}_FULL.zip")
        model_path_default = os.path.join(self.save_path, f"{self.name_prefix}_{dt}_DEFAULT.zip")

        LOGGER.info("OK: Saving final models...")
        self.model.save(model_path_full, exclude=["env"], include=include)
        self.model.save(model_path_default)

        # 1. save the last model additionally as last_full.zip and replace always with the newest trained version
        last_full_path = os.path.join(self.save_path, "last_full.zip")
        for filename in glob.glob(last_full_path):
            os.remove(filename)
        self.model.save(last_full_path, exclude=["env"], include=include)

        # 2. save also the last default model as last_default.zip wo replay and experience buffers as default saving by model.save()
        last_def_path = os.path.join(self.save_path, "last_default.zip")
        for filename in glob.glob(last_def_path):
            os.remove(filename)
        self.model.save(last_def_path)

        LOGGER.info("OK: All models are successfully saved on training end, training is finished!")


class DrlPrintStepCallback(BaseCallback):
    """
    Prints each transition with all state vars and reward parts.
    For the clear and concise output, it works only with a single environment or with a 1-dim vectorized one.

    :param verbose: Verbosity level (0 -- 2)
    """

    def __init__(self, verbose: int = 0, eval_env: VecEnv | gymnasium.Env | None = None):
        super().__init__(verbose)
        self.start_time = datetime.datetime.now()
        self.eval_env = eval_env

    def _init_callback(self):
        self.env = self.eval_env if self.eval_env is not None else self.training_env

    def _on_step(self):
        if isinstance(self.env, VecEnv):
            episodes = self.env.get_attr("episodes")[0]
            steps = self.env.get_attr("steps")[0]
            last_action = self.env.get_attr("last_action")[0]
            state = self.env.get_attr("state")[0]
            rewards = self.env.get_attr("reward_parts")[0]
            is_finished = self.env.get_attr("is_finished")[0]
        else:
            episodes = self.env.episodes
            steps = self.env.steps
            last_action = self.env.last_action
            state = self.env.state
            rewards = self.env.reward_parts
            is_finished = self.env.is_finished

        state = {k: v.tolist() for k, v in state.items()}
        time_elapsed = datetime.datetime.now() - self.start_time

        if steps > 0 and not is_finished:
            if self.verbose > 1:
                # print on every step
                LOGGER.info(
                    "INFO: Training step info: \n "
                    + f"Time elapsed (hh:mm:ss.ms) {time_elapsed}"
                    + "\n"
                    + f"Episodes: {episodes}"
                    + "\n"
                    + f"Step: {steps},"
                    + "\n"
                    + f"Last action: {last_action},"
                    + "\n"
                    + f"State: {state},"
                    + "\n"
                    + f"Reward: {rewards}"
                    + "\n"
                )
            else:
                # print short version on every 100th step (and 1th as well)
                if steps == 1 or (steps >= 100 and steps % 100 == 0):
                    LOGGER.info(
                        "INFO: Training step info: \n "
                        + f"Time elapsed (hh:mm:ss.ms) {time_elapsed}"
                        + "\n"
                        + f"Episodes: {episodes}"
                        + "\n"
                        + f"Step: {steps},"
                        + "\n"
                    )
        return True


class DrlSaveStepCallback(BaseCallback):
    """
    Saves env step info to a csv file.
    For the clear and concise output, it works only with a single environment or with a 1-dim vectorized one.

    :param save_path: Path for the folder where csv files are saved.
    :param model_name: Current model name.
    :param verbose: Verbosity level (0 -- 2)
    """

    def __init__(
        self,
        save_path: str = "./logs",
        model_name: str = "",
        verbose: int = 0,
        eval_env: VecEnv | gymnasium.Env | None = None,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.model_name = model_name
        self.eval_env = eval_env
        self.time = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    def _init_callback(self):
        self.env = self.eval_env if self.eval_env is not None else self.training_env
        os.makedirs(self.save_path, exist_ok=True)
        self.file_handler = open(self._get_csv_filename(), mode="a", newline="\n")
        self.csv_writer = csv.DictWriter(self.file_handler, fieldnames=self._get_step_info().keys())
        if os.stat(self._get_csv_filename()).st_size == 0:
            self.csv_writer.writeheader()
        self.file_handler.flush()

    def _on_step(self):
        steps = self.env.get_attr("steps")[0] if isinstance(self.env, VecEnv) else self.env.steps
        if steps > 0:
            # might be a bug writing zero dummy step because of vecenv wrapper
            self.csv_writer.writerow(self._get_step_info())
        return True

    def _on_training_end(self):
        self.file_handler.close()

    def _get_csv_filename(self):
        return os.path.join(self.save_path, f"drl_training_{self.model_name}_{self.time}.csv")

    def _get_step_info(self):
        if isinstance(self.env, VecEnv):
            return self.env.env_method("get_step_info")[0]
        else:
            return self.env.get_step_info()


class DrlBreakCallback(BaseCallback):
    """
    Provides a congruity between the model and environment for breaking cases:
        a) if env reached its max episode (controlled with 'max_episodes' param), finishes the training and closes the env
        b) closes the env if the training was externally interrupted

    :param verbose: Verbosity level (0 -- 2)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self):
        if all(self._is_env_finished()):
            LOGGER.info("OK: DrlBreakCallback -- the env is finished, breaking the training...")
            return False
        return True

    def _on_training_end(self):
        finished_envs = self._is_env_finished()
        if not all(finished_envs):
            LOGGER.info("OK: DrlBreakCallback -- training was interrupted, trigger env to stop...")
            unfinished_env_idxs = [i for i, x in enumerate(finished_envs) if not x]
            self._trigger_stop(unfinished_env_idxs)

    def _is_env_finished(self):
        if isinstance(self.training_env, VecEnv):
            return self.training_env.get_attr("is_finished")
        else:
            return self.training_env.is_finished

    def _trigger_stop(self, unfinished_indices):
        if isinstance(self.training_env, VecEnv):
            for env_idx in unfinished_indices:
                self.training_env.env_method("close")[env_idx]
        else:
            self.training_env.close()


class DrlEvaluatingCallback:
    """
    auxilary class for self-producing a default callback for an old way of calling in evaluate_policy() function.
    All other evaluating callbacks must inherit this class.
    """

    def __init__(self, env, save_path: str = "./logs", model_name: str = "", verbose: int = 0):
        self.step_callback = DrlSaveStepCallback(save_path, model_name, verbose, env)
        self.step_printing_callback = DrlPrintStepCallback(verbose, env)
        self.is_env_reset = False
        self.verbose = verbose

    def on_step(self):
        if not self.is_env_reset:
            if self.verbose >= 1:
                self.step_printing_callback._init_callback()
            if self.verbose == 2:
                self.step_callback._init_callback()
            self.is_env_reset = True
        if self.verbose >= 1:
            self.step_printing_callback._on_step()
        if self.verbose == 2:
            self.step_callback._on_step()
