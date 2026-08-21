from pathlib import Path
from collections import deque
from datetime import datetime

import gymnasium as gym
import gymnasium_robotics
import numpy as np

# ============================================================
# IMPORTANT:
# Force Matplotlib to use a non-GUI backend.
# This avoids Tkinter/TkAgg threading errors such as:
#
# Tcl_AsyncDelete: async handler deleted by the wrong thread
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3 import HerReplayBuffer

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


# ============================================================
# REGISTER ROBOTICS ENVIRONMENTS
# ============================================================

gym.register_envs(gymnasium_robotics)


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "HandManipulateBlockRotateZ-v1"

# Available:
#
# "PPO"
# "A2C"
# "SAC"
# "TD3"
# "SAC_HER"
# "TD3_HER"
#
# Recommended first:
# SAC_HER
#
ALGORITHM = "PPO"


TOTAL_TIMESTEPS = 1_000_000


# ============================================================
# RUN ID
# ============================================================
#
# Every execution gets its own folder/files.
#
# Example:
#
# sac_her_20260821_132400
#
# So running SAC_HER multiple times does NOT overwrite
# previous experiments.
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_NAME = f"{ALGORITHM.lower()}_{RUN_ID}"


# ============================================================
# DIRECTORIES
# ============================================================

BASE_LOG_DIR = Path("logs")
BASE_MODEL_DIR = Path("models")
BASE_IMAGE_DIR = Path("images")


LOG_DIR = BASE_LOG_DIR / RUN_NAME
MODEL_DIR = BASE_MODEL_DIR / RUN_NAME
IMAGE_DIR = BASE_IMAGE_DIR / RUN_NAME


LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


PROGRESS_IMAGE = IMAGE_DIR / "training_progress.png"


# ============================================================
# TRAINING PROGRESS CALLBACK
# ============================================================

class TrainingProgressCallback(BaseCallback):

    def __init__(
        self,
        save_path,
        plot_every=25_000,
        reward_window=100,
        success_window=100,
        verbose=1,
    ):

        super().__init__(verbose)

        self.save_path = Path(save_path)

        self.plot_every = plot_every
        self.reward_window = reward_window
        self.success_window = success_window

        self.episode_timesteps = []
        self.episode_rewards = []

        self.reward_history = deque(
            maxlen=reward_window
        )

        self.success_history = deque(
            maxlen=success_window
        )

        self.mean_rewards = []
        self.mean_successes = []

        self.last_plot_step = 0


    # ========================================================
    # CALLED ON EVERY ENVIRONMENT STEP
    # ========================================================

    def _on_step(self) -> bool:

        infos = self.locals.get("infos", [])

        for info in infos:

            # Monitor adds this when an episode ends.
            if "episode" not in info:
                continue


            # ------------------------------------------------
            # Episode reward
            # ------------------------------------------------

            episode_reward = float(
                info["episode"]["r"]
            )

            self.episode_timesteps.append(
                self.num_timesteps
            )

            self.episode_rewards.append(
                episode_reward
            )

            self.reward_history.append(
                episode_reward
            )

            mean_reward = float(
                np.mean(self.reward_history)
            )

            self.mean_rewards.append(
                mean_reward
            )


            # ------------------------------------------------
            # Success information
            # ------------------------------------------------

            success = info.get(
                "is_success",
                info.get("success", None),
            )


            if success is not None:

                success = float(success)

                self.success_history.append(
                    success
                )

                mean_success = float(
                    np.mean(self.success_history)
                )

                self.mean_successes.append(
                    mean_success
                )

            else:

                self.mean_successes.append(
                    np.nan
                )


        # ----------------------------------------------------
        # Periodically regenerate progress PNG
        # ----------------------------------------------------

        if (
            self.num_timesteps - self.last_plot_step
            >= self.plot_every
        ):

            self.save_plot()

            self.last_plot_step = (
                self.num_timesteps
            )


        return True


    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    def _on_training_end(self):

        self.save_plot()


    # ========================================================
    # SAVE PROGRESS IMAGE
    # ========================================================

    def save_plot(self):

        if len(self.episode_rewards) == 0:
            return


        timesteps = np.asarray(
            self.episode_timesteps
        )

        rewards = np.asarray(
            self.episode_rewards
        )

        mean_rewards = np.asarray(
            self.mean_rewards
        )


        fig, ax1 = plt.subplots(
            figsize=(12, 7)
        )


        try:

            # ================================================
            # RAW REWARD
            # ================================================

            ax1.plot(
                timesteps,
                rewards,
                alpha=0.20,
                label="Episode reward",
            )


            # ================================================
            # MOVING AVERAGE REWARD
            # ================================================

            ax1.plot(
                timesteps,
                mean_rewards,
                linewidth=2,
                label=(
                    f"Mean reward "
                    f"({self.reward_window} episodes)"
                ),
            )


            ax1.set_xlabel(
                "Training timesteps"
            )

            ax1.set_ylabel(
                "Episode reward"
            )

            ax1.grid(
                True,
                alpha=0.3
            )


            # ================================================
            # SUCCESS RATE
            # ================================================

            success_values = np.asarray(
                self.mean_successes
            )


            if np.any(
                ~np.isnan(success_values)
            ):

                ax2 = ax1.twinx()


                ax2.plot(
                    timesteps,
                    success_values * 100,
                    linestyle="--",
                    linewidth=2,
                    label=(
                        f"Success rate "
                        f"({self.success_window} episodes)"
                    ),
                )


                ax2.set_ylabel(
                    "Success rate [%]"
                )

                ax2.set_ylim(
                    0,
                    100
                )


                # Merge legends from both axes.

                lines1, labels1 = (
                    ax1.get_legend_handles_labels()
                )

                lines2, labels2 = (
                    ax2.get_legend_handles_labels()
                )


                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc="best",
                )

            else:

                ax1.legend(
                    loc="best"
                )


            # ================================================
            # TITLE
            # ================================================

            ax1.set_title(
                f"{ALGORITHM} training progression\n"
                f"{ENV_ID}\n"
                f"Run: {RUN_ID}"
            )


            fig.tight_layout()


            # ================================================
            # SAVE PNG
            # ================================================

            self.save_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            fig.savefig(
                self.save_path,
                dpi=150,
                bbox_inches="tight",
            )


            if self.verbose:

                print(
                    "\n"
                    f"[Plot] Saved: "
                    f"{self.save_path}"
                    "\n"
                )


        finally:

            # Always close the figure.
            #
            # Important for long training runs because
            # otherwise matplotlib figures can accumulate
            # in memory.

            plt.close(fig)


# ============================================================
# ENVIRONMENT
# ============================================================

env = gym.make(
    ENV_ID
)


# Monitor logs episode reward / episode length to CSV.

env = Monitor(
    env,
    filename=str(
        LOG_DIR / "monitor.csv"
    ),
)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    algorithm,
    env,
):

    # Shared neural network configuration.

    policy_kwargs = dict(
        net_arch=[
            256,
            256,
            256,
        ]
    )


    # ========================================================
    # PPO
    # ========================================================

    if algorithm == "PPO":

        return PPO(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=3e-4,

            n_steps=4096,

            batch_size=256,

            n_epochs=10,

            gamma=0.99,

            gae_lambda=0.95,

            clip_range=0.2,

            ent_coef=0.001,

            policy_kwargs=policy_kwargs,

            verbose=1,

            tensorboard_log=str(
                LOG_DIR
            ),
        )


    # ========================================================
    # A2C
    # ========================================================

    elif algorithm == "A2C":

        return A2C(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=7e-4,

            n_steps=20,

            gamma=0.99,

            gae_lambda=0.95,

            ent_coef=0.001,

            policy_kwargs=policy_kwargs,

            verbose=1,

            tensorboard_log=str(
                LOG_DIR
            ),
        )


    # ========================================================
    # SAC
    # ========================================================

    elif algorithm == "SAC":

        return SAC(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=3e-4,

            buffer_size=1_000_000,

            learning_starts=10_000,

            batch_size=256,

            tau=0.005,

            gamma=0.98,

            train_freq=1,

            gradient_steps=1,

            policy_kwargs=policy_kwargs,

            verbose=1,

            tensorboard_log=str(
                LOG_DIR
            ),
        )


    # ========================================================
    # SAC + HER
    # ========================================================

    elif algorithm == "SAC_HER":

        return SAC(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=3e-4,

            buffer_size=1_000_000,

            learning_starts=10_000,

            batch_size=256,

            tau=0.005,

            gamma=0.98,

            train_freq=1,

            gradient_steps=1,

            replay_buffer_class=(
                HerReplayBuffer
            ),

            replay_buffer_kwargs=dict(

                n_sampled_goal=4,

                goal_selection_strategy=(
                    "future"
                ),
            ),

            policy_kwargs=policy_kwargs,

            verbose=1,

            tensorboard_log=str(
                LOG_DIR
            ),
        )


    # ========================================================
    # TD3
    # ========================================================

    elif algorithm == "TD3":

        n_actions = (
            env.action_space.shape[-1]
        )


        action_noise = NormalActionNoise(

            mean=np.zeros(
                n_actions
            ),

            sigma=0.1 * np.ones(
                n_actions
            ),
        )


        return TD3(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=1e-3,

            buffer_size=1_000_000,

            learning_starts=10_000,

            batch_size=256,

            gamma=0.98,

            tau=0.005,

            train_freq=1,

            gradient_steps=1,

            action_noise=action_noise,

            policy_kwargs=policy_kwargs,

            verbose=1,

            tensorboard_log=str(
                LOG_DIR
            ),
        )


    # ========================================================
    # TD3 + HER
    # ========================================================

    elif algorithm == "TD3_HER":

        n_actions = (
            env.action_space.shape[-1]
        )


        action_noise = NormalActionNoise(

            mean=np.zeros(
                n_actions
            ),

            sigma=0.1 * np.ones(
                n_actions
            ),
        )


        return TD3(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=1e-3,

            buffer_size=1_000_000,

            learning_starts=10_000,

            batch_size=256,

            gamma=0.98,

            tau=0.005,

            train_freq=1,

            gradient_steps=1,

            action_noise=action_noise,

            replay_buffer_class=(
                HerReplayBuffer
            ),

            replay_buffer_kwargs=dict(

                n_sampled_goal=4,

                goal_selection_strategy=(
                    "future"
                ),
            ),

            policy_kwargs=policy_kwargs,

            verbose=1,

            tensorboard_log=str(
                LOG_DIR
            ),
        )


    # ========================================================
    # UNKNOWN ALGORITHM
    # ========================================================

    else:

        raise ValueError(

            f"Unknown algorithm: "
            f"{algorithm}\n\n"

            "Available algorithms:\n"
            "PPO\n"
            "A2C\n"
            "SAC\n"
            "TD3\n"
            "SAC_HER\n"
            "TD3_HER"
        )


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)

print(
    f"Environment : {ENV_ID}"
)

print(
    f"Algorithm   : {ALGORITHM}"
)

print(
    f"Timesteps   : "
    f"{TOTAL_TIMESTEPS:,}"
)

print(
    f"Run ID      : {RUN_ID}"
)

print(
    f"Logs        : {LOG_DIR}"
)

print(
    f"Models      : {MODEL_DIR}"
)

print(
    f"Images      : {IMAGE_DIR}"
)

print("=" * 70)
print()


model = create_model(
    ALGORITHM,
    env,
)


# ============================================================
# PROGRESS PLOT CALLBACK
# ============================================================

progress_callback = (
    TrainingProgressCallback(

        save_path=PROGRESS_IMAGE,

        # Rewrite this run's PNG every 25k steps.
        #
        # Other runs have their own folder, therefore
        # they are not overwritten.

        plot_every=25_000,

        reward_window=100,

        success_window=100,

        verbose=1,
    )
)


# ============================================================
# CHECKPOINT CALLBACK
# ============================================================
#
# Saves the current model every 50,000 timesteps.
#
# If training crashes at e.g. 350k steps, you still have:
#
# *_300000_steps.zip
# *_350000_steps.zip
#
# depending on where the checkpoint occurred.
# ============================================================

checkpoint_callback = (
    CheckpointCallback(

        save_freq=50_000,

        save_path=str(
            MODEL_DIR
        ),

        name_prefix=(
            f"{ALGORITHM.lower()}_hand_rotate_z"
        ),

        save_replay_buffer=(
            ALGORITHM
            in [
                "SAC",
                "TD3",
                "SAC_HER",
                "TD3_HER",
            ]
        ),

        save_vecnormalize=False,

        verbose=2,
    )
)


# ============================================================
# COMBINE CALLBACKS
# ============================================================

callbacks = CallbackList(
    [
        progress_callback,
        checkpoint_callback,
    ]
)


# ============================================================
# TRAIN
# ============================================================

try:

    model.learn(

        total_timesteps=(
            TOTAL_TIMESTEPS
        ),

        callback=callbacks,

        progress_bar=True,

        tb_log_name=RUN_NAME,
    )


# ============================================================
# CTRL+C HANDLING
# ============================================================
#
# If you manually stop training with Ctrl+C,
# save the current model before quitting.
# ============================================================

except KeyboardInterrupt:

    print()
    print(
        "Training interrupted by user."
    )

    print(
        "Saving emergency checkpoint..."
    )


    emergency_path = (
        MODEL_DIR
        / "interrupted_model"
    )


    model.save(
        emergency_path
    )


    print(
        f"Saved: "
        f"{emergency_path}.zip"
    )


    # Save latest graph too.

    progress_callback.save_plot()


    raise


# ============================================================
# FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / f"{ALGORITHM.lower()}_hand_rotate_z_final"
)


model.save(
    final_model_path
)


# ============================================================
# FINAL PLOT
# ============================================================

progress_callback.save_plot()


# ============================================================
# CLOSE ENVIRONMENT
# ============================================================

env.close()


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("Training finished successfully.")
print()

print(
    f"Final model:"
    f"\n{final_model_path}.zip"
)

print()

print(
    f"Training plot:"
    f"\n{PROGRESS_IMAGE}"
)

print()

print(
    f"Checkpoints:"
    f"\n{MODEL_DIR}"
)

print()

print(
    f"TensorBoard logs:"
    f"\n{LOG_DIR}"
)

print("=" * 70)