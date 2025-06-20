import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import imageio
import time

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    BinaryRewardClassifierWrapper,
)


if __name__ == "__main__":
    env = gym.make("FrankaEnv-Vision", save_video=True)
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]
    print(image_keys)

    obs, _ = env.reset()

    all_transitions = []
    success_count = 0
    success_needed = 70
    total_count = 0
    timesteps = 0

    pbar = tqdm(total=success_needed)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./gene_demos/task_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            transitions = pkl.load(f)
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    transitions = []
    while success_count < success_needed:
        actions = np.zeros((6,))
        # getting actions from saved replay to sanity check them
        # actions = task1transitions[i]["action"]
        # i += 1

        next_obs, rew, done, truncated, info = env.step(action=actions)
        # print(next_obs)
        # breakpoint()
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # imageio.imwrite(f"{timesteps}.png", next_obs["images"]["kinect"])
        transition = copy.deepcopy(
            dict(
                observations=copy.deepcopy(obs),
                actions=actions,
                next_observations=copy.deepcopy(next_obs),
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        if not np.all(actions == 0):
            timesteps += 1
            transitions.append(transition)
        else:
            continue
            # for debugging, now works as intended
            # print("caught zero action!")

        obs = next_obs

        if done:
            # Wait for user to press spacemouse button: right=save, left=discard
            print("Episode finished. Press RIGHT button to save, LEFT button to discard.")
            save_episode = None
            while save_episode is None:
                # Poll the environment for button press
                dummy_action = np.zeros_like(actions)
                _, _, _, _, info = env.step(dummy_action)
                if info.get("right", False):
                    save_episode = True
                elif info.get("left", False):
                    save_episode = False
                else:
                    time.sleep(0.05)  # avoid busy waiting
            if save_episode:
                all_transitions.append(transitions)
                success_count += rew
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(rew)

                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                with open(file_path, "wb") as f:
                    pkl.dump(all_transitions, f)
                    print(
                        f"saved {len(all_transitions)} demos and {timesteps} timesteps to {file_path}"
                    )
            else:
                print("Episode discarded.")

            transitions = []
            print("Reset the scene and press any spacemouse button to continue.")
            # Wait for any button to be pressed to continue
            waiting = True
            while waiting:
                dummy_action = np.zeros_like(actions)
                _, _, _, _, info = env.step(dummy_action)
                if info.get("left", False) or info.get("right", False):
                    waiting = False
                else:
                    time.sleep(0.05)
            obs, _ = env.reset()

    env.close()
    pbar.close()
