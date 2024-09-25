from Simulations import *
from Simulations.PPO_Policy import train, continue_train, test
from panda3d.core import loadPrcFile
import time
from os import path
from typing import Optional
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}
loadPrcFile("myConfig.prc")

class CloseDoor(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type: str = "dense", **kwargs):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "closeDoor_door.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


        self._model_names = MujocoModelNames(self.model)

        # whether to have sparse rewards
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
                self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.door_hinge_addrs = self.model.jnt_dofadr[
            self._model_names.joint_name2id["door_hinge"]
        ]
        self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.handle_site_id = self._model_names.site_name2id["S_handle"]
        self.door_body_id = self._model_names.body_name2id["frame"]

        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "door_body_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        # compute the sparse reward variant first
        goal_distance = self.data.qpos[self.door_hinge_addrs]
        goal_achieved = True if goal_distance >= 1.29 else False

        if goal_achieved:
            reward = 10.0

        else:
            handle_pos = self.data.site_xpos[self.handle_site_id].ravel()
            palm_pos = self.data.site_xpos[self.grasp_site_id].ravel()

            # get to handle
            reward = 2 * -np.linalg.norm(palm_pos - handle_pos)
            # open door
            reward += 1.5 * goal_distance
            # velocity cost
            reward += -1e-5 * np.sum(self.data.qvel ** 2)

            # Bonus reward
            if goal_distance > 0.1:
                reward += 1
            if goal_distance > 0.3:
                reward += 1
            if goal_distance > 0.5:
                reward += 1
            if goal_distance > 0.7:
                reward += 1
            if goal_distance > 0.9:
                reward += 1
            if goal_distance > 1:
                reward += 1
            if goal_distance > 1.1:
                reward += 0.5
            if goal_distance > 1.2:
                reward += 0.5

        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, dict(success=goal_achieved)

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_site_id].ravel()
        palm_pos = self.data.site_xpos[self.grasp_site_id].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_addrs]])
        if door_pos >= 1.29:
            door_open = 1.0
        else:
            door_open = -1.0

        return np.concatenate(
            [
                qpos,
                palm_pos,
                handle_pos,
                palm_pos - handle_pos,
                [door_open],
            ]
        )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)

        if options is not None and "initial_state_dict" in options:
            self.set_env_state(options["initial_state_dict"])
            obs = self._get_obs()

        return obs, info

    def reset_model(self):
        self.model.body_pos[self.door_body_id, 0] = self.np_random.uniform(
            low=-0.3, high=-0.2
        )
        self.model.body_pos[self.door_body_id, 1] = self.np_random.uniform(
            low=0.25, high=0.35
        )
        self.model.body_pos[self.door_body_id, 2] = self.np_random.uniform(
            low=0.252, high=0.35
        )
        self.set_state(self.init_qpos, self.init_qvel)

        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_body_id].ravel().copy()  # here
        return dict(qpos=qpos, qvel=qvel, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        self.model.body_pos[self.door_body_id] = state_dict["door_body_pos"]
        self.set_state(qp, qv)

environment_name = "CloseDoor-v1"
checkpoint_path = "/Users/eirinipanteli/PycharmProjects/robotic-hand-simulations/Simulations/CloseDoor/Checkpoints/round_2_PPO_checkpoint_seed_1.pth"
save_path = ""
# train(environment_name, save_path, episode_steps=350)
continue_train(environment_name, checkpoint_path, save_path, current_best_reward=10, round=3, episode_steps=350)
# test(environment_name, checkpoint_path, save_path, episode_steps=100)

