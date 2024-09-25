from Simulations import *
from Simulations.PPO_Policy import train, continue_train, test
from panda3d.core import loadPrcFile
import time
from os import path
from typing import Optional
import torch
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

class Replace(MujocoEnv, EzPickle):
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
            "replace_door.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64
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

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
                self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

        self.ARTx = self.model.jnt_dofadr[
            self._model_names.joint_name2id["ARTx"]
        ]

        self.ARRy = self.model.jnt_dofadr[
            self._model_names.joint_name2id["ARRy"]
        ]

        self.ARRz = self.model.jnt_dofadr[
            self._model_names.joint_name2id["ARRz"]
        ]

        self.door_site = self._model_names.site_name2id["door_site"]
        self.door = self.data.site_xpos[self.door_site].ravel()
        self.S_si_unit = self._model_names.site_name2id["si_unit"]
        self.si_unit = self.data.site_xpos[self.S_si_unit].ravel()
        self.door_body_id = self._model_names.body_name2id["frame"]  # this was frame instead of door2
        # here
        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
                ),
                "door_body_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)

    def get_distance(self, door, si_unit):
        distance = torch.sum(torch.norm(torch.from_numpy(door - si_unit), p=2, dim=-1), dim=-1)
        # distance should be 0.0224, we'll say less than 0.03

        return distance.item()

    def step(self, a):
        a = np.clip(a, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        a = self.act_mean + a * self.act_rng  # mean center and scale
        # a = np.concatenate(self.act_mean,a) * self.act_rng

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        # compute the sparse reward variant first
        goal_distance = self.get_distance(self.door, self.si_unit)
        goal_achieved = True if goal_distance <= 0.0375 else False
        reward = 10.0 if goal_achieved else -0.1

        # override reward if not sparse reward
        if not goal_achieved:

            # get close to the handle, the smaller the distance the better
            reward = -goal_distance

            # velocity cost
            reward += -1e-5 * np.sum(self.data.qvel ** 2)

            # Bonus reward
            if goal_distance < 0.4:
                reward += 1
            if goal_distance < 0.3:
                reward += 0.5
            if goal_distance < 0.2:
                reward += 0.5
            if goal_distance < 0.1:
                reward += 0.5
            if goal_distance < 0.09:
                reward += 0.5
            if goal_distance < 0.08:
                reward += 1
            if goal_distance < 0.07:
                reward += 1
            if goal_distance < 0.06:
                reward += 1
            if goal_distance < 0.05:
                reward += 1
            if goal_distance < 0.04:
                reward += 1
            if goal_distance < 0.035:
                reward += 1

        if self.render_mode == "human":
            self.render()
            # time.sleep(.1)

        return obs, reward, False, False, dict(success=goal_achieved)

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = self.data.qpos.ravel()
        distance = self.get_distance(self.door, self.si_unit)
        if distance <= 0.0375:
            si_unit_on_base = 1.0
        else:
            si_unit_on_base = -1.0

        return np.concatenate(
            [
                qpos,
                self.door - self.si_unit,
                [si_unit_on_base],
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
        # assert self._state_space.contains(
        #     state_dict
        # ), f"The state dictionary {state_dict} must be a member of {self._state_space}."
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        self.model.body_pos[self.door_body_id] = state_dict["door_body_pos"]
        self.set_state(qp, qv)


environment_name = "Replace-v1"
checkpoint_path = "/Users/eirinipanteli/PycharmProjects/robotic-hand-simulations/Simulations/Replace/Checkpoints/PPO_checkpoint_seed_1.pth"
save_path = ""
train(environment_name, save_path, episode_steps=300)
# continue_train(environment_name, checkpoint_path, save_path, current_best_reward=10.0, episode_steps=300)
# test(environment_name, checkpoint_path, save_path, episode_steps=200)