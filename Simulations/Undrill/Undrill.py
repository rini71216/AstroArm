from Simulations.PPO_Policy import train, continue_train, test
from panda3d.core import loadPrcFile
from Simulations.Drill import *
import time
from os import path
from typing import Optional
import numpy as np
from gymnasium import spaces
import torch
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}
loadPrcFile("myConfig.prc")

class Undrill(MujocoEnv, EzPickle):
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
            "undrill_door.xml",
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
        # self.door_hinge_addrs = self.model.jnt_dofadr[
        #     self._model_names.joint_name2id["door_hinge"]
        # ]
        # self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        # self.handle_site_id = self._model_names.site_name2id["S_handle"]

        # self.scre = self._model_names.site_name2id["s_gripper_right_top"]
        # self.gripper_left_site_id = self._model_names.site_name2id["s_gripper_left_top"]
        self.screw_site_id = self._model_names.site_name2id["S_screw"]
        self.screw = self.data.site_xpos[self.screw_site_id].ravel()
        self.screw_addrs = self.model.jnt_dofadr[
            self._model_names.joint_name2id["screw_inwards"]
        ]
        self.screw_driver_site_id = self._model_names.site_name2id["S_screw_driver"]
        self.screw_driver = self.data.site_xpos[self.screw_driver_site_id].ravel()
        self.door_site_id = self._model_names.site_name2id["S_door"]
        self.door_site = self.data.site_xpos[self.door_site_id].ravel()
        self.screw_extension1_site_id = self._model_names.site_name2id["S_screw_extension_1"]
        self.screw_extension1 = self.data.site_xpos[self.screw_extension1_site_id].ravel()
        self.screw_extension2_site_id = self._model_names.site_name2id["S_screw_extension_2"]
        self.screw_extension2 = self.data.site_xpos[self.screw_extension2_site_id].ravel()

        # self.gripper_pos = [self.data.site_xpos[self.gripper_right_site_id].ravel(), self.data.site_xpos[self.gripper_left_site_id].ravel()]

        self.door_body_id = self._model_names.body_name2id["frame"]  # this was frame instead of door2
        # here
        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
                "door_body_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mean + a * self.act_rng  # mean center and scale
        # a = np.concatenate(self.act_mean,a) * self.act_rng

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        # compute the sparse reward variant first
        # self.nut_pos_site = [self.data.site_xpos[self.nut_site_id].ravel(),
        #                      self.data.site_xpos[self.nut_site_id_2].ravel()]
        goal_distance1, goal_distance2, goal_distance3 = self.get_distance(self.screw, self.screw_addrs,
                                                                           self.screw_extension1, self.screw_extension2)
        # goal_distance = self.data.qpos[self.wrench_pos]

        # Check to see the screw has been screw by drilling and just by pushing it. It needs 5 rounds for the screw to be in
        self.screw_around_joint = self.model.jnt_dofadr[
            self._model_names.joint_name2id["screw_around"]
        ]

        goal_achieved = True if (goal_distance1 >= 0.055) else False

        if goal_achieved:
            reward = 10.0

        else:

            # reward for drilling
            reward = goal_distance1

            # velocity cost
            reward += -1e-8 * np.sum(self.data.qvel ** 2)

            # Bonus reward
            if goal_distance1 > 0:
                reward += 1

            if goal_distance1 > 0.005:
                reward += 1

            if goal_distance1 > 0.01:
                reward += 1

            if goal_distance1 > 0.02:
                reward += 1

            if goal_distance1 > 0.03:
                reward += 1

            if goal_distance1 > 0.05:
                reward += 2

            if goal_distance1 > 0.053:
                reward += 2

        # print("Reward:", reward)

        # if self.render_mode == "human":
        #     self.render()
        self.render()
        # time.sleep(.1)

        return obs, reward, False, False, dict(success=goal_achieved)

    def get_distance(self, screw, screw_address, screw_extension1, screw_extension2):
        distance_screw_door = self.data.qpos[screw_address]
        distance_screw_driver_extension_1 = torch.sum(
            torch.norm(torch.from_numpy(screw_extension1 - screw), p=2, dim=-1), dim=-1)
        distance_screw_driver_extension_2 = torch.sum(
            torch.norm(torch.from_numpy(screw - screw_extension2), p=2, dim=-1), dim=-1)
        # distance_extension_1 should be greater than 0.0140 and distance_extension_2 should be less than 0.0128

        return distance_screw_door.item(), distance_screw_driver_extension_1.item(), distance_screw_driver_extension_2.item()

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        # Qpositions excepct fot ARTz
        qpos = self.data.qpos.ravel()

        # gripper_pos_1 = self.data.site_xpos[self.gripper_right_site_id].ravel()
        # gripper_pos_2 = self.data.site_xpos[self.gripper_left_site_id].ravel()

        distance_screw_door, distance_screw_driver_extension_1, distance_screw_driver_extension_2 = self.get_distance(
            self.screw, self.screw_addrs, self.screw_extension1, self.screw_extension2)
        # distance should be greater than 0.0592
        if distance_screw_door >= 0.055:
            screw_out = 1.0
        else:
            screw_out = -1.0

        return np.concatenate(
            [
                qpos,
                self.screw_driver,
                self.screw,
                self.screw - self.door_site,
                [screw_out],
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


environment_name = "Undrill-v1"
save_path = ""
checkpoint_path = "/Users/eirinipanteli/PycharmProjects/robotic-hand-simulations/Simulations/Drill/Step_3/Checkpoints/round_2_PPO_checkpoint_seed_1.pth"
train(environment_name, save_path, episode_steps=150)
# continue_train(environment_name, checkpoint_path, save_path, current_best_reward=8.5)
# test(environment_name, checkpoint_path, save_path, episode_steps=150)