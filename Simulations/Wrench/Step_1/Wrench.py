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

class Wrench(MujocoEnv, EzPickle):
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
            "wrench_door.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float64
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


        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_ARTz"
            ] : self._model_names.actuator_name2id["A_ARRz"]
            + 1,
            :3,
        ] = np.array([500, 0, 0])


        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_ARTz"
            ] : self._model_names.actuator_name2id["A_ARRz"]
            + 1,
            :3,
        ] = np.array([0, -100, 0])

        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_gripper_right"
            ] : self._model_names.actuator_name2id["A_gripper_left"]
            + 1,
            :3,
        ] = np.array([1, 0, 0])


        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_gripper_right"
            ] : self._model_names.actuator_name2id["A_gripper_left"]
            + 1,
            :3,
        ] = np.array([0, -1, 0])



        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        # self.door_hinge_addrs = self.model.jnt_dofadr[
        #     self._model_names.joint_name2id["door_hinge"]
        # ]
        # self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        # self.handle_site_id = self._model_names.site_name2id["S_handle"]

        self.gripper_right_site_id = self._model_names.site_name2id["s_gripper_right_top"]
        self.gripper_left_site_id = self._model_names.site_name2id["s_gripper_left_top"]
        self.nut_site_id = self._model_names.site_name2id["S_nut"]
        self.nut_site_id_2 = self._model_names.site_name2id["S_nut_2"]

        self.gripper_pos = [self.data.site_xpos[self.gripper_right_site_id].ravel(), self.data.site_xpos[self.gripper_left_site_id].ravel()]

        self.door_body_id = self._model_names.body_name2id["frame"] #this was frame instead of door2
        # here
        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
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
        self.nut_pos_site = [self.data.site_xpos[self.nut_site_id].ravel(),
                             self.data.site_xpos[self.nut_site_id_2].ravel()]
        goal_distance_1, goal_distance_2, distance_between_grippers = self.get_distance(self.gripper_pos, self.nut_pos_site)
        # goal_distance = self.data.qpos[self.wrench_pos]
        goal_achieved = True if goal_distance_1 <= 0.0165 and goal_distance_2 <= 0.026 else False

        if goal_achieved:
            reward = 10.0

        else:
            # get close to the right site of the nut
            reward = 0.1 * -goal_distance_1
            # get close to the left site of the nut
            reward += 0.1 * -goal_distance_2

            # bring the hand vertically
            self.ARRx_address = self.model.jnt_dofadr[
                self._model_names.joint_name2id["ARRx"]
            ]
            hand_vertically = self.data.qpos[self.ARRx_address]
            # Hand vertically has a negative value so we multiply with -0.1 to make it positive
            reward += -hand_vertically

            # velocity cost
            reward += -1e-5 * np.sum(self.data.qvel ** 2)

            # Bonus reward
            if goal_distance_1 < 0.2:
                reward += 0.3
            if goal_distance_1 < 0.1:
                reward += 0.3
            if goal_distance_1 < 0.09:
                reward += 0.5
            if goal_distance_1 < 0.08:
                reward += 0.5
            if goal_distance_1 < 0.07:
                reward += 0.5
            if goal_distance_1 < 0.06:
                reward += 0.5
            if goal_distance_1 < 0.05:
                reward += 0.5
            if goal_distance_1 < 0.04:
                reward += 0.5
            if goal_distance_1 < 0.03:
                reward += 0.5
            if goal_distance_1 < 0.02:
                reward += 0.5

            if goal_distance_2 < 0.2:
                reward += 0.3
            if goal_distance_2 < 0.1:
                reward += 0.3
            if goal_distance_2 < 0.09:
                reward += 0.5
            if goal_distance_2 < 0.08:
                reward += 0.5
            if goal_distance_2 < 0.07:
                reward += 0.5
            if goal_distance_2 < 0.06:
                reward += 0.5
            if goal_distance_2 < 0.05:
                reward += 0.5
            if goal_distance_2 < 0.04:
                reward += 0.5
            if goal_distance_2 < 0.03:
                reward += 0.5
            if goal_distance_2 < 0.02:
                reward += 0.5

        self.render()

        return obs, reward, False, False, dict(success=goal_achieved)

    def get_distance(self, gripper_positions, nut_positions_site):
        gripper_pos_1 = gripper_positions[0]
        gripper_pos_2 = gripper_positions[1]
        nut_pos_site_1 = nut_positions_site[0]
        nut_pos_site_2 = nut_positions_site[1]
        distance_1 = torch.sum(torch.norm(torch.from_numpy(gripper_pos_1 - nut_pos_site_1), p=2, dim=-1), dim=-1)
        # distance_1 should be between 0.0140 and 0.0149
        distance_2 = torch.sum(torch.norm(torch.from_numpy(gripper_pos_2 - nut_pos_site_2), p=2, dim=-1), dim=-1)
        # distance_1 should be between 0 and 0.0230
        distance_between_grippers = torch.sum(torch.norm(torch.from_numpy(gripper_pos_1 - gripper_pos_2), p=2, dim=-1), dim=-1)

        return distance_1.item(), distance_2.item(), distance_between_grippers.item()

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = self.data.qpos.ravel()
        nut_pos_site = self.data.site_xpos[self.nut_site_id].ravel()
        nut_pos_site_2 = self.data.site_xpos[self.nut_site_id_2].ravel()
        gripper_pos_1 = self.data.site_xpos[self.gripper_right_site_id].ravel()
        gripper_pos_2 = self.data.site_xpos[self.gripper_left_site_id].ravel()


        distance_1, distance_2, distance_between_grippers = self.get_distance([gripper_pos_1, gripper_pos_2], [nut_pos_site, nut_pos_site_2])
        # distance_1 should be between 0.0140 and 0.0149
        # distance_1 should be between 0 and 0.0230
        if distance_1 <= 0.0165 and distance_2 <= 0.026:
            gripper_on_nut = 1.0
        else:
            gripper_on_nut = -1.0

        return np.concatenate(
            [
                qpos,
                gripper_pos_1,
                gripper_pos_2,
                nut_pos_site,
                nut_pos_site_2,
                gripper_pos_1 - nut_pos_site,
                gripper_pos_2 - nut_pos_site_2,
                [gripper_on_nut],
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
        door_body_pos = self.model.body_pos[self.door_body_id].ravel().copy() #here
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

environment_name = "Wrench_Step1-v1"
save_path = ""
checkpoint_path = "/Users/eirinipanteli/PycharmProjects/robotic-hand-simulations/Simulations/Wrench/Step_1/Checkpoints/round_3_PPO_checkpoint_seed_1.pth"
# train(environment_name, save_path, episode_steps=150)
# continue_train(environment_name, checkpoint_path, save_path, current_best_reward=10.0, round=3)
test(environment_name, checkpoint_path, save_path, episode_steps=150, total_num_episodes=10)