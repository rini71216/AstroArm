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
from sklearn import preprocessing

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
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
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

        # self.model.actuator_gainprm[
        # self._model_names.actuator_name2id[
        #     "A_gripper_right"
        # ]: self._model_names.actuator_name2id["A_gripper_left"]
        #    + 1,
        # :3,
        # ] = np.array([1, 0, 0])
        #
        # self.model.actuator_biasprm[
        # self._model_names.actuator_name2id[
        #     "A_gripper_right"
        # ]: self._model_names.actuator_name2id["A_gripper_left"]
        #    + 1,
        # :3,
        # ] = np.array([0, -1, 0])

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
                self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

        self.gripper_right_site_id = self._model_names.site_name2id["s_gripper_right_top"]
        self.gripper_left_site_id = self._model_names.site_name2id["s_gripper_left_top"]
        self.S_grasp_id = self._model_names.site_name2id["S_grasp"]
        self.nut_site_id = self._model_names.site_name2id["S_nut"]
        self.bolt_site_id = self._model_names.site_name2id["bolt"]
        self.nut_down_addrs = self.model.jnt_dofadr[
            self._model_names.joint_name2id["nut_downwards"]
        ]

        self.S_grasp = self.data.site_xpos[self.S_grasp_id].ravel()
        self.bolt = self.data.site_xpos[self.bolt_site_id].ravel()
        self.nut = self.data.site_xpos[self.nut_site_id].ravel()

        self.gripper_pos = [self.data.site_xpos[self.gripper_right_site_id].ravel(),
                            self.data.site_xpos[self.gripper_left_site_id].ravel()]

        self.door_body_id = self._model_names.body_name2id["frame"]  # this was frame instead of door2
        # here
        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
                ),
                "door_body_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)

        # Scale the gripper to the actuator limits because sometimes they go over
        # self.scaler_gripper_right = preprocessing.MinMaxScaler(feature_range=(self.model.actuator_ctrlrange[2][0], self.model.actuator_ctrlrange[2][1]))
        # self.scaler_gripper_left = preprocessing.MinMaxScaler(
        #     feature_range=(self.model.actuator_ctrlrange[3][0], self.model.actuator_ctrlrange[3][1]))

    def step(self, a):

        # a = np.clip(a, -1.0, 1.0)
        # a = self.act_mean + a * self.act_rng
        a[0] = abs(30 * a[0])

        # # Scale to actual range because the network will return values between -1 and 1
        # a[2] = float(self.scaler_gripper_right.fit_transform(np.array(a[2]).reshape(-1,1)).squeeze())
        # a[3] = float(self.scaler_gripper_left.fit_transform(np.array(a[3]).reshape(-1,1)).squeeze())

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        nut_downwards = self.data.qpos[self.nut_down_addrs]
        goal_achieved = True if nut_downwards >= 0.045 else False


        self.ARRz_address = self.model.jnt_dofadr[
            self._model_names.joint_name2id["ARRz"]
        ]

        # roll hand around and unscrew
        hand_twisting = self.data.qpos[self.ARRz_address]

        # Touch sensors
        touch = self.data.sensordata

        if goal_achieved:
            reward = 10.0

        else:
            # Touch the screw
            reward = 1e-6 * (touch[0] + touch[1] + touch[2])

            reward += 0.1 * hand_twisting

            reward += nut_downwards

            # velocity cost
            reward += -1e-6 * np.sum(self.data.qvel ** 2)

            # bonus
            if nut_downwards > 0.02:
                reward += 2

            if nut_downwards > 0.03:
                reward += 2

            if nut_downwards > 0.04:
                reward += 2

            if nut_downwards > 0.041:
                reward += 2

            if nut_downwards > 0.043:
                reward += 1


        self.render()
        if reward > 8:
            print()


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
        distance_between_grippers = torch.sum(torch.norm(torch.from_numpy(gripper_pos_1 - gripper_pos_2), p=2, dim=-1),
                                              dim=-1)

        # if distance_1 <= 0.0163 and distance_2 <= 0.03 and (distance_between_grippers > 0.03 and distance_between_grippers < 0.034):
        #     print('h')
        return distance_1.item(), distance_2.item(), distance_between_grippers.item()

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = self.data.qpos.ravel()
        nut_downwards = np.array([self.data.qpos[self.nut_down_addrs]])
        bolt = self.bolt
        nut = self.nut
        gripper1 = self.gripper_pos[0]
        gripper2 = self.gripper_pos[1]

        if nut_downwards >= 0.045:
            nut_off_bolt = 1.0
        else:
            nut_off_bolt = -1.0

        return np.concatenate(
            [
                qpos,
                nut_downwards,
                gripper1,
                gripper2,
                nut,
                nut - bolt,
                [nut_off_bolt]
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


environment_name = "Wrench_Step2-v1"
save_path = ""
checkpoint_path = "/Users/eirinipanteli/PycharmProjects/robotic-hand-simulations/Simulations/Wrench/Step_2/Checkpoints/round_2_PPO_checkpoint_seed_1.pth"
# train(environment_name, save_path, episode_steps=200)
# continue_train(environment_name, checkpoint_path, save_path, current_best_reward=9.6, episode_steps=200)
test(environment_name, checkpoint_path, save_path, episode_steps=200, total_num_episodes=10)