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

class Pull_Out(MujocoEnv, EzPickle):
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
            "pull_out_door.xml",
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

        # Override action_space to -1, 1
        # self.action_space = spaces.Box(
        #     low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        # )

        # change actuator sensitivity
        # self.model.actuator_gainprm[
        #     self._model_names.actuator_name2id[
        #         "A_WRJ1"
        #     ] : self._model_names.actuator_name2id["A_WRJ0"]
        #     + 1,
        #     :3,
        # ] = np.array([10, 0, 0])

        # self.model.actuator_gainprm[
        #     self._model_names.actuator_name2id[
        #         "A_ARTz"
        #     ] : self._model_names.actuator_name2id["A_ARRz"]
        #     + 1,
        #     :3,
        # ] = np.array([500, 0, 0])
        #
        # self.model.actuator_gainprm[
        # self._model_names.actuator_name2id[
        #     "A_gripper_right_top"
        # ]: self._model_names.actuator_name2id["A_gripper_left_top"]
        #    + 1,
        # :3,
        # ] = np.array([10, 0, 0])
        #
        # # self.model.actuator_biasprm[
        # #     self._model_names.actuator_name2id[
        # #         "A_WRJ1"
        # #     ] : self._model_names.actuator_name2id["A_WRJ0"]
        # #     + 1,
        # #     :3,
        # # ] = np.array([0, -10, 0])
        #
        # self.model.actuator_biasprm[
        #     self._model_names.actuator_name2id[
        #         "A_ARTz"
        #     ] : self._model_names.actuator_name2id["A_gripper_left_top"]
        #     + 1,
        #     :3,
        # ] = np.array([0, -100, 0])
        #
        # self.model.actuator_biasprm[
        # self._model_names.actuator_name2id[
        #     "A_ARTz"
        # ]: self._model_names.actuator_name2id["A_gripper_left_top"]
        #    + 1,
        # :3,
        # ] = np.array([0, -100, 0])

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
                self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.si_unit_hinge_addrs = self.model.jnt_dofadr[
            self._model_names.joint_name2id["si_unit"]
        ]
        self.ARTz = self.model.jnt_dofadr[
            self._model_names.joint_name2id["ARTz"]
        ]
        self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.handle_site_id = self._model_names.site_name2id["S_si_unit_handle"]
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
        # a = np.clip(a, -1.0, 1.0)
        # a = self.act_mean + a * self.act_rng  # mean center and scale
        # a = np.concatenate(self.act_mean,a) * self.act_rng

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        # compute the sparse reward variant first
        goal_distance = self.data.qpos[self.si_unit_hinge_addrs]
        goal_achieved = True if goal_distance >= 0.05 else False
        reward = 10.0 if goal_achieved else -0.1

        # override reward if not sparse reward
        if not goal_achieved:
            handle_pos = self.data.site_xpos[self.handle_site_id].ravel()
            gripper_pos = self.data.site_xpos[self.grasp_site_id].ravel()

            # get to handle, the closer the better
            reward = -10 * np.linalg.norm(gripper_pos - handle_pos)
            # pull out si_unit
            reward += 10 * goal_distance
            # velocity cost
            reward += -1e-5 * np.sum(self.data.qvel ** 2)

            # Bonus reward
            if goal_distance > 0.01:
                reward += 0.5
            if goal_distance > 0.02:
                reward += 0.5
            if goal_distance > 0.03:
                reward += 0.5
            if goal_distance > 0.035:
                reward += 0.5
            if goal_distance > 0.04:
                reward += 1
            if goal_distance > 0.045:
                reward += 1
            if goal_distance > 0.048:
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
        handle_pos = self.data.site_xpos[self.handle_site_id].ravel()
        gripper_pos = self.data.site_xpos[self.grasp_site_id].ravel()
        si_unit_pos = self.data.qpos[self.si_unit_hinge_addrs]
        if si_unit_pos >= 0.05:
            si_unit_pulled = 1.0
        else:
            si_unit_pulled = -1.0

        return np.concatenate(
            [
                qpos,
                gripper_pos,
                handle_pos,
                gripper_pos - handle_pos,
                [si_unit_pulled],
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


environment_name = "Pull_Out-v1"
save_path = ""
checkpoint_path = "/Users/eirinipanteli/PycharmProjects/robotic-hand-simulations/Simulations/Pull_Out/Checkpoints/round_2_PPO_checkpoint_seed_1.pth"
# train(environment_name, save_path, episode_steps=200)
# continue_train(environment_name, checkpoint_path, save_path, episode_steps=200, current_best_reward=10, round=2)
test(environment_name, checkpoint_path, save_path, episode_steps=200, total_num_episodes=10)