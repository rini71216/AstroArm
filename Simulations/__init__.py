from gymnasium.envs.registration import register

kwargs = {
            "reward_type": "Dense",
        }

register(
            id=f'OpenDoor-v1',
            entry_point='Simulations.OpenDoor.Open_Door:OpenDoor',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Drill-v1',
            entry_point='Simulations.Drill.Step_1.Drill:Drill',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Drill_Step2-v1',
            entry_point='Simulations.Drill.Step_2.Drill:Drill',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Drill_Step3-v1',
            entry_point='Simulations.Drill.Step_3.Drill:Drill',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Wrench_Step1-v1',
            entry_point='Simulations.Wrench.Step_1.Wrench:Wrench',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Wrench_Step2-v1',
            entry_point='Simulations.Wrench.Step_2.Wrench:Wrench',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Pull_Out-v1',
            entry_point='Simulations.Pull_Out.Pull_Out:Pull_Out',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Hand_Off-v1',
            entry_point='Simulations.Hand_Off.Hand_Off:Hand_Off',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Undrill-v1',
            entry_point='Simulations.Undrill.Undrill:Undrill',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Wrench_In-v1',
            entry_point='Simulations.Wrench_In.Wrench_In:Wrench_In',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'Replace-v1',
            entry_point='Simulations.Replace.Replace:Replace',
            max_episode_steps=2000,
            kwargs=kwargs,
        )

register(
            id=f'CloseDoor-v1',
            entry_point='Simulations.CloseDoor.Close_Door:CloseDoor',
            max_episode_steps=2000,
            kwargs=kwargs,
        )