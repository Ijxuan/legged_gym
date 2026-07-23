# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace
import unittest

import isaacgym
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.minich.minich_config import MiniChRoughCfg


class TestMiniChRewardShaping(unittest.TestCase):
    def test_hip_symmetry_uses_mirrored_offsets(self):
        robot = SimpleNamespace(
            num_envs=2,
            device="cpu",
            default_dof_pos=torch.tensor([[0.1, -0.1, 0.1, -0.1]]),
            dof_pos=torch.tensor([
                [0.3, -0.3, 0.0, 0.0],
                [0.3, 0.1, 0.1, -0.1],
            ]),
            hip_symmetry_indices=torch.tensor([[0, 1], [2, 3]]),
        )

        reward = LeggedRobot._reward_hip_symmetry(robot)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0, 0.16])))

    def test_zero_command_default_pose_requires_all_velocity_commands_zero(self):
        robot = SimpleNamespace(
            default_dof_pos=torch.zeros(1, 4),
            dof_pos=torch.tensor([
                [0.2, -0.2, 0.0, 0.0],
                [0.2, -0.2, 0.0, 0.0],
            ]),
            commands=torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ]),
            cfg=SimpleNamespace(rewards=SimpleNamespace(zero_command_threshold=0.1)),
        )

        reward = LeggedRobot._reward_zero_command_default_pose(robot)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.02, 0.0])))

    def test_orientation_is_gated_to_low_translation_commands(self):
        robot = SimpleNamespace(
            projected_gravity=torch.tensor([
                [0.6, 0.0, -0.8],
                [0.6, 0.0, -0.8],
            ]),
            commands=torch.tensor([
                [0.25, 0.0, 1.0],
                [0.26, 0.0, 0.0],
            ]),
            cfg=SimpleNamespace(rewards=SimpleNamespace(orientation_command_threshold=0.25)),
        )

        reward = LeggedRobot._reward_orientation(robot)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.36, 0.0])))

    def test_base_height_ignores_only_the_reset_settling_window(self):
        robot = SimpleNamespace(
            root_states=torch.tensor([
                [0.0, 0.0, 0.40],
                [0.0, 0.0, 0.40],
            ]),
            measured_heights=torch.zeros(2, 1),
            episode_length_buf=torch.tensor([24, 25]),
            dt=0.02,
            cfg=SimpleNamespace(rewards=SimpleNamespace(
                base_height_target=0.30,
                base_height_warmup_s=0.5,
            )),
        )

        reward = LeggedRobot._reward_base_height(robot)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0, 0.01])))

    def test_base_height_floor_only_penalizes_below_point_two_two(self):
        robot = SimpleNamespace(
            root_states=torch.tensor([
                [0.0, 0.0, 0.30],
                [0.0, 0.0, 0.22],
                [0.0, 0.0, 0.20],
            ]),
            measured_heights=torch.zeros(3, 1),
            episode_length_buf=torch.tensor([25, 25, 25]),
            dt=0.02,
            cfg=SimpleNamespace(rewards=SimpleNamespace(
                base_height_target=0.26,
                base_height_min=0.22,
                base_height_warmup_s=0.5,
            )),
        )

        reward = LeggedRobot._reward_base_height(robot)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0, 0.0, 0.0004])))

    def test_base_height_target_reward_linearly_decays_to_zero_on_both_sides(self):
        robot = SimpleNamespace(
            root_states=torch.tensor([
                [0.0, 0.0, 0.30],
                [0.0, 0.0, 0.26],
                [0.0, 0.0, 0.24],
                [0.0, 0.0, 0.22],
                [0.0, 0.0, 0.20],
            ]),
            measured_heights=torch.zeros(5, 1),
            episode_length_buf=torch.zeros(5),
            dt=0.02,
            cfg=SimpleNamespace(rewards=SimpleNamespace(
                base_height_target=0.26,
                base_height_reward_drop_height=0.04,
                base_height_warmup_s=0.0,
            )),
        )

        reward = LeggedRobot._reward_base_height(robot)

        # With scale +0.3 and dt=0.02, raw 1.0 becomes +0.006/step.
        # The reward falls linearly to zero at 0.22 m and 0.30 m and never
        # turns into a penalty; the separate 0.15 m threshold handles failure.
        self.assertTrue(torch.allclose(
            reward, torch.tensor([0.0, 1.0, 0.5, 0.0, 0.0]), atol=1e-6
        ))

    def test_actuator_randomization_samples_gains_per_episode_and_clears_delay_history(self):
        robot = SimpleNamespace(
            num_actions=2,
            device="cpu",
            p_gains=torch.tensor([10.0, 20.0]),
            d_gains=torch.tensor([1.0, 2.0]),
            p_gains_per_env=torch.zeros(3, 2),
            d_gains_per_env=torch.zeros(3, 2),
            action_delay_steps=torch.zeros(3, dtype=torch.long),
            action_delay_buffer=torch.ones(3, 2, 2),
            cfg=SimpleNamespace(domain_rand=SimpleNamespace(
                randomize_pd_gains=True,
                kp_scale_range=[0.5, 0.5],
                kd_scale_range=[1.5, 1.5],
                randomize_action_delay=True,
                action_delay_sim_steps=[1, 1],
            )),
        )

        LeggedRobot._reset_actuator_domain_randomization(robot, torch.tensor([0, 2]))

        self.assertTrue(torch.equal(
            robot.p_gains_per_env, torch.tensor([[5.0, 10.0], [0.0, 0.0], [5.0, 10.0]])
        ))
        self.assertTrue(torch.equal(
            robot.d_gains_per_env, torch.tensor([[1.5, 3.0], [0.0, 0.0], [1.5, 3.0]])
        ))
        self.assertTrue(torch.equal(robot.action_delay_steps, torch.tensor([1, 0, 1])))
        self.assertTrue(torch.equal(robot.action_delay_buffer[torch.tensor([0, 2])], torch.zeros(2, 2, 2)))

    def test_action_delay_is_measured_in_physics_substeps(self):
        robot = SimpleNamespace(
            num_envs=2,
            device="cpu",
            action_delay_buffer=torch.zeros(2, 3, 1),
            action_delay_steps=torch.tensor([0, 2]),
            action_delay_buffer_index=0,
            all_env_indices=torch.tensor([0, 1]),
            applied_actions=torch.zeros(2, 1),
            cfg=SimpleNamespace(domain_rand=SimpleNamespace(randomize_action_delay=True)),
        )

        first = LeggedRobot._apply_action_delay(robot, torch.tensor([[1.0], [1.0]])).clone()
        second = LeggedRobot._apply_action_delay(robot, torch.tensor([[2.0], [2.0]])).clone()
        third = LeggedRobot._apply_action_delay(robot, torch.tensor([[3.0], [3.0]])).clone()

        self.assertTrue(torch.equal(first, torch.tensor([[1.0], [0.0]])))
        self.assertTrue(torch.equal(second, torch.tensor([[2.0], [0.0]])))
        self.assertTrue(torch.equal(third, torch.tensor([[3.0], [1.0]])))

    def test_foot_air_time_uses_one_newton_normally_ten_newtons_low_speed_and_is_off_at_point_two(self):
        robot = SimpleNamespace(
            contact_forces=torch.tensor([
                [[0.0, 0.0, 1.0]] * 4,
                [[0.0, 0.0, 1.1]] * 4,
                [[0.0, 0.0, 10.0]] * 4,
                [[0.0, 0.0, 10.1]] * 4,
                [[0.0, 0.0, 1.1]] * 4,
                [[0.0, 0.0, 1.1]] * 4,
            ]),
            feet_indices=torch.tensor([0, 1, 2, 3]),
            feet_air_time=torch.full((6, 4), 0.6),
            last_contacts=torch.zeros(6, 4, dtype=torch.bool),
            dt=0.02,
            commands=torch.tensor([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.19, 0.0, 0.0],
                [0.19, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [0.21, 0.0, 0.0],
            ]),
            cfg=SimpleNamespace(rewards=SimpleNamespace(
                feet_air_time_contact_force_threshold=1.0,
                feet_air_time_low_speed_contact_force_threshold=10.0,
                feet_air_time_low_speed_command_threshold=0.2,
            )),
        )
        robot._foot_normal_forces = lambda: LeggedRobot._foot_normal_forces(robot)

        reward = LeggedRobot._reward_feet_air_time(robot)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0, 0.48, 0.0, 0.0, 0.0, 0.48])))
        self.assertTrue(torch.equal(robot.last_contacts[0], torch.zeros(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(robot.last_contacts[1], torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(robot.last_contacts[2], torch.zeros(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(robot.last_contacts[3], torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(robot.last_contacts[4], torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(robot.last_contacts[5], torch.ones(4, dtype=torch.bool)))

    def test_low_speed_support_rewards_are_thresholded_order_free_and_warmup_gated(self):
        contact_forces = torch.zeros(4, 4, 3)
        contact_forces[0, :, 2] = torch.tensor([20.0, 20.0, 20.0, 20.0])
        contact_forces[1, :, 2] = torch.tensor([20.0, 20.0, 20.0, 10.0])
        contact_forces[2, :, 2] = torch.tensor([20.0, 20.0, 20.0, 40.0])
        contact_forces[3, :, 2] = torch.tensor([20.0, 20.0, 20.0, 20.0])
        robot = SimpleNamespace(
            contact_forces=contact_forces,
            feet_indices=torch.tensor([0, 1, 2, 3]),
            commands=torch.tensor([
                [0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.26, 0.0, 0.0],
            ]),
            episode_length_buf=torch.tensor([25, 25, 25, 24]),
            dt=0.02,
            cfg=SimpleNamespace(rewards=SimpleNamespace(
                foot_contact_force_threshold=10.0,
                low_speed_support_command_threshold=0.25,
                low_speed_support_warmup_s=0.5,
            )),
        )
        robot._foot_normal_forces = lambda: LeggedRobot._foot_normal_forces(robot)
        robot._low_speed_support_mask = lambda: LeggedRobot._low_speed_support_mask(robot)

        missing_support = LeggedRobot._reward_low_speed_missing_support_feet(robot)
        load_balance = LeggedRobot._reward_low_speed_load_balance(robot)

        self.assertTrue(torch.allclose(missing_support, torch.tensor([0.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(load_balance, torch.tensor([0.0, 0.0, 0.03, 0.0])))

    def test_zero_command_sampling_can_force_an_exact_zero_target(self):
        robot = SimpleNamespace(
            device="cpu",
            commands=torch.ones(3, 4),
            zero_command_mask=torch.zeros(3, dtype=torch.bool),
            command_ranges={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [-3.14, 3.14],
            },
            cfg=SimpleNamespace(commands=SimpleNamespace(
                heading_command=True,
                zero_command_probability=1.0,
            )),
        )

        LeggedRobot._resample_commands(robot, torch.tensor([0, 1, 2]))

        self.assertTrue(torch.all(robot.zero_command_mask))
        self.assertEqual(torch.count_nonzero(robot.commands).item(), 0)

    def test_flat_tilt_termination_is_strictly_above_90_degrees(self):
        contact_forces = torch.zeros(5, 1, 3)
        contact_forces[3, 0, 0] = 1.01
        robot = SimpleNamespace(
            contact_forces=contact_forces,
            termination_contact_indices=torch.tensor([0]),
            projected_gravity=torch.tensor([
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [math.sin(math.radians(91.0)), 0.0, -math.cos(math.radians(91.0))],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]),
            episode_length_buf=torch.tensor([0, 0, 0, 0, 101]),
            max_episode_length=100,
            cfg=SimpleNamespace(env=SimpleNamespace(max_base_tilt=math.pi / 2)),
        )

        LeggedRobot.check_termination(robot)

        # Upright and exactly 90 degrees continue.  A 91-degree tilt, the
        # original trunk contact, and the original timeout all reset.
        self.assertTrue(torch.equal(
            robot.reset_buf,
            torch.tensor([False, False, True, True, True]),
        ))
        self.assertTrue(torch.equal(
            robot.time_out_buf,
            torch.tensor([False, False, False, False, True]),
        ))

    def test_min_base_height_termination_is_terrain_relative_and_strict(self):
        robot = SimpleNamespace(
            contact_forces=torch.zeros(3, 1, 3),
            termination_contact_indices=torch.tensor([0]),
            root_states=torch.tensor([
                [0.0, 0.0, 0.150] + [0.0] * 10,
                [0.0, 0.0, 0.349] + [0.0] * 10,
                [0.0, 0.0, 0.251] + [0.0] * 10,
            ]),
            measured_heights=torch.tensor([
                [0.0, 0.0],
                [0.2, 0.2],
                [0.1, 0.1],
            ]),
            episode_length_buf=torch.tensor([0, 0, 0]),
            max_episode_length=100,
            cfg=SimpleNamespace(env=SimpleNamespace(
                max_base_tilt=None,
                min_base_height=0.15,
            )),
        )

        LeggedRobot.check_termination(robot)

        # Exactly 0.15 m remains valid. A 0.149 m terrain-relative height
        # resets, while 0.151 m remains valid even on raised terrain.
        self.assertTrue(torch.equal(robot.reset_buf, torch.tensor([False, True, False])))

    def test_minich_enables_only_the_new_shaping_terms(self):
        self.assertEqual(
            MiniChRoughCfg.rewards.hip_symmetry_joint_pairs,
            (("FL_hip_joint", "FR_hip_joint"), ("RL_hip_joint", "RR_hip_joint")),
        )
        self.assertEqual(MiniChRoughCfg.rewards.scales.hip_symmetry, -5.0)
        self.assertEqual(MiniChRoughCfg.rewards.scales.zero_command_default_pose, -2.0)
        self.assertEqual(MiniChRoughCfg.rewards.scales.orientation, -1.0)
        self.assertEqual(MiniChRoughCfg.rewards.scales.base_height, 0.3)
        self.assertIsNone(MiniChRoughCfg.rewards.base_height_min)
        self.assertEqual(MiniChRoughCfg.rewards.base_height_target, 0.26)
        self.assertEqual(MiniChRoughCfg.rewards.base_height_reward_drop_height, 0.04)
        self.assertEqual(MiniChRoughCfg.rewards.scales.low_speed_missing_support_feet, -1.0)
        self.assertEqual(MiniChRoughCfg.rewards.scales.low_speed_load_balance, -5.0)
        self.assertEqual(MiniChRoughCfg.rewards.orientation_command_threshold, 0.25)
        self.assertEqual(MiniChRoughCfg.rewards.base_height_warmup_s, 0.0)
        self.assertEqual(MiniChRoughCfg.rewards.foot_contact_force_threshold, 10.0)
        self.assertEqual(MiniChRoughCfg.rewards.feet_air_time_contact_force_threshold, 1.0)
        self.assertEqual(MiniChRoughCfg.rewards.feet_air_time_low_speed_contact_force_threshold, 10.0)
        self.assertEqual(MiniChRoughCfg.rewards.feet_air_time_low_speed_command_threshold, 0.2)
        self.assertEqual(MiniChRoughCfg.rewards.low_speed_support_command_threshold, 0.25)
        self.assertEqual(MiniChRoughCfg.rewards.low_speed_support_warmup_s, 0.5)
        self.assertEqual(MiniChRoughCfg.commands.zero_command_probability, 0.10)
        self.assertEqual(MiniChRoughCfg.env.min_base_height, 0.15)


if __name__ == "__main__":
    unittest.main()
