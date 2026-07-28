# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import math
import unittest

from legged_gym.envs import task_registry
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.envs.minich.minich_flat_config import (
    MINI_CHEETAH_CALF_LINK_LENGTH_RANGE_M,
    MINI_CHEETAH_LEG_LENGTH_RANDOMIZATION_NUM_VARIANTS,
    MINI_CHEETAH_THIGH_LINK_LENGTH_RANGE_M,
    MiniChFlatCfg,
    MiniChFlatCfgPPO,
)


class TestMiniChFlatConfig(unittest.TestCase):
    def test_task_is_registered_with_the_flat_configs(self):
        env_cfg, train_cfg = task_registry.get_cfgs("minich_flat")
        self.assertIsInstance(env_cfg, MiniChFlatCfg)
        self.assertIsInstance(train_cfg, MiniChFlatCfgPPO)

    def test_rough_task_keeps_its_235_dimensional_interface(self):
        rough_cfg, _ = task_registry.get_cfgs("minich")
        self.assertIsInstance(rough_cfg, MiniChRoughCfg)
        self.assertEqual(rough_cfg.env.num_observations, 235)
        self.assertTrue(rough_cfg.terrain.measure_heights)

    def test_flat_task_uses_48_dimensional_proprioceptive_observation(self):
        self.assertEqual(MiniChFlatCfg.env.num_observations, 48)
        self.assertEqual(MiniChFlatCfg.terrain.mesh_type, "plane")
        self.assertFalse(MiniChFlatCfg.terrain.measure_heights)
        self.assertFalse(MiniChFlatCfg.terrain.curriculum)
        self.assertEqual(MiniChFlatCfg.env.max_base_tilt, math.pi / 2)
        self.assertIsNone(MiniChRoughCfg.env.max_base_tilt)

    def test_flat_task_uses_direct_three_axis_velocity_commands(self):
        self.assertEqual(MiniChFlatCfg.commands.num_commands, 3)
        self.assertFalse(MiniChFlatCfg.commands.heading_command)
        self.assertEqual(MiniChFlatCfg.commands.ranges.lin_vel_x, [-1.0, 1.0])
        self.assertEqual(MiniChFlatCfg.commands.ranges.lin_vel_y, [-1.0, 1.0])
        self.assertEqual(MiniChFlatCfg.commands.ranges.ang_vel_yaw, [-1, 1])
        self.assertEqual(MiniChFlatCfg.commands.resampling_time, 5.0)
        self.assertEqual(MiniChFlatCfg.commands.zero_command_probability, 0.30)

    def test_flat_task_keeps_domain_randomization_and_starts_fresh(self):
        self.assertTrue(MiniChFlatCfg.domain_rand.randomize_friction)
        self.assertTrue(MiniChFlatCfg.domain_rand.randomize_base_mass)
        self.assertEqual(MiniChFlatCfg.domain_rand.added_mass_range, [-1.0, 1.0])
        self.assertTrue(MiniChFlatCfg.domain_rand.push_robots)
        self.assertTrue(MiniChFlatCfg.domain_rand.randomize_leg_lengths)
        self.assertTrue(MiniChFlatCfg.domain_rand.randomize_pd_gains)
        self.assertEqual(MiniChFlatCfg.domain_rand.kp_scale_range, [0.6, 1.0])
        self.assertEqual(MiniChFlatCfg.domain_rand.kd_scale_range, [0.6, 1.0])
        self.assertTrue(MiniChFlatCfg.domain_rand.randomize_action_delay)
        self.assertEqual(MiniChFlatCfg.domain_rand.action_delay_sim_steps, [0, 3])
        self.assertEqual(
            MiniChFlatCfg.domain_rand.thigh_link_length_range_m,
            MINI_CHEETAH_THIGH_LINK_LENGTH_RANGE_M,
        )
        self.assertEqual(
            MiniChFlatCfg.domain_rand.calf_link_length_range_m,
            MINI_CHEETAH_CALF_LINK_LENGTH_RANGE_M,
        )
        self.assertEqual(
            MiniChFlatCfg.domain_rand.leg_length_randomization_num_variants,
            MINI_CHEETAH_LEG_LENGTH_RANDOMIZATION_NUM_VARIANTS,
        )
        self.assertEqual(MiniChFlatCfg.control.action_scale, 0.25)
        self.assertEqual(MiniChFlatCfg.rewards.scales.hip_symmetry, -5.0)
        self.assertEqual(MiniChFlatCfg.rewards.scales.zero_command_default_pose, -10.0)
        self.assertEqual(MiniChFlatCfg.rewards.base_height_target, 0.26)
        self.assertEqual(MiniChFlatCfg.rewards.base_height_reward_drop_height, 0.04)
        self.assertEqual(MiniChFlatCfg.rewards.scales.base_height, 0.3)
        self.assertEqual(MiniChFlatCfg.rewards.foot_contact_force_threshold, 10.0)
        self.assertEqual(MiniChFlatCfg.rewards.low_speed_support_command_threshold, 0.25)
        self.assertEqual(MiniChFlatCfg.rewards.low_speed_support_warmup_s, 0.5)
        self.assertEqual(MiniChFlatCfg.rewards.low_speed_support_zero_command_delay_s, 1.0)
        self.assertEqual(MiniChFlatCfg.rewards.scales.low_speed_missing_support_feet, -1.0)
        self.assertEqual(MiniChFlatCfg.rewards.scales.low_speed_load_balance, -1.0)
        self.assertEqual(MiniChFlatCfgPPO.runner.experiment_name, "flat_minich")
        self.assertFalse(MiniChFlatCfgPPO.runner.resume)
        self.assertEqual(MiniChFlatCfgPPO.runner.jit_export_interval, 1000)
        self.assertTrue(MiniChFlatCfgPPO.runner.jit_export_on_exit)


if __name__ == "__main__":
    unittest.main()
