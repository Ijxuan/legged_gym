# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground training configuration for the Mini Cheetah.

This task intentionally shares the Mini Cheetah asset, controller, reward
shaping, and PPO settings with ``minich``.  It uses Isaac Gym's plane ground,
direct x/y/yaw-rate commands, and no 17 x 11 terrain-height observation grid,
so the policy interface is the deployable 48-dimensional proprioceptive
observation.
"""

import math

from legged_gym.envs.minich.minich_config import MiniChRoughCfg, MiniChRoughCfgPPO

# ===== Mini Cheetah 平地 48 维训练：腿部连杆长度随机化（单位：米） =====
# 大腿连杆为髋关节到膝关节，原始 URDF 长度为 0.209 m；可直接修改下面的最小/最大值。
MINI_CHEETAH_THIGH_LINK_LENGTH_RANGE_M = [0.203, 0.215]
# 小腿连杆为膝关节到足端球心，原始 URDF 长度为 0.190 m；可直接修改下面的最小/最大值。
MINI_CHEETAH_CALF_LINK_LENGTH_RANGE_M = [0.184, 0.196]
# 启动训练时生成的不同形态数量；每个并行环境固定随机选一种，不会在 episode 中途改变腿长。
MINI_CHEETAH_LEG_LENGTH_RANDOMIZATION_NUM_VARIANTS = 32


class MiniChFlatCfg(MiniChRoughCfg):
    """Mini Cheetah flat-ground task with 48 proprioceptive observations."""

    class env(MiniChRoughCfg.env):
        # base lin/ang velocity (6), projected gravity (3), x/y/yaw command
        # (3), joint position/velocity (24), and previous action (12).
        num_observations = 48
        # End irrecoverable rolls / upside-down states even when the trunk
        # collision mesh is held above the plane by the motors or legs.
        max_base_tilt = math.pi / 2

    class terrain(MiniChRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(MiniChRoughCfg.commands):
        # Keep direct x/y/yaw-rate targets.  A heading target would make the
        # yaw-rate command depend on the simulated body heading and does not
        # match the three command values consumed by deployment.
        num_commands = 3
        heading_command = False
        curriculum = False

    class domain_rand(MiniChRoughCfg.domain_rand):
        # Explicitly retain the sim-to-real randomization used by the rough
        # task while enabling base-mass variation for this new task.
        randomize_friction = True
        randomize_base_mass = True
        push_robots = True
        # 对四条腿同时使用同一组大腿/小腿长度，避免引入非真实的左右不对称装配误差。
        randomize_leg_lengths = True
        thigh_link_length_range_m = MINI_CHEETAH_THIGH_LINK_LENGTH_RANGE_M
        calf_link_length_range_m = MINI_CHEETAH_CALF_LINK_LENGTH_RANGE_M
        leg_length_randomization_num_variants = MINI_CHEETAH_LEG_LENGTH_RANDOMIZATION_NUM_VARIANTS
        # Per-episode actuator uncertainty. Nominal gains remain 17/34 Kp and
        # 0.4/0.8 Kd; each joint samples an independent multiplier. The policy
        # runs at 50 Hz (20 ms), so 0..4 physics substeps is 0..20 ms delay.
        randomize_pd_gains = True
        kp_scale_range = [0.6, 1.0]
        kd_scale_range = [0.6, 1.0]
        randomize_action_delay = True
        action_delay_sim_steps = [0, 3]


class MiniChFlatCfgPPO(MiniChRoughCfgPPO):
    class runner(MiniChRoughCfgPPO.runner):
        # A fresh 48-D policy cannot resume a 235-D rough-terrain checkpoint.
        experiment_name = "flat_minich"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        # Produce deployable 48-D policy snapshots during the full run, plus
        # one stable latest-named artifact at a controlled program exit.
        jit_export_interval = 1000
        jit_export_on_exit = True
