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
        added_mass_range = [-1.0, 1.0]
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
        action_delay_sim_steps = [0, 2]

    class rewards(MiniChRoughCfg.rewards):
        class scales(MiniChRoughCfg.rewards.scales):
            termination = -0.0  # 终止惩罚
            tracking_lin_vel = 1.0  # 线速度跟踪
            tracking_ang_vel = 1.0  # 角速度跟踪
            lin_vel_z = -2.0  # 竖直速度惩罚
            ang_vel_xy = -0.05  # 横滚/俯仰角速度惩罚
            orientation = -2.0  # 低速姿态对齐
            torques = -0.0002  # 关节力矩惩罚
            dof_vel = -0.0  # 关节速度惩罚
            dof_acc = -2.5e-7  # 关节加速度惩罚
            action_rate = -0.01  # 动作变化率惩罚
            base_height = 0.3  # 机身高度奖励
            feet_air_time = 1.0  # 腾空时间奖励
            collision = 0.0  # 通用碰撞惩罚
            thigh_collision = -1.0  # 大腿碰撞惩罚
            calf_collision = -1.0  # 小腿碰撞惩罚
            feet_stumble = -0.0  # 绊倒惩罚
            stand_still = -0.0  # 低速站立关节偏移惩罚
            feet_contact_forces = -0.0  # 足端接触力过大惩罚
            dof_pos_limits = -10.0  # 关节位置限制惩罚
            dof_vel_limits = -0.0  # 关节速度限制惩罚
            torque_limits = -0.0  # 力矩限制惩罚
            hip_symmetry = -5.0  # 左右髋关节对称惩罚
            zero_command_default_pose = -3.0  # 零命令默认站姿惩罚
            low_speed_missing_support_feet = -1.0  # 低速缺支撑足惩罚
            low_speed_load_balance = -5.0  # 低速四足载荷均衡惩罚
            joint_power = 0.0  # 未实现
            foot_clearance = 0.5  # 转向时的四足周期最大抬腿高度奖励
            smoothness = 0.0  # 未实现

        # 说明：
        # - 这里是 minich_flat 平地训练专用奖励权重；写在这里会覆盖
        #   MiniChRoughCfg 继承来的同名权重，但不会修改 rough 任务。
        # - 权重为 0.0 或 -0.0 的项会在 LeggedRobot._prepare_reward_function()
        #   中被移除，不会参与训练，也不会要求存在对应奖励函数。
        # - joint_power、smoothness 目前没有对应的
        #   _reward_joint_power/_reward_smoothness 函数，所以只能保持 0.0；
        #   如果改成非零，启动环境时会报错。
        # - foot_clearance 已实现；原始值为 0~1，当前 0.5 权重对应最高约
        #   +0.01/step。它只在 |cmd_yaw| > 0.2 时启用，每 1.0 s 统计
        #   每条腿的最大足底离地高度，再取四条腿 max 的最小值评分；
        #   纯前进、纯后退和零速都不会拿这个奖励。
        # - orientation 只在 ||cmd_xy|| <= orientation_command_threshold 时生效；
        #   原地旋转时 cmd_xy 为 0，因此姿态惩罚仍会生效。


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
