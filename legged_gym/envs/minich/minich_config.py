# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MiniChRoughCfg(LeggedRobotCfg):
    """Mini Cheetah task using the standard 12-DOF URDF asset."""

    class env(LeggedRobotCfg.env):
        # End collapsed/crawling states before they become an attractive
        # low-speed solution. This is height above the local terrain.
        min_base_height = 0.15

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,

            "FL_thigh_joint": -0.8,
            "RL_thigh_joint": -0.8,
            "FR_thigh_joint": -0.8,
            "RR_thigh_joint": -0.8,

            "FL_calf_joint": 1.62,
            "RL_calf_joint": 1.62,
            "FR_calf_joint": 1.62,
            "RR_calf_joint": 1.62,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 17.0, "calf_joint": 34.0}
        damping = {"joint": 0.4, "calf_joint": 0.8}
        action_scale = 0.25
        decimation = 4

    class commands(LeggedRobotCfg.commands):
        # 每 5 s 重采样一次命令：20 s episode 内有四段命令，既能保留足够
        # 长的稳定/调整窗口，也能更频繁地练习行走与零速站立切换。
        resampling_time = 5.0
        # 每次重采样时，有 30% 概率将 x/y/yaw 三个命令强制设为精确零，
        # 让策略学习稳定站立，而不是只学习跟踪非零速度。
        zero_command_probability = 0.30

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
        name = "minich"
        foot_name = "foot"
        # Mini Cheetah separates thigh/calf collision rewards so the calf
        # mesh can use a lower penalty while preserving the original A1 path.
        penalize_contacts_on = []
        penalize_thigh_contacts_on = ["thigh"]
        penalize_calf_contacts_on = ["calf"]
        # The Mini Cheetah trunk has the torso collision geometry; its base link
        # is only the fixed root link and does not carry the torso collision.
        terminate_after_contacts_on = ["trunk"]
        # Keep the fixed foot links as separate rigid bodies for foot indexing.
        collapse_fixed_joints = False
        # calf 碰撞网格与后补的 foot 球形碰撞体存在几何交叠；1 表示关闭
        # actor 内部自碰撞，避免 calf-foot 固定连接处被持续当作自碰撞处理。
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        # Match the A1 task's position-limit and torque regularization.
        # 关节软位置限位比例：将 URDF 的硬限位收缩到 90%，超出后产生惩罚。
        soft_dof_pos_limit = 0.9
        # Give a target-height bonus rather than using the previous one-sided
        # low-height penalty. It is maximal at 0.26 m, then decays linearly to
        # zero at 0.22 m and 0.30 m; the 0.15 m reset remains the hard floor.
        base_height_target = 0.26
        base_height_reward_drop_height = 0.04
        # Disable the old one-sided floor reward for this task. The 0.15 m
        # hard reset remains configured separately in env.min_base_height.
        base_height_min = None
        # Left/right hip defaults are mirrored (+0.1 / -0.1). Compare their
        # offsets from that default so the gait remains physically symmetric.
        hip_symmetry_joint_pairs = (
            ("FL_hip_joint", "FR_hip_joint"),
            ("RL_hip_joint", "RR_hip_joint"),
        )
        # Treat x/y/yaw command norm below this threshold as a stand command.
        zero_command_threshold = 0.1
        # 仅在 sqrt(cmd_vx^2 + cmd_vy^2) <= 0.25 m/s 时计算 roll/pitch
        # 姿态惩罚：零速/慢速时要求机身保持水平；超过 0.25 m/s 后该项为
        # 0，让快速行走自行选择机身姿态。0.25 的单位是 m/s，不是 0.25 秒；
        # 偏航目标 cmd_yaw 不参与这个门控。
        orientation_command_threshold = 0.25
        # The target band includes the reset height, so apply it immediately:
        # a 0.5 s free window previously let the zero-command policy crouch
        # below the band before it received any height feedback.
        base_height_warmup_s = 0.0
        # Low-speed support/load shaping needs a real supporting contact.
        foot_contact_force_threshold = 10.0
        low_speed_support_command_threshold = 0.25
        low_speed_support_warmup_s = 0.5
        # 从非零行走命令切换到精确零速命令后，前 1.0 s 暂不计算四足
        # 支撑缺失/载荷均衡项，给策略调整落脚和机身姿态的时间。低速但非零
        # 的命令仍仅使用上面的 reset warmup，不会额外等待。
        low_speed_support_zero_command_delay_s = 1.0
        # Preserve the original 1 N walking touchdown event. Below 0.2 m/s,
        # however, a 10 N threshold prevents a lightly tapping RR foot from
        # being treated as an adequate ground contact by feet_air_time.
        feet_air_time_contact_force_threshold = 1.0
        feet_air_time_low_speed_contact_force_threshold = 10.0
        feet_air_time_low_speed_command_threshold = 0.2

        class scales(LeggedRobotCfg.rewards.scales):
            # Disable the combined collision reward for minich; thigh/calf
            # are logged and weighted independently.
            collision = 0.0
            thigh_collision = -1.0
            calf_collision = -1.0 / 3.0
            # 力矩平方和惩罚权重，抑制策略使用过大的关节驱动力矩。
            torques = -0.0002
            # 关节位置超出软限位后的误差惩罚权重。
            dof_pos_limits = -10.0
            # 抑制左右髋同向内收/外展，避免出现内八站姿。
            hip_symmetry = -5.0
            # 仅在零速度目标下将全身关节拉回初始化 default 站姿。
            zero_command_default_pose = -10.0
            # The target-height function returns +1 at 0.26 m. With the
            # current +0.3 scale and 20 ms control step, that is +0.006/step.
            orientation = -1.0
            base_height = 0.3
            # At low x/y commands, first require all four feet to support and
            # then nudge their normal-force shares toward an even split.
            low_speed_missing_support_feet = -1.0
            low_speed_load_balance = -1.0


class MiniChRoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_minich"
