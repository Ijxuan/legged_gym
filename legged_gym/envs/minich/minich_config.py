# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MiniChRoughCfg(LeggedRobotCfg):
    """使用标准 12 自由度 URDF 资产的 Mini Cheetah 任务配置。"""

    class env(LeggedRobotCfg.env):
        # 在其演化为吸引人但低速的折叠/爬行解之前，先终止这类状态。
        # 这里的高度是相对于局部地形的高度。
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
        # 每 5 s 重采样一次命令：20 s 的 episode 内有四段命令，既能保留足够
        # 长的稳定/调整窗口，也能更频繁地练习行走与零速站立切换。
        resampling_time = 5.0
        # 每次重采样时，有 10% 概率将 x/y/yaw 三个命令强制设为精确零，
        # 让策略学习稳定站立，而不是只学习跟踪非零速度。
        zero_command_probability = 0.10

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
        name = "minich"
        foot_name = "foot"
        # Mini Cheetah 将大腿/小腿碰撞奖励拆开处理，这样小腿网格可以使用更低的惩罚，
        # 同时保留原始 A1 路径的行为。
        penalize_contacts_on = []
        penalize_thigh_contacts_on = ["thigh"]
        penalize_calf_contacts_on = ["calf"]
        # Mini Cheetah 的躯干具有躯干碰撞几何；其基础连杆只是固定根节点，
        # 不承载躯干碰撞。
        terminate_after_contacts_on = ["trunk"]
        # 保留固定足部连杆作为独立刚体，以便进行足部索引。
        collapse_fixed_joints = False
        # 小腿碰撞网格与后补的 foot 球形碰撞体存在几何重叠；1 表示关闭
        # actor 内部自碰撞，避免 calf-foot 固定连接处被持续当作自碰撞处理。
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        # 与 A1 任务保持一致的位置极限和扭矩正则化。
        # 关节软位置限位比例：将 URDF 的硬限位收缩到 90%，超出后产生惩罚。
        soft_dof_pos_limit = 0.9
        # 使用目标高度奖励，而不是之前的单侧低高度惩罚。
        # 其在 0.26 m 处达到最大值，然后线性衰减到 0.22 m 和 0.30 m 处为 0；
        # 0.15 m 的重置高度仍然是硬下限。
        base_height_target = 0.26
        base_height_reward_drop_height = 0.04
        # 对这个任务禁用旧的单侧地面奖励。0.15 m 的硬重置高度仍然
        # 通过 env.min_base_height 单独配置。
        base_height_min = None
        # 左/右髋关节默认值是镜像的（+0.1 / -0.1）。
        # 通过比较它们相对该默认值的偏移，保持步态物理对称。
        hip_symmetry_joint_pairs = (
            ("FL_hip_joint", "FR_hip_joint"),
            ("RL_hip_joint", "RR_hip_joint"),
        )
        # 当 x/y/yaw 命令范数低于该阈值时，视为站立命令。
        zero_command_threshold = 0.1
        # 仅在 sqrt(cmd_vx^2 + cmd_vy^2) <= 0.25 m/s 时计算 roll/pitch
        # 姿态惩罚：零速/慢速时要求机身保持水平；超过 0.25 m/s 后该项为
        # 0，让快速行走自行选择机身姿态。0.25 的单位是 m/s，不是 0.25 秒；
        # 偏航目标 cmd_yaw 不参与这个门控。
        orientation_command_threshold = 0.25
        # 目标带包含重置高度，因此立即应用：
        # 之前 0.5 s 的自由窗口会让零命令策略在收到任何高度反馈前先下蹲到目标带以下。
        base_height_warmup_s = 0.0
        # 低速支撑/载荷整形需要真实的支撑接触。
        foot_contact_force_threshold = 10.0
        low_speed_support_command_threshold = 0.25
        low_speed_support_warmup_s = 0.5
        # 从非零行走命令切换到精确零速命令后，前 1.0 s 暂不计算四足
        # 支撑缺失/载荷均衡项，给策略调整落脚和机身姿态的时间。低速但非零
        # 的命令仍仅使用上面的 reset warmup，不会额外等待。
        low_speed_support_zero_command_delay_s = 1.0
        # 保留原始 1 N 的行走触地事件。下面 0.2 m/s 的情况下，
        # 10 N 阈值可避免轻轻点触 RR 足被 feet_air_time 误判为充分接地。
        feet_air_time_contact_force_threshold = 1.0
        feet_air_time_low_speed_contact_force_threshold = 10.0
        feet_air_time_low_speed_command_threshold = 0.2

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_ang_vel = 1.0
            # 禁用 minich 的组合碰撞奖励；大腿/小腿会分别记录并加权。
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
            # 目标高度函数在 0.26 m 处返回 +1。结合当前 +0.3 的权重和 20 ms
            # 控制步长，其每步增益约为 +0.006/step。
            orientation = -1.0
            base_height = 0.3
            # 在低 x/y 命令下，先要求四个足部都支撑，然后把法向力份额逐渐拉向均匀分布。
            low_speed_missing_support_feet = -1.0
            low_speed_load_balance = -5.0


class MiniChRoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_minich"
