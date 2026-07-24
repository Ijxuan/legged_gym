# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic flat-ground viewer for the deployed Mini Cheetah policy.

This is intentionally separate from ``play.py``.  It fixes the environment to
one Mini Cheetah on a plane and removes command, domain, observation, and reset
state randomization so an initial tilt can be inspected in Isaac Gym.
"""

import math
from types import MethodType

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import Logger, get_args, task_registry

INITIAL_BASE_HEIGHT_M = 0.50
# 设置回放开始时的机身高度，单位 m。

# ===== 回放专用基座质量覆盖 =====
# None：不覆盖，使用 URDF 的原始基座刚体质量。
# 填入数值：给 URDF 基座刚体额外增加该质量，单位 kg；不是整机总质量。
# 可设置范围为 [-1.0, 7.0] kg（含端点），与当前 minich_flat 训练时的
# domain_rand.added_mass_range 完全一致。回放会固定使用该值，不做随机采样。
BASE_MASS_ADDED_KG = 7.0

# ===== 回放对象：这几项必须与要看的训练结果匹配 =====
# 注册的平地任务名。它决定使用 plane 地面，以及 48 维本体感觉观测。
TASK_NAME = "minich_flat"
# logs/flat_minich/ 下的训练 run 文件夹名；脚本不会自动选择最新 run。
LOAD_RUN = "Jul23_16-03-31_flat_48obs_2048x4000"
# 上述 run 中加载的 model_<轮数>.pt；改为 -1 才表示该 run 中最新的模型。
CHECKPOINT = 2050
# 策略输入维度。平地策略固定为 48：速度(6)、重力投影(3)、命令(3)、
# 关节位置/速度(24)和上一帧动作(12)。它必须与 checkpoint 的网络输入一致。
OBSERVATION_DIM = 48

# ===== 回放命令时序：零速站立 -> 前进 -> 零速站立 =====
# 中间阶段施加的机身前向速度目标，单位 m/s；横移和偏航目标始终为 0。
FORWARD_SPEED_M_S = 1.5
# 起始零速度站立、前进和末尾零速度站立各自持续时间，单位 s。
STAND_BEFORE_S = 2.0
FORWARD_S = 10
STAND_AFTER_S = 10.0
# 是否允许环境在触发接触、倾角、高度或超时早停后调用 reset。
# False 仅关闭本 viewer 的重置，方便持续观察低蹲/跌倒后的策略输出；不改变训练配置。
ENABLE_TERMINATION_RESET = False
# 是否在回放期间打印机身相对地面的高度。当前关闭，避免高度日志淹没足端诊断输出。
ENABLE_BASE_HEIGHT_PRINT = False
# 仅在 ENABLE_BASE_HEIGHT_PRINT=True 时使用：机身相对地面的高度打印间隔，单位 s。
HEIGHT_PRINT_INTERVAL_S = 0.2
# 左右两侧前、后足端间距的打印间隔，单位 s。每个命令阶段切换时也会额外打印一次。
FOOT_SPACING_PRINT_INTERVAL_S = 0.2

# ===== Isaac Gym 初始相机 =====
# 相机位置 = 初始机身位置 + 此偏移，单位 m；负 x/负 y 表示从左后上方看机器人。
CAMERA_OFFSET = np.array([-2.5, -2.5, 1.5], dtype=np.float64)
# 相机注视点 = 初始机身位置 + 此偏移，稍微抬高以对准躯干而非地面。
CAMERA_LOOKAT_OFFSET = np.array([0.0, 0.0, 0.2], dtype=np.float64)

# 仅用于能耗/接触力输出的四腿显示顺序，不改变 URDF、策略动作或关节映射。
LEG_ORDER = ("FL", "FR", "RL", "RR")
# 三个阶段的内部标签，供统计和重置日志使用，不是额外的策略输入。
PHASE_ORDER = ("stand_before", "forward", "stand_after")

# ===== 回放专用 PD 增益：只手动填写一条模板腿 =====
# 只修改下面这条“模板腿”的 hip / thigh / calf 参数；运行时会原样复制到
# FL、FR、RL、RR 四条腿对应的关节，因此四腿的 Kp、Kd 始终完全一致。
# 这直接覆盖 viewer 环境的 p_gains / d_gains，不会改训练任务配置。
TEMPLATE_LEG_KP = {
    "hip": 17.0,
    "thigh": 17.0,
    "calf": 34.0,
}
TEMPLATE_LEG_KD = {
    "hip": 0.4,
    "thigh": 0.4,
    "calf": 0.5,
}
TEMPLATE_LEG_JOINTS = ("hip", "thigh", "calf")


class JointEnergyMeter:
    """Integrate joint-side mechanical work without changing the simulation.

    Positive work is actuator motoring work (tau * qdot > 0), while negative
    work is braking work.  Absolute work is their sum.  These are mechanical
    joint-side quantities, not an estimate of battery energy: motor/driver
    efficiency and regenerative braking behaviour are intentionally excluded.
    """

    def __init__(self, dof_names, device):
        unexpected = [name for name in dof_names if not any(name.startswith(f"{leg}_") for leg in LEG_ORDER)]
        if unexpected:
            raise ValueError(f"cannot assign Mini Cheetah DOFs to legs: {unexpected}")

        leg_joint_indices = []
        for leg in LEG_ORDER:
            indices = [index for index, name in enumerate(dof_names) if name.startswith(f"{leg}_")]
            if len(indices) != 3:
                raise ValueError(f"expected three actuated joints for {leg}, found {indices} in {dof_names}")
            leg_joint_indices.append(indices)

        self.leg_joint_indices = torch.tensor(leg_joint_indices, device=device, dtype=torch.long)
        self.phase = PHASE_ORDER[0]
        self.positive_work = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.negative_work = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.torque_sq_integral = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.peak_abs_torque = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.phase_duration = {phase: 0.0 for phase in PHASE_ORDER}

    def set_phase(self, phase):
        if phase not in self.positive_work:
            raise ValueError(f"unknown energy-meter phase: {phase}")
        self.phase = phase

    @torch.no_grad()
    def accumulate(self, torques, dof_velocities, sim_dt):
        """Integrate one physics substep for robot 0 using pre-step tau and qdot."""
        joint_power = torques[0] * dof_velocities[0]
        leg_joint_power = joint_power[self.leg_joint_indices]
        self.positive_work[self.phase] += torch.clamp(leg_joint_power, min=0.0).sum(dim=1) * sim_dt
        self.negative_work[self.phase] += torch.clamp(-leg_joint_power, min=0.0).sum(dim=1) * sim_dt
        leg_torques = torques[0][self.leg_joint_indices]
        self.torque_sq_integral[self.phase] += torch.sum(torch.square(leg_torques), dim=1) * sim_dt
        self.peak_abs_torque[self.phase] = torch.maximum(
            self.peak_abs_torque[self.phase], torch.max(torch.abs(leg_torques), dim=1).values
        )
        self.phase_duration[self.phase] += sim_dt

    def phase_values(self, phase):
        positive = self.positive_work[phase].detach().cpu().numpy()
        negative = self.negative_work[phase].detach().cpu().numpy()
        return positive, negative, positive + negative

    def print_summary(self):
        print("\nJoint-side mechanical energy [J] (tau*qdot integrated each physics substep):")
        print("phase         metric       FL       FR       RL       RR    total")
        for phase in PHASE_ORDER:
            positive, negative, absolute = self.phase_values(phase)
            for metric, values in (("motoring +", positive), ("braking  -", negative), ("absolute  ", absolute)):
                print(f"{phase:13s} {metric:10s}" + "".join(f" {value:8.3f}" for value in values) +
                      f" {values.sum():8.3f}")

        total_positive = sum((self.positive_work[phase] for phase in PHASE_ORDER)).detach().cpu().numpy()
        total_negative = sum((self.negative_work[phase] for phase in PHASE_ORDER)).detach().cpu().numpy()
        total_absolute = total_positive + total_negative
        print("total         motoring +" + "".join(f" {value:8.3f}" for value in total_positive) +
              f" {total_positive.sum():8.3f}")
        print("total         braking  -" + "".join(f" {value:8.3f}" for value in total_negative) +
              f" {total_negative.sum():8.3f}")
        print("total         absolute  " + "".join(f" {value:8.3f}" for value in total_absolute) +
              f" {total_absolute.sum():8.3f}")
        if total_absolute.sum() > 0.0:
            shares = total_absolute / total_absolute.sum() * 100.0
            print("absolute-work share [%]:" + "".join(f" {leg}={share:.1f}" for leg, share in zip(LEG_ORDER, shares)))
        print("Note: positive work is the relevant joint-side energy draw; braking work may be dissipated or regenerated by hardware.")

        print("\nJoint torque summary [N*m, (N*m)^2*s] (three joints per leg):")
        print("phase         metric                   FL       FR       RL       RR")
        for phase in PHASE_ORDER:
            duration = self.phase_duration[phase]
            torque_sq = self.torque_sq_integral[phase].detach().cpu().numpy()
            peak_abs = self.peak_abs_torque[phase].detach().cpu().numpy()
            rms_per_joint = np.sqrt(torque_sq / (3.0 * duration)) if duration else torque_sq
            for metric, values in (
                ("RMS per joint [N*m]", rms_per_joint),
                ("peak abs [N*m]", peak_abs),
                ("integral sum(tau^2)", torque_sq),
            ):
                print(f"{phase:13s} {metric:23s}" + "".join(f" {value:8.3f}" for value in values))


def scheduled_phase(step, env_dt):
    """Return the named phase for one step of the fixed inspection sequence."""
    stand_before_steps = int(round(STAND_BEFORE_S / env_dt))
    forward_end_step = stand_before_steps + int(round(FORWARD_S / env_dt))
    if step < stand_before_steps:
        return "stand_before"
    if step < forward_end_step:
        return "forward"
    return "stand_after"


class TerminationMeter:
    """在环境真正执行 reset 前，记录本回放触发早停的精确条件。"""

    def __init__(self, env, resets_enabled):
        self.env = env
        self.resets_enabled = bool(resets_enabled)
        self.phase = PHASE_ORDER[0]
        self.viewer_time_s = 0.0
        self.last_condition_signature = None
        self.phase_counts = {
            phase: {"events": 0, "contact": 0, "tilt": 0, "height": 0, "timeout": 0}
            for phase in PHASE_ORDER
        }

    def set_context(self, step, env_dt, phase):
        if phase not in self.phase_counts:
            raise ValueError(f"unknown termination-meter phase: {phase}")
        self.phase = phase
        self.viewer_time_s = (step + 1) * env_dt

    @torch.no_grad()
    def record_before_reset(self):
        """镜像 LeggedRobot.check_termination 的条件，但不修改 reset_buf。"""
        env = self.env
        device = env.device
        contact_termination = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        max_termination_contact = torch.zeros(env.num_envs, device=device)
        if env.termination_contact_indices.numel() > 0:
            contact_magnitudes = torch.norm(
                env.contact_forces[:, env.termination_contact_indices, :], dim=-1
            )
            contact_termination = torch.any(contact_magnitudes > 1.0, dim=1)
            max_termination_contact = torch.max(contact_magnitudes, dim=1).values

        max_base_tilt = env.cfg.env.max_base_tilt
        base_tilt = torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1.0, 1.0))
        tilt_termination = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        if max_base_tilt is not None:
            tilt_termination = base_tilt > max_base_tilt

        min_base_height = getattr(env.cfg.env, "min_base_height", None)
        base_height = torch.mean(
            env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1
        )
        height_termination = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        if min_base_height is not None:
            height_termination = base_height < min_base_height

        timeout_termination = env.episode_length_buf > env.max_episode_length
        terminated = contact_termination | tilt_termination | height_termination | timeout_termination
        if not bool(torch.any(terminated)):
            self.last_condition_signature = None
            return

        # 本脚本强制单环境，因此一行日志就对应 viewer 中看到的那一只机器人。
        robot_index = 0
        reasons = []
        if bool(contact_termination[robot_index]):
            reasons.append("终止碰撞体接触")
        if bool(tilt_termination[robot_index]):
            reasons.append(f"倾角>{math.degrees(max_base_tilt):.0f}°")
        if bool(height_termination[robot_index]):
            reasons.append(f"机身高度<{min_base_height:.2f}m")
        if bool(timeout_termination[robot_index]):
            reasons.append("episode 超时")
        if not reasons:
            return

        condition_signature = (
            self.phase,
            bool(contact_termination[robot_index]),
            bool(tilt_termination[robot_index]),
            bool(height_termination[robot_index]),
            bool(timeout_termination[robot_index]),
        )
        # 关闭 reset 后，同一早停条件可能持续很多帧；仅在它首次出现，
        # 或原因/阶段变化时打印，避免每个策略步都刷屏。
        if not self.resets_enabled and condition_signature == self.last_condition_signature:
            return
        self.last_condition_signature = condition_signature

        counts = self.phase_counts[self.phase]
        counts["events"] += 1
        counts["contact"] += int(bool(contact_termination[robot_index]))
        counts["tilt"] += int(bool(tilt_termination[robot_index]))
        counts["height"] += int(bool(height_termination[robot_index]))
        counts["timeout"] += int(bool(timeout_termination[robot_index]))
        event_name = "重置" if self.resets_enabled else "早停条件（未重置）"
        print(
            f"[{event_name}] 回放 t={self.viewer_time_s:.2f}s, phase={self.phase}, "
            f"episode={float(env.episode_length_buf[robot_index].item()) * env.dt:.2f}s, "
            f"原因={' + '.join(reasons)}; "
            f"终止体最大接触力={float(max_termination_contact[robot_index]):.1f}N, "
            f"倾角={math.degrees(float(base_tilt[robot_index])):.1f}°, "
            f"机身高度={float(base_height[robot_index]):.3f}m"
        )

    def print_summary(self):
        total = sum(stats["events"] for stats in self.phase_counts.values())
        reset_mode = "reset enabled" if self.resets_enabled else "reset disabled"
        print(f"\nEarly-termination summary ({reset_mode}; conditions sampled before reset handling):")
        if total == 0:
            print("No early-termination condition occurred during this playback.")
            return
        event_column = "resets" if self.resets_enabled else "signals"
        print(f"phase         {event_column:7s}  contact  tilt  low-height  timeout")
        for phase in PHASE_ORDER:
            counts = self.phase_counts[phase]
            print(
                f"{phase:13s} {counts['events']:7d} {counts['contact']:8d} "
                f"{counts['tilt']:5d} {counts['height']:11d} {counts['timeout']:8d}"
            )


def install_termination_meter(env, enable_reset):
    """Wrap only this viewer's termination check, optionally suppressing its resets."""
    meter = TerminationMeter(env, resets_enabled=enable_reset)
    original_check_termination = env.check_termination

    def check_termination_and_record(self):
        meter.record_before_reset()
        if enable_reset:
            return original_check_termination()
        # 此处不能调用原函数：它会写入 reset_buf，随后 post_physics_step
        # 会立刻执行 reset_idx()。同时初始化 timeout 状态，避免终止奖励读取到旧值。
        self.reset_buf.zero_()
        self.time_out_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)

    env.check_termination = MethodType(check_termination_and_record, env)
    return meter


def install_joint_energy_meter(env):
    """Observe every substep torque calculation on this viewer environment only."""
    meter = JointEnergyMeter(env.dof_names, env.device)
    original_compute_torques = env._compute_torques
    sim_dt = env.dt / env.cfg.control.decimation

    def compute_torques_and_measure(self, actions):
        torques = original_compute_torques(actions)
        meter.accumulate(torques, self.dof_vel, sim_dt)
        return torques

    env._compute_torques = MethodType(compute_torques_and_measure, env)
    return meter


class FootContactForceMeter:
    """Summarize direct foot contact forces from Isaac Gym's net-force tensor.

    Isaac Gym supplies a force vector per rigid body, but not a separate contact
    moment/wrench.  The reported impulse is therefore a policy-rate sampled
    estimate: sum(max(Fz, 0) * env.dt), where Fz is world-up normal force.
    """

    def __init__(self, foot_body_names, device, support_force_threshold):
        leg_foot_indices = []
        for leg in LEG_ORDER:
            indices = [index for index, name in enumerate(foot_body_names) if name.startswith(f"{leg}_")]
            if len(indices) != 1:
                raise ValueError(f"expected one foot body for {leg}, found {indices} in {foot_body_names}")
            leg_foot_indices.append(indices[0])

        self.leg_foot_indices = torch.tensor(leg_foot_indices, device=device, dtype=torch.long)
        self.support_force_threshold = float(support_force_threshold)
        self.phase = PHASE_ORDER[0]
        self.sample_duration = {phase: 0.0 for phase in PHASE_ORDER}
        self.samples = {phase: 0 for phase in PHASE_ORDER}
        self.contact_duration = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.normal_impulse = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.tangential_impulse = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.peak_normal_force = {phase: torch.zeros(len(LEG_ORDER), device=device) for phase in PHASE_ORDER}
        self.support_count_sum = {phase: 0.0 for phase in PHASE_ORDER}
        self.four_support_duration = {phase: 0.0 for phase in PHASE_ORDER}
        self.missing_support_integral = {phase: 0.0 for phase in PHASE_ORDER}
        self.full_support_balance_sum = {phase: 0.0 for phase in PHASE_ORDER}
        self.full_support_samples = {phase: 0 for phase in PHASE_ORDER}

    def set_phase(self, phase):
        if phase not in self.sample_duration:
            raise ValueError(f"unknown contact-force-meter phase: {phase}")
        self.phase = phase

    @torch.no_grad()
    def sample(self, foot_contact_forces, sample_dt):
        """Record direct [F_x, F_y, F_z] force samples for robot 0's four feet."""
        leg_forces = foot_contact_forces[self.leg_foot_indices]
        normal_force = torch.clamp(leg_forces[:, 2], min=0.0)
        tangential_force = torch.norm(leg_forces[:, :2], dim=1)
        contact = normal_force > self.support_force_threshold
        support_count = int(contact.sum().item())

        self.sample_duration[self.phase] += sample_dt
        self.samples[self.phase] += 1
        self.contact_duration[self.phase] += contact.float() * sample_dt
        self.normal_impulse[self.phase] += normal_force * sample_dt
        self.tangential_impulse[self.phase] += tangential_force * sample_dt
        self.peak_normal_force[self.phase] = torch.maximum(self.peak_normal_force[self.phase], normal_force)
        self.support_count_sum[self.phase] += support_count
        self.missing_support_integral[self.phase] += (len(LEG_ORDER) - support_count) * sample_dt
        if support_count == len(LEG_ORDER):
            self.four_support_duration[self.phase] += sample_dt
            load_share = normal_force / torch.clamp(normal_force.sum(), min=1e-6)
            self.full_support_balance_sum[self.phase] += float(
                torch.sum(torch.square(load_share - 1.0 / len(LEG_ORDER))).item()
            )
            self.full_support_samples[self.phase] += 1

    @staticmethod
    def _sum_phase_tensors(values):
        return sum((values[phase] for phase in PHASE_ORDER))

    def print_summary(self):
        print(
            "\nDirect foot contact-force summary [N, N*s] "
            f"(net_contact_force_tensor; Fz > {self.support_force_threshold:.1f} N is support):"
        )
        print("phase         metric                  FL       FR       RL       RR    total")
        for phase in PHASE_ORDER:
            duration = self.sample_duration[phase]
            normal_impulse = self.normal_impulse[phase].detach().cpu().numpy()
            tangential_impulse = self.tangential_impulse[phase].detach().cpu().numpy()
            contact_duration = self.contact_duration[phase].detach().cpu().numpy()
            peak_normal = self.peak_normal_force[phase].detach().cpu().numpy()
            mean_normal = normal_impulse / duration if duration else normal_impulse
            mean_tangential = tangential_impulse / duration if duration else tangential_impulse
            impulse_share = normal_impulse / normal_impulse.sum() * 100.0 if normal_impulse.sum() > 0.0 else normal_impulse
            for metric, values in (
                ("Fz>threshold time [s]", contact_duration),
                ("mean normal [N]", mean_normal),
                ("peak normal [N]", peak_normal),
                ("mean tangent [N]", mean_tangential),
                ("normal impulse [Ns]", normal_impulse),
                ("impulse share [%]", impulse_share),
            ):
                total = 100.0 if metric == "impulse share [%]" and normal_impulse.sum() > 0.0 else values.sum()
                print(f"{phase:13s} {metric:22s}" + "".join(f" {value:8.3f}" for value in values) +
                      f" {total:8.3f}")

        total_duration = sum(self.sample_duration.values())
        total_contact_duration = self._sum_phase_tensors(self.contact_duration).detach().cpu().numpy()
        total_normal_impulse = self._sum_phase_tensors(self.normal_impulse).detach().cpu().numpy()
        total_tangential_impulse = self._sum_phase_tensors(self.tangential_impulse).detach().cpu().numpy()
        total_peak_normal = torch.stack([self.peak_normal_force[phase] for phase in PHASE_ORDER]).max(dim=0).values
        total_peak_normal = total_peak_normal.detach().cpu().numpy()
        total_mean_normal = total_normal_impulse / total_duration if total_duration else total_normal_impulse
        total_mean_tangential = total_tangential_impulse / total_duration if total_duration else total_tangential_impulse
        total_impulse_share = total_normal_impulse / total_normal_impulse.sum() * 100.0 if total_normal_impulse.sum() > 0.0 else total_normal_impulse
        for metric, values in (
            ("Fz>threshold time [s]", total_contact_duration),
            ("mean normal [N]", total_mean_normal),
            ("peak normal [N]", total_peak_normal),
            ("mean tangent [N]", total_mean_tangential),
            ("normal impulse [Ns]", total_normal_impulse),
            ("impulse share [%]", total_impulse_share),
        ):
            total = 100.0 if metric == "impulse share [%]" and total_normal_impulse.sum() > 0.0 else values.sum()
            print(f"total         {metric:22s}" + "".join(f" {value:8.3f}" for value in values) +
                  f" {total:8.3f}")

        print("\nFour-foot support diagnostics (direct Fz, independent of command gate):")
        print("phase         mean support  four-foot time  four-foot [%]  mean missing  mean imbalance*")
        for phase in (*PHASE_ORDER, "total"):
            if phase == "total":
                samples = sum(self.samples.values())
                duration = sum(self.sample_duration.values())
                support_count_sum = sum(self.support_count_sum.values())
                four_support_duration = sum(self.four_support_duration.values())
                missing_support_integral = sum(self.missing_support_integral.values())
                balance_sum = sum(self.full_support_balance_sum.values())
                full_samples = sum(self.full_support_samples.values())
            else:
                samples = self.samples[phase]
                duration = self.sample_duration[phase]
                support_count_sum = self.support_count_sum[phase]
                four_support_duration = self.four_support_duration[phase]
                missing_support_integral = self.missing_support_integral[phase]
                balance_sum = self.full_support_balance_sum[phase]
                full_samples = self.full_support_samples[phase]
            mean_support = support_count_sum / samples if samples else 0.0
            four_support_percent = 100.0 * four_support_duration / duration if duration else 0.0
            mean_missing = missing_support_integral / duration if duration else 0.0
            mean_imbalance = balance_sum / full_samples if full_samples else 0.0
            print(
                f"{phase:13s} {mean_support:12.3f} {four_support_duration:16.3f} "
                f"{four_support_percent:14.1f} {mean_missing:13.3f} {mean_imbalance:16.6f}"
            )
        print("* mean sum((Fz_i/sum(Fz) - 0.25)^2), evaluated only when all four feet exceed threshold.")
        print("Note: Isaac Gym's net-contact tensor supplies direct foot force, not a direct foot contact moment/wrench.")


def make_foot_contact_force_meter(env):
    """Bind the direct foot-force meter to the viewer's actual rigid-body order."""
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    foot_body_names = [body_names[index] for index in env.feet_indices.detach().cpu().tolist()]
    return FootContactForceMeter(
        foot_body_names,
        env.device,
        support_force_threshold=env.cfg.rewards.foot_contact_force_threshold,
    )


class OrientationPenaltyMeter:
    """Measure roll/pitch and reproduce the configured orientation reward."""

    def __init__(self, orientation_scale_per_step, command_threshold):
        if command_threshold is None:
            raise ValueError("the flat inspection requires a low-speed orientation threshold")
        self.orientation_scale_per_step = float(orientation_scale_per_step)
        self.command_threshold = float(command_threshold)
        self.phase = PHASE_ORDER[0]
        self.stats = {
            phase: {
                "samples": 0,
                "gated_samples": 0,
                "sum_abs_roll_rad": 0.0,
                "sum_abs_pitch_rad": 0.0,
                "sum_tilt_rad": 0.0,
                "max_abs_roll_rad": 0.0,
                "max_abs_pitch_rad": 0.0,
                "max_tilt_rad": 0.0,
                "sum_orientation_error": 0.0,
                "sum_reward": 0.0,
            }
            for phase in PHASE_ORDER
        }

    def set_phase(self, phase):
        if phase not in self.stats:
            raise ValueError(f"unknown orientation-meter phase: {phase}")
        self.phase = phase

    @torch.no_grad()
    def sample(self, base_quaternion_xyzw, projected_gravity, command_xy):
        """Sample robot 0 after a policy step using the reward's exact inputs."""
        x, y, z, w = (float(value) for value in base_quaternion_xyzw.detach().cpu())
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(pitch_sine)

        gravity = projected_gravity.detach().cpu()
        orientation_error = float(torch.sum(torch.square(gravity[:2])))
        tilt = math.acos(max(-1.0, min(1.0, -float(gravity[2]))))
        low_speed = float(torch.norm(command_xy.detach()).cpu()) <= self.command_threshold
        reward = self.orientation_scale_per_step * orientation_error if low_speed else 0.0

        stats = self.stats[self.phase]
        stats["samples"] += 1
        stats["gated_samples"] += int(low_speed)
        stats["sum_abs_roll_rad"] += abs(roll)
        stats["sum_abs_pitch_rad"] += abs(pitch)
        stats["sum_tilt_rad"] += tilt
        stats["max_abs_roll_rad"] = max(stats["max_abs_roll_rad"], abs(roll))
        stats["max_abs_pitch_rad"] = max(stats["max_abs_pitch_rad"], abs(pitch))
        stats["max_tilt_rad"] = max(stats["max_tilt_rad"], tilt)
        stats["sum_orientation_error"] += orientation_error
        stats["sum_reward"] += reward

    def print_summary(self):
        print(
            "\nBase attitude and low-speed orientation penalty "
            f"(scale per policy step={self.orientation_scale_per_step:.5f}, "
            f"active when ||cmd_xy|| <= {self.command_threshold:.2f} m/s):"
        )
        print(
            "phase         gated  mean|roll| max|roll| mean|pitch| max|pitch| "
            "mean tilt max tilt mean error  mean penalty  total penalty"
        )
        for phase in PHASE_ORDER:
            stats = self.stats[phase]
            samples = stats["samples"]
            if samples == 0:
                continue
            degrees = 180.0 / math.pi
            print(
                f"{phase:13s} {stats['gated_samples']:3d}/{samples:<3d} "
                f"{stats['sum_abs_roll_rad'] / samples * degrees:10.3f} "
                f"{stats['max_abs_roll_rad'] * degrees:9.3f} "
                f"{stats['sum_abs_pitch_rad'] / samples * degrees:11.3f} "
                f"{stats['max_abs_pitch_rad'] * degrees:10.3f} "
                f"{stats['sum_tilt_rad'] / samples * degrees:9.3f} "
                f"{stats['max_tilt_rad'] * degrees:8.3f} "
                f"{stats['sum_orientation_error'] / samples:10.6f} "
                f"{stats['sum_reward'] / samples:12.8f} "
                f"{stats['sum_reward']:13.8f}"
            )


@torch.no_grad()
def base_height_above_terrain(env, robot_index=0):
    """Return the same terrain-relative base height used by the reward and reset logic."""
    measured_heights = env.measured_heights
    if torch.is_tensor(measured_heights):
        if measured_heights.ndim == 0:
            terrain_height = float(measured_heights.item())
        else:
            terrain_height = float(torch.mean(measured_heights[robot_index]).item())
    else:
        terrain_height = float(measured_heights)
    return float(env.root_states[robot_index, 2].item()) - terrain_height


class BaseHeightPrinter:
    """Print the viewer's terrain-relative base height at a bounded rate."""

    def __init__(self, env, interval_s):
        if interval_s <= 0.0:
            raise ValueError("HEIGHT_PRINT_INTERVAL_S must be positive")
        self.interval_steps = max(1, int(round(interval_s / env.dt)))
        self.height_target = float(env.cfg.rewards.base_height_target)
        self.height_floor = getattr(env.cfg.rewards, "base_height_min", None)
        self.height_reward_drop_height = getattr(
            env.cfg.rewards, "base_height_reward_drop_height", None
        )
        self.min_height = getattr(env.cfg.env, "min_base_height", None)
        self.last_phase = None

    def _print(self, env, time_s, phase):
        base_height = base_height_above_terrain(env)
        if self.height_reward_drop_height is not None:
            low_zero_reward_height = self.height_target - self.height_reward_drop_height
            high_zero_reward_height = self.height_target + self.height_reward_drop_height
            objective = (
                f", 目标={self.height_target:.3f} m"
                f", 偏至{low_zero_reward_height:.3f}/{high_zero_reward_height:.3f} m降至0"
            )
        elif self.height_floor is not None:
            objective = f", 高度惩罚下限<{self.height_floor:.3f} m"
        else:
            objective = f", 目标={self.height_target:.3f} m"
        threshold = "" if self.min_height is None else f", 早停阈值<{self.min_height:.3f} m"
        print(
            f"[高度] t={time_s:.2f}s, phase={phase}, "
            f"机身高度={base_height:.3f} m{objective}{threshold}"
        )

    def print_initial(self, env):
        self._print(env, time_s=0.0, phase="initial")

    def sample(self, env, step, phase):
        phase_changed = phase != self.last_phase
        if phase_changed or step % self.interval_steps == 0:
            self._print(env, time_s=(step + 1) * env.dt, phase=phase)
        self.last_phase = phase


class FootSpacingPrinter:
    """Print the same-side front/rear foot spacing from actual rigid-body poses.

    The four ``*_foot`` rigid bodies are the Mini Cheetah foot collision
    spheres, so their origins are the foot-end centers visible in the viewer.
    ``env.feet_indices`` is not assumed to have a particular leg order.
    """

    def __init__(self, env, interval_s):
        if interval_s <= 0.0:
            raise ValueError("FOOT_SPACING_PRINT_INTERVAL_S must be positive")
        self.interval_steps = max(1, int(round(interval_s / env.dt)))
        body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        foot_body_indices = env.feet_indices.detach().cpu().tolist()
        leg_body_indices = []
        for leg in LEG_ORDER:
            matches = [
                int(body_index)
                for body_index in foot_body_indices
                if body_names[int(body_index)].startswith(f"{leg}_")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one foot rigid body for {leg}, found {matches}; "
                    f"available feet={[body_names[int(index)] for index in foot_body_indices]}"
                )
            leg_body_indices.append(matches[0])

        # LeggedRobot refreshes root, DOF and contact tensors in env.step(),
        # but not this tensor.  Keep one wrapped view and refresh it before
        # every read so these coordinates belong to the just-finished step.
        rigid_body_state_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            env.num_envs, -1, 13
        )
        if self.rigid_body_states.shape[1] != len(body_names):
            raise RuntimeError(
                "rigid-body state tensor shape does not match the actor rigid-body names: "
                f"{self.rigid_body_states.shape[1]} vs {len(body_names)}"
            )
        self.leg_body_indices = torch.tensor(leg_body_indices, device=env.device, dtype=torch.long)
        self.last_phase = None

    @torch.no_grad()
    def _print(self, env, time_s, phase):
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        feet_xyz = self.rigid_body_states[0, self.leg_body_indices, :3]
        left_delta = feet_xyz[0] - feet_xyz[2]   # FL - RL
        right_delta = feet_xyz[1] - feet_xyz[3]  # FR - RR

        def describe(delta):
            absolute = torch.abs(delta)
            return (
                # f"3D={torch.norm(delta).item():.3f} m, "
                f"XY={torch.norm(delta[:2]).item():.3f} m, "
                f"|dx|={absolute[0].item():.3f}, "
                # f"|dy|={absolute[1].item():.3f}, "
                # f"|dz|={absolute[2].item():.3f} m"
            )

        print(
            f"[足端间距] t={time_s:.2f}s, phase={phase}; "
            f"左 FL-RL: {describe(left_delta)}; "
            f"右 FR-RR: {describe(right_delta)}"
        )

    def print_initial(self, env):
        self._print(env, time_s=0.0, phase="initial")

    def sample(self, env, step, phase):
        phase_changed = phase != self.last_phase
        # t=0 is already printed by print_initial(), so start periodic output
        # at the next interval rather than printing an almost identical line.
        if step > 0 and (phase_changed or step % self.interval_steps == 0):
            self._print(env, time_s=(step + 1) * env.dt, phase=phase)
        self.last_phase = phase


class PhaseRewardMeter:
    """Accumulate exact training reward contributions for each viewer phase.

    This meter observes ``episode_sums`` deltas inside ``compute_reward``. It
    therefore reuses each reward calculation exactly once and does not disturb
    stateful functions such as ``_reward_feet_air_time``.
    """

    def __init__(self, env):
        self.reward_names = tuple(env.episode_sums.keys())
        if not self.reward_names:
            raise RuntimeError("the environment has no enabled reward terms to measure")
        self.config_scales = np.asarray(
            [float(env.reward_scales[name] / env.dt) for name in self.reward_names], dtype=np.float64
        )
        self.termination_index = self.reward_names.index("termination") \
            if "termination" in self.reward_names else None
        self.only_positive_rewards = bool(env.cfg.rewards.only_positive_rewards)
        self.phase = PHASE_ORDER[0]
        self.stats = {
            phase: {
                "steps": 0,
                "term_sums": np.zeros(len(self.reward_names), dtype=np.float64),
                "raw_nontermination_sum": 0.0,
                "clipped_nontermination_sum": 0.0,
                "returned_total_sum": 0.0,
            }
            for phase in PHASE_ORDER
        }

    def set_phase(self, phase):
        if phase not in self.stats:
            raise ValueError(f"unknown reward-meter phase: {phase}")
        self.phase = phase

    def sample(self, term_rewards, returned_reward):
        """Record one policy step after LeggedRobot.compute_reward has run."""
        values = torch.cat((term_rewards, returned_reward.reshape(1))).detach().cpu().numpy()
        term_values = values[:-1].astype(np.float64, copy=False)
        returned_value = float(values[-1])
        termination_value = 0.0 if self.termination_index is None else float(term_values[self.termination_index])
        raw_nontermination = float(term_values.sum() - termination_value)
        clipped_nontermination = max(raw_nontermination, 0.0) if self.only_positive_rewards else raw_nontermination
        expected_return = clipped_nontermination + termination_value
        if not np.isclose(returned_value, expected_return, rtol=1e-5, atol=2e-6):
            raise RuntimeError(
                "phase reward capture disagrees with LeggedRobot.compute_reward: "
                f"returned={returned_value:.8f}, expected={expected_return:.8f}"
            )

        stats = self.stats[self.phase]
        stats["steps"] += 1
        stats["term_sums"] += term_values
        stats["raw_nontermination_sum"] += raw_nontermination
        stats["clipped_nontermination_sum"] += clipped_nontermination
        stats["returned_total_sum"] += returned_value

    def print_summary(self):
        print(
            "\nPhase reward summary (exact LeggedRobot.compute_reward contributions; "
            "term totals already include the policy-step dt factor):"
        )
        print(
            "phase         steps  raw nonterm  clipped nonterm  returned total  "
            "mean return/step"
        )
        for phase in PHASE_ORDER:
            stats = self.stats[phase]
            steps = stats["steps"]
            if steps == 0:
                continue
            print(
                f"{phase:13s} {steps:5d} {stats['raw_nontermination_sum']:12.6f} "
                f"{stats['clipped_nontermination_sum']:16.6f} "
                f"{stats['returned_total_sum']:15.6f} "
                f"{stats['returned_total_sum'] / steps:17.8f}"
            )

        print("\nphase         reward term                         cfg scale        total  mean/step")
        for phase in PHASE_ORDER:
            stats = self.stats[phase]
            steps = stats["steps"]
            if steps == 0:
                continue
            for name, config_scale, term_sum in zip(
                self.reward_names, self.config_scales, stats["term_sums"]
            ):
                print(
                    f"{phase:13s} {name:35s} {config_scale:10.3g} "
                    f"{term_sum:12.6f} {term_sum / steps:10.8f}"
                )

    def validate_step_count(self, expected_steps):
        measured_steps = sum(stats["steps"] for stats in self.stats.values())
        if measured_steps != expected_steps:
            raise RuntimeError(
                f"phase reward meter captured {measured_steps} steps, expected {expected_steps}; "
                "a non-playback environment step was included"
            )


def install_phase_reward_meter(env):
    """Wrap this viewer's compute_reward without evaluating any reward function twice."""
    meter = PhaseRewardMeter(env)
    original_compute_reward = env.compute_reward

    def compute_reward_and_measure(self):
        before_episode_sums = torch.stack(
            [self.episode_sums[name][0] for name in meter.reward_names]
        ).clone()
        original_compute_reward()
        after_episode_sums = torch.stack(
            [self.episode_sums[name][0] for name in meter.reward_names]
        )
        meter.sample(after_episode_sums - before_episode_sums, self.rew_buf[0])

    env.compute_reward = MethodType(compute_reward_and_measure, env)
    return meter


class BaseHeightAndSupportMeter:
    """Report the exact raw and scaled height/support rewards in playback."""

    def __init__(self, env):
        self.base_height_scale_per_step = float(env.reward_scales["base_height"])
        self.missing_support_scale_per_step = float(env.reward_scales["low_speed_missing_support_feet"])
        self.load_balance_scale_per_step = float(env.reward_scales["low_speed_load_balance"])
        self.height_target = float(env.cfg.rewards.base_height_target)
        self.height_floor = getattr(env.cfg.rewards, "base_height_min", None)
        self.height_reward_drop_height = getattr(
            env.cfg.rewards, "base_height_reward_drop_height", None
        )
        self.height_warmup_s = float(env.cfg.rewards.base_height_warmup_s)
        self.support_command_threshold = float(env.cfg.rewards.low_speed_support_command_threshold)
        self.support_warmup_s = float(env.cfg.rewards.low_speed_support_warmup_s)
        self.zero_command_threshold = float(env.cfg.rewards.zero_command_threshold)
        self.zero_command_delay_s = float(getattr(
            env.cfg.rewards, "low_speed_support_zero_command_delay_s", 0.0
        ))
        self.phase = PHASE_ORDER[0]
        self.stats = {
            phase: {
                "samples": 0,
                "height_active_samples": 0,
                "support_active_samples": 0,
                "sum_height_m": 0.0,
                "sum_height_error_m": 0.0,
                "sum_height_raw": 0.0,
                "sum_height_reward": 0.0,
                "sum_missing_support": 0.0,
                "sum_missing_reward": 0.0,
                "sum_load_balance": 0.0,
                "sum_load_balance_reward": 0.0,
            }
            for phase in PHASE_ORDER
        }

    def set_phase(self, phase):
        if phase not in self.stats:
            raise ValueError(f"unknown constraint-meter phase: {phase}")
        self.phase = phase

    @torch.no_grad()
    def sample(self, env):
        """Sample robot 0 with the same raw functions and scaled values as training."""
        robot_index = 0
        episode_seconds = float(env.episode_length_buf[robot_index].item()) * env.dt
        command_speed = float(torch.norm(env.commands[robot_index, :2]).item())
        exact_zero_command = bool(
            torch.norm(env.commands[robot_index, :3]).item() < self.zero_command_threshold
        )
        zero_command_elapsed = float(env.zero_command_elapsed[robot_index].item())
        raw_height = float(env._reward_base_height()[robot_index].item())
        raw_missing_support = float(env._reward_low_speed_missing_support_feet()[robot_index].item())
        raw_load_balance = float(env._reward_low_speed_load_balance()[robot_index].item())

        measured_heights = env.measured_heights
        if torch.is_tensor(measured_heights):
            terrain_height = float(torch.mean(measured_heights[robot_index]).item()) if measured_heights.ndim else float(measured_heights.item())
        else:
            terrain_height = float(measured_heights)
        base_height = float(env.root_states[robot_index, 2].item()) - terrain_height

        stats = self.stats[self.phase]
        stats["samples"] += 1
        stats["height_active_samples"] += int(episode_seconds >= self.height_warmup_s)
        support_active = (
            episode_seconds >= self.support_warmup_s
            and command_speed <= self.support_command_threshold
            and (not exact_zero_command or zero_command_elapsed >= self.zero_command_delay_s)
        )
        stats["support_active_samples"] += int(support_active)
        stats["sum_height_m"] += base_height
        if self.height_reward_drop_height is not None:
            height_error_m = abs(base_height - self.height_target)
        elif self.height_floor is not None:
            height_error_m = max(0.0, self.height_floor - base_height)
        else:
            height_error_m = base_height - self.height_target
        stats["sum_height_error_m"] += height_error_m
        stats["sum_height_raw"] += raw_height
        stats["sum_height_reward"] += self.base_height_scale_per_step * raw_height
        stats["sum_missing_support"] += raw_missing_support
        stats["sum_missing_reward"] += self.missing_support_scale_per_step * raw_missing_support
        stats["sum_load_balance"] += raw_load_balance
        stats["sum_load_balance_reward"] += self.load_balance_scale_per_step * raw_load_balance

    def print_summary(self):
        if self.height_reward_drop_height is not None:
            low_zero_reward_height = self.height_target - self.height_reward_drop_height
            high_zero_reward_height = self.height_target + self.height_reward_drop_height
            height_objective = (
                f"height target={self.height_target:.3f} m, zero reward at "
                f"{low_zero_reward_height:.3f}/{high_zero_reward_height:.3f} m"
            )
            height_error_label = "mean deviation"
        elif self.height_floor is not None:
            height_objective = f"height floor={self.height_floor:.3f} m"
            height_error_label = "mean deficit"
        else:
            height_objective = f"height target={self.height_target:.3f} m"
            height_error_label = "mean error"
        print(
            "\nBase-height and low-speed support rewards "
            f"({height_objective}; support active when "
            f"||cmd_xy|| <= {self.support_command_threshold:.2f} m/s; "
            f"exact zero waits {self.zero_command_delay_s:.1f} s):"
        )
        print(
            f"phase         height gate mean height  {height_error_label:11s}  height reward   support gate "
            "mean missing  missing penalty  mean balance  balance penalty"
        )
        for phase in PHASE_ORDER:
            stats = self.stats[phase]
            samples = stats["samples"]
            if samples == 0:
                continue
            print(
                f"{phase:13s} {stats['height_active_samples']:3d}/{samples:<3d} "
                f"{stats['sum_height_m'] / samples:11.4f} "
                f"{stats['sum_height_error_m'] / samples:11.4f} "
                f"{stats['sum_height_reward']:14.8f} "
                f"{stats['support_active_samples']:3d}/{samples:<3d} "
                f"{stats['sum_missing_support'] / samples:12.4f} "
                f"{stats['sum_missing_reward']:15.8f} "
                f"{stats['sum_load_balance'] / samples:12.6f} "
                f"{stats['sum_load_balance_reward']:15.8f}"
            )


def total_play_steps(env_dt):
    return int(round((STAND_BEFORE_S + FORWARD_S + STAND_AFTER_S) / env_dt))


def configure_flat_sequence_env(env_cfg):
    """Apply only viewer-local overrides; the registered training cfg is untouched."""
    env_cfg.env.num_envs = 1
    env_cfg.init_state.pos[2] = INITIAL_BASE_HEIGHT_M
    # This policy is the deployment-oriented 48-D flat task: no 187-point
    # height grid may be initialized or appended to its observation.
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_kwargs = None

    # Disable all configurable observation, actuator, domain, and morphology randomization.
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_action_delay = False
    # 腿长变体在环境创建时就会随机抽样；即使只有一个环境也必须显式关闭，
    # 否则“固定初始状态”的回放仍可能每次加载不同形态。
    env_cfg.domain_rand.randomize_leg_lengths = False

    # 训练用的是“原始基座质量 + U[-1.0, 7.0] kg”。若指定回放覆盖值，
    # 则沿用同一实现，但把采样范围收窄为 [value, value]，保证可重复。
    if BASE_MASS_ADDED_KG is not None:
        added_mass_kg = float(BASE_MASS_ADDED_KG)
        training_min_kg, training_max_kg = (
            float(value) for value in env_cfg.domain_rand.added_mass_range
        )
        if not math.isfinite(added_mass_kg):
            raise ValueError("BASE_MASS_ADDED_KG must be a finite number or None")
        if not training_min_kg <= added_mass_kg <= training_max_kg:
            raise ValueError(
                "BASE_MASS_ADDED_KG must stay within the minich_flat training range "
                f"[{training_min_kg:.1f}, {training_max_kg:.1f}] kg, got {added_mass_kg:.3f} kg"
            )
        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.added_mass_range = [added_mass_kg, added_mass_kg]

    # Commands are written directly by the three-phase viewer schedule below.
    # Prevent the environment's periodic resampling from changing them first.
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_command_probability = 0.0
    env_cfg.commands.resampling_time = STAND_BEFORE_S + FORWARD_S + STAND_AFTER_S + 1.0
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]


def set_command(env, forward_speed):
    """Write a forward-only target: vx is requested speed; vy and yaw are zero."""
    env.commands.zero_()
    env.commands[:, 0] = forward_speed


def scheduled_forward_speed(step, env_dt):
    """Return zero -> 1 m/s -> zero command for one inspection sequence."""
    stand_before_steps = int(round(STAND_BEFORE_S / env_dt))
    forward_end_step = stand_before_steps + int(round(FORWARD_S / env_dt))
    if stand_before_steps <= step < forward_end_step:
        return FORWARD_SPEED_M_S
    return 0.0


def aim_camera_at_robot(env):
    """Set the initial viewer shot relative to robot 0, looking at its trunk."""
    if env.viewer is None:
        return
    base_position = env.root_states[0, :3].detach().cpu().numpy()
    env.set_camera(base_position + CAMERA_OFFSET, base_position + CAMERA_LOOKAT_OFFSET)


def print_base_mass_setting(env):
    """Report the same root-link mass that LeggedRobot randomizes at creation."""
    rigid_body_props = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.actor_handles[0]
    )
    base_body_name = env.gym.get_actor_rigid_body_names(
        env.envs[0], env.actor_handles[0]
    )[0]
    base_mass_kg = float(rigid_body_props[0].mass)
    if BASE_MASS_ADDED_KG is None:
        print(
            f"Viewer base mass: {base_mass_kg:.3f} kg "
            f"({base_body_name}, URDF nominal; no override)"
        )
    else:
        print(
            f"Viewer base mass: {base_mass_kg:.3f} kg "
            f"({base_body_name}, URDF + {float(BASE_MASS_ADDED_KG):.3f} kg; fixed replay override)"
        )


def install_fixed_reset(env):
    """Replace the base task's randomized reset methods on this viewer instance."""
    def reset_dofs_fixed(self, env_ids):
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids),
        )

    def reset_root_states_fixed(self, env_ids):
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids),
        )

    env._reset_dofs = MethodType(reset_dofs_fixed, env)
    env._reset_root_states = MethodType(reset_root_states_fixed, env)


def apply_shared_leg_pd_gains(env):
    """Copy one template leg's three PD pairs to the corresponding joints of all legs."""
    if set(TEMPLATE_LEG_KP) != set(TEMPLATE_LEG_JOINTS):
        raise ValueError("TEMPLATE_LEG_KP must define hip, thigh, and calf exactly once")
    if set(TEMPLATE_LEG_KD) != set(TEMPLATE_LEG_JOINTS):
        raise ValueError("TEMPLATE_LEG_KD must define hip, thigh, and calf exactly once")

    dof_indices = {name: index for index, name in enumerate(env.dof_names)}
    expected_joint_names = [
        f"{leg}_{joint}_joint" for leg in LEG_ORDER for joint in TEMPLATE_LEG_JOINTS
    ]
    missing_joint_names = [name for name in expected_joint_names if name not in dof_indices]
    unexpected_joint_names = [name for name in env.dof_names if name not in expected_joint_names]
    if missing_joint_names or unexpected_joint_names:
        raise ValueError(
            "cannot apply symmetric PD gains; "
            f"missing={missing_joint_names}, unexpected={unexpected_joint_names}"
        )

    template_kp = torch.tensor(
        [TEMPLATE_LEG_KP[joint] for joint in TEMPLATE_LEG_JOINTS], device=env.device
    )
    template_kd = torch.tensor(
        [TEMPLATE_LEG_KD[joint] for joint in TEMPLATE_LEG_JOINTS], device=env.device
    )
    for leg in LEG_ORDER:
        indices = torch.tensor(
            [dof_indices[f"{leg}_{joint}_joint"] for joint in TEMPLATE_LEG_JOINTS],
            dtype=torch.long,
            device=env.device,
        )
        env.p_gains[indices] = template_kp
        env.d_gains[indices] = template_kd
        if not torch.equal(env.p_gains[indices], template_kp) or not torch.equal(env.d_gains[indices], template_kd):
            raise RuntimeError(f"failed to copy template PD gains to {leg}")

    print(
        "Viewer PD gains copied to all four legs: "
        + ", ".join(
            f"{joint}(Kp={TEMPLATE_LEG_KP[joint]:g}, Kd={TEMPLATE_LEG_KD[joint]:g})"
            for joint in TEMPLATE_LEG_JOINTS
        )
    )


def reset_to_fixed_state(env):
    """Apply the fixed reset once after construction, then rebuild its observation."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    set_command(env, 0.0)
    env.actions.zero_()
    env.last_actions.zero_()
    env.last_dof_vel.zero_()
    env.last_root_vel.zero_()
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel.zero_()
    env.base_ang_vel.zero_()
    env.projected_gravity[:] = env.gravity_vec
    env.compute_observations()
    return env.get_observations()


def validate_fixed_start(env, obs):
    """Fail early instead of inspecting an accidental randomized rollout."""
    if tuple(obs.shape) != (1, OBSERVATION_DIM):
        raise RuntimeError(f"expected observation shape (1, {OBSERVATION_DIM}), got {tuple(obs.shape)}")
    if tuple(env.commands.shape) != (1, 3):
        raise RuntimeError(f"expected three direct velocity commands, got {tuple(env.commands.shape)}")
    if env.cfg.terrain.measure_heights:
        raise RuntimeError("48-D flat policy must not initialize terrain-height observations")
    if torch.count_nonzero(env.commands).item() != 0:
        raise RuntimeError("initial stand phase received a non-zero command")
    if not torch.allclose(env.dof_pos, env.default_dof_pos.expand_as(env.dof_pos)):
        raise RuntimeError("fixed reset did not restore the default joint positions")
    expected_root = env.base_init_state.expand(env.num_envs, -1).clone()
    expected_root[:, :3] += env.env_origins
    if not torch.allclose(env.root_states[:, :7], expected_root[:, :7]):
        raise RuntimeError("fixed reset did not restore the default base pose")
    if torch.count_nonzero(env.dof_vel).item() != 0 or torch.count_nonzero(env.root_states[:, 7:13]).item() != 0:
        raise RuntimeError("fixed reset did not clear joint or base velocity")


def play(args):
    # This viewer is deliberately bound to the Mini Cheetah policy under test.
    args.task = TASK_NAME
    args.load_run = None
    args.checkpoint = None
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_flat_sequence_env(env_cfg)

    # Do not let command-line checkpoint flags accidentally select a different
    # policy than the model that was exported for the current deployment test.
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = LOAD_RUN
    train_cfg.runner.checkpoint = CHECKPOINT

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    install_fixed_reset(env)
    print_base_mass_setting(env)

    # OnPolicyRunner 构造时会调用 env.reset() 并额外执行一帧零动作。先让它
    # 完成模型加载，再恢复固定初始状态并安装统计器，避免那一帧混入起始站立阶段。
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name=TASK_NAME, args=args, train_cfg=train_cfg
    )
    obs = reset_to_fixed_state(env)
    apply_shared_leg_pd_gains(env)
    validate_fixed_start(env, obs)
    aim_camera_at_robot(env)
    termination_meter = install_termination_meter(env, enable_reset=ENABLE_TERMINATION_RESET)
    reward_meter = install_phase_reward_meter(env)
    energy_meter = install_joint_energy_meter(env)
    contact_force_meter = make_foot_contact_force_meter(env)
    orientation_meter = OrientationPenaltyMeter(
        orientation_scale_per_step=env.reward_scales["orientation"],
        command_threshold=env.cfg.rewards.orientation_command_threshold,
    )
    constraint_meter = BaseHeightAndSupportMeter(env)
    height_printer = (
        BaseHeightPrinter(env, HEIGHT_PRINT_INTERVAL_S)
        if ENABLE_BASE_HEIGHT_PRINT else None
    )
    foot_spacing_printer = FootSpacingPrinter(env, FOOT_SPACING_PRINT_INTERVAL_S)
    play_steps = total_play_steps(env.dt)
    print(
        f"Flat command sequence viewer: task={TASK_NAME}, run={LOAD_RUN}, "
        f"checkpoint={CHECKPOINT}, stand={STAND_BEFORE_S:.1f}s -> "
        f"vx={FORWARD_SPEED_M_S:.1f} m/s for {FORWARD_S:.1f}s -> "
        f"stand={STAND_AFTER_S:.1f}s, obs_shape={tuple(obs.shape)}, "
        f"termination_reset={'on' if ENABLE_TERMINATION_RESET else 'off'}"
    )
    if height_printer is not None:
        height_printer.print_initial(env)
    foot_spacing_printer.print_initial(env)
    policy = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = play_steps - 1
    stop_rew_log = env.max_episode_length + 1

    for i in range(play_steps):
        # Rebuild the observation after each phase change so policy input and
        # command tensor always agree, including after a termination reset.
        phase = scheduled_phase(i, env.dt)
        energy_meter.set_phase(phase)
        contact_force_meter.set_phase(phase)
        orientation_meter.set_phase(phase)
        constraint_meter.set_phase(phase)
        reward_meter.set_phase(phase)
        termination_meter.set_context(i, env.dt, phase)
        set_command(env, scheduled_forward_speed(i, env.dt))
        env.compute_observations()
        obs = env.get_observations()
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if not ENABLE_TERMINATION_RESET and torch.any(dones):
            raise RuntimeError("termination reset is disabled, but env.step still returned done")
        foot_spacing_printer.sample(env, i, phase)
        contact_force_meter.sample(env.contact_forces[robot_index, env.feet_indices, :], env.dt)
        orientation_meter.sample(
            env.root_states[robot_index, 3:7],
            env.projected_gravity[robot_index],
            env.commands[robot_index, :2],
        )
        constraint_meter.sample(env)
        if height_printer is not None:
            height_printer.sample(env, i, phase)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log and infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    reward_meter.validate_step_count(play_steps)
    energy_meter.print_summary()
    contact_force_meter.print_summary()
    orientation_meter.print_summary()
    constraint_meter.print_summary()
    reward_meter.print_summary()
    termination_meter.print_summary()


if __name__ == "__main__":
    args = get_args()
    play(args)
