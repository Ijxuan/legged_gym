# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import signal

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.policy_export import attach_periodic_jit_export, export_latest_jit


class TrainingInterrupted(KeyboardInterrupt):
    """Raised at a safe Python boundary after SIGINT or SIGTERM."""


def _raise_training_interrupted(signum, _frame):
    signal_name = signal.Signals(signum).name
    print(f"Received {signal_name}; exporting the latest complete checkpoint before exit.")
    raise TrainingInterrupted(signal_name)


def _install_exit_signal_handlers():
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_training_interrupted)
    return previous_handlers


def _restore_signal_handlers(previous_handlers):
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    jit_export_interval = train_cfg.runner.jit_export_interval
    jit_export_on_exit = train_cfg.runner.jit_export_on_exit
    if jit_export_interval > 0:
        if train_cfg.runner.save_interval <= 0 or jit_export_interval % train_cfg.runner.save_interval != 0:
            raise ValueError(
                "jit_export_interval must be a positive multiple of save_interval so an exact checkpoint exists."
            )
        attach_periodic_jit_export(ppo_runner, jit_export_interval)
        print(f"Automatic TorchScript export enabled every {jit_export_interval} iterations.")

    previous_handlers = _install_exit_signal_handlers() if jit_export_on_exit else {}
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except BaseException:
        if jit_export_on_exit:
            try:
                export_latest_jit(ppo_runner)
            except FileNotFoundError:
                print("No completed model checkpoint exists yet; skipping policy_latest.jit export.")
            except Exception as error:
                print(f"ERROR: failed to export policy_latest.jit during exit: {error}")
        raise
    else:
        if jit_export_on_exit:
            export_latest_jit(ppo_runner)
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == '__main__':
    args = get_args()
    train(args)
