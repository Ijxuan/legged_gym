# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from zipfile import is_zipfile

import isaacgym
import torch
from torch import nn

# Initialize the env registry first; its package import establishes the
# task_registry cycle used by the legacy legged_gym module layout.
from legged_gym.envs import task_registry
from legged_gym.utils.policy_export import (
    attach_periodic_jit_export,
    checkpoint_iteration,
    export_checkpoint_as_jit,
    export_latest_jit,
    latest_model_checkpoint,
    policy_jit_path,
)


class TestPolicyExport(unittest.TestCase):
    def test_periodic_wrapper_exports_only_numbered_intervals(self):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            saved_paths = []
            exported = []
            runner = SimpleNamespace(
                log_dir=str(run_dir),
                env=SimpleNamespace(num_obs=48),
            )

            def original_save(path, infos=None):
                saved_paths.append((Path(path).name, infos))

            def exporter(checkpoint, output, expected_obs_dim):
                exported.append((checkpoint.name, output.name, expected_obs_dim))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                return output

            runner.save = original_save
            attach_periodic_jit_export(runner, 1000, exporter=exporter)
            runner.save(run_dir / "model_999.pt")
            runner.save(run_dir / "model_1000.pt", infos={"tag": "interval"})
            runner.save(run_dir / "model_2000.pt")

            self.assertEqual(
                saved_paths,
                [("model_999.pt", None), ("model_1000.pt", {"tag": "interval"}), ("model_2000.pt", None)],
            )
            self.assertEqual(
                exported,
                [
                    ("model_1000.pt", "policy_iteration_1000.jit", 48),
                    ("model_2000.pt", "policy_iteration_2000.jit", 48),
                ],
            )

    def test_latest_export_uses_the_newest_numeric_checkpoint(self):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "model_50.pt").touch()
            (run_dir / "model_1000.pt").touch()
            (run_dir / "model_latest.pt").touch()
            exported = []
            runner = SimpleNamespace(log_dir=str(run_dir), env=SimpleNamespace(num_obs=48))

            def exporter(checkpoint, output, expected_obs_dim):
                exported.append((checkpoint.name, output.name, expected_obs_dim))
                return output

            self.assertEqual(latest_model_checkpoint(run_dir).name, "model_1000.pt")
            result = export_latest_jit(runner, exporter=exporter)

            self.assertEqual(result, policy_jit_path(run_dir, latest=True))
            self.assertEqual(exported, [("model_1000.pt", "policy_latest.jit", 48)])
            self.assertEqual(checkpoint_iteration(run_dir / "model_1000.pt"), 1000)
            self.assertIsNone(checkpoint_iteration(run_dir / "model_latest.pt"))

    def test_checkpoint_export_runs_in_a_jit_enabled_child_process(self):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint = run_dir / "model_1000.pt"
            actor = nn.Sequential(nn.Linear(48, 8), nn.ELU(), nn.Linear(8, 12))
            torch.save(
                {
                    "model_state_dict": {
                        f"actor.{name}": value for name, value in actor.state_dict().items()
                    },
                    "iter": 1000,
                },
                checkpoint,
            )
            output = policy_jit_path(run_dir, 1000)
            previous_jit_setting = os.environ.get("PYTORCH_JIT")
            os.environ["PYTORCH_JIT"] = "0"
            try:
                result = export_checkpoint_as_jit(checkpoint, output, expected_obs_dim=48)
            finally:
                if previous_jit_setting is None:
                    os.environ.pop("PYTORCH_JIT", None)
                else:
                    os.environ["PYTORCH_JIT"] = previous_jit_setting

            self.assertEqual(result, output.resolve())
            self.assertTrue(output.is_file())
            self.assertTrue(is_zipfile(output))


if __name__ == "__main__":
    unittest.main()
