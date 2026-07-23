# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Safe checkpoint-based TorchScript exports for Mini Cheetah training.

The Isaac training environment deliberately starts with ``PYTORCH_JIT=0``.
Exporting in the training interpreter would therefore fail, so this module
always invokes the existing CPU-only exporter in a fresh child interpreter
with that variable removed.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Optional, Union

from legged_gym import LEGGED_GYM_ROOT_DIR


PathLike = Union[str, os.PathLike]
CheckpointExporter = Callable[[Path, Path, int], Path]
_MODEL_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt")


def checkpoint_iteration(checkpoint_path: PathLike) -> Optional[int]:
    """Return the numeric iteration encoded in ``model_<N>.pt``."""
    match = _MODEL_CHECKPOINT_PATTERN.fullmatch(Path(checkpoint_path).name)
    return int(match.group(1)) if match is not None else None


def latest_model_checkpoint(run_dir: PathLike) -> Path:
    """Find the newest numeric RSL-RL checkpoint in one run directory."""
    run_path = Path(run_dir)
    candidates = [
        checkpoint
        for checkpoint in run_path.glob("model_*.pt")
        if checkpoint_iteration(checkpoint) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No numeric model_*.pt checkpoint found in: {run_path}")
    return max(candidates, key=lambda checkpoint: checkpoint_iteration(checkpoint))


def policy_jit_path(run_dir: PathLike, iteration: Optional[int] = None, *, latest: bool = False) -> Path:
    """Return the stable per-run location for an automatic JIT export."""
    if latest:
        filename = "policy_latest.jit"
    else:
        if iteration is None or iteration <= 0:
            raise ValueError("A positive iteration is required for a numbered JIT export.")
        filename = f"policy_iteration_{iteration}.jit"
    return Path(run_dir) / "exported" / "policies" / filename


def export_checkpoint_as_jit(
    checkpoint_path: PathLike,
    output_path: PathLike,
    expected_obs_dim: int,
) -> Path:
    """Export one complete checkpoint using a CPU-only JIT-enabled child process."""
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    exporter_script = Path(LEGGED_GYM_ROOT_DIR) / "legged_gym" / "scripts" / "export_minich_torchscript.py"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not exporter_script.is_file():
        raise FileNotFoundError(f"TorchScript exporter does not exist: {exporter_script}")

    command = [
        sys.executable,
        str(exporter_script),
        "--checkpoint_file",
        str(checkpoint),
        "--expected_obs_dim",
        str(expected_obs_dim),
        "--output",
        str(output),
        "--overwrite",
    ]
    child_env = os.environ.copy()
    # ``conda run`` sets this only for the trainer process.  A direct child of
    # sys.executable with this variable removed can create TorchScript safely.
    child_env.pop("PYTORCH_JIT", None)

    print(f"Exporting TorchScript: checkpoint={checkpoint.name} -> {output}")
    subprocess.run(command, check=True, env=child_env)
    if not output.is_file():
        raise RuntimeError(f"TorchScript exporter completed without creating: {output}")
    return output


def attach_periodic_jit_export(
    runner,
    interval: int,
    exporter: CheckpointExporter = export_checkpoint_as_jit,
) -> None:
    """Wrap one runner's save method to export each numbered interval checkpoint."""
    interval = int(interval)
    if interval <= 0:
        return
    if runner.log_dir is None:
        raise ValueError("Periodic TorchScript export requires a training log directory.")
    if getattr(runner, "_jit_export_save_wrapped", False):
        raise RuntimeError("Periodic TorchScript export is already attached to this runner.")

    expected_obs_dim = int(runner.env.num_obs)
    original_save = runner.save

    def save_with_jit_export(path, infos=None):
        original_save(path, infos)
        iteration = checkpoint_iteration(path)
        if iteration is not None and iteration > 0 and iteration % interval == 0:
            exporter(
                Path(path),
                policy_jit_path(runner.log_dir, iteration),
                expected_obs_dim,
            )

    runner.save = save_with_jit_export
    runner._jit_export_save_wrapped = True


def export_latest_jit(
    runner,
    exporter: CheckpointExporter = export_checkpoint_as_jit,
) -> Path:
    """Export the most recently completed on-disk checkpoint as ``policy_latest.jit``."""
    if runner.log_dir is None:
        raise ValueError("Latest TorchScript export requires a training log directory.")
    checkpoint = latest_model_checkpoint(runner.log_dir)
    return exporter(
        checkpoint,
        policy_jit_path(runner.log_dir, latest=True),
        int(runner.env.num_obs),
    )
