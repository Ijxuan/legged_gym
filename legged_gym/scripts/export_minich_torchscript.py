#!/usr/bin/env python3
"""Export a Mini Cheetah RSL-RL checkpoint as a CPU TorchScript actor.

This intentionally does *not* create an Isaac Gym environment or a PPO runner.
It reconstructs the feed-forward actor directly from ``model_state_dict`` and
therefore does not allocate GPU/PhysX memory.  The exported file is a policy
only: its input/output contract is checked to be
``[N, --expected_obs_dim] -> [N, 12]``.  The default observation dimension is
235 for the existing rough-terrain policy; pass ``--expected_obs_dim 48`` for
the flat-terrain policy.

The current Mini Cheetah training configuration uses an ELU MLP with hidden
layers 512, 256, 128.  The script infers the layer widths from the checkpoint
and verifies that its expected module layout matches that configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Keep this process CPU-only even when the installed PyTorch build has CUDA.
# This must precede importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rough_minich"
DEFAULT_EXPECTED_OBSERVATIONS = 235
EXPECTED_ACTIONS = 12


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got: {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Mini Cheetah model_*.pt checkpoint as TorchScript without Isaac Gym."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint_file",
        type=Path,
        help="Exact training checkpoint path. Overrides --load_run and --checkpoint.",
    )
    source.add_argument(
        "--load_run",
        "--load-run",
        type=str,
        help="Run directory below --logs_root. Omit to use the newest run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint number, e.g. 4000 for model_4000.pt; -1 selects the newest.",
    )
    parser.add_argument(
        "--logs_root",
        "--logs-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Experiment log root (default: {DEFAULT_LOG_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "TorchScript destination. Default: "
            "<run>/exported/policies/<model_name>.jit. The deployment checkpoint is "
            "not changed automatically."
        ),
    )
    parser.add_argument(
        "--activation",
        choices=("elu", "relu", "selu", "lrelu", "tanh", "sigmoid"),
        default="elu",
        help="Actor activation used during training (MiniChRoughCfgPPO uses elu).",
    )
    parser.add_argument(
        "--expected_obs_dim",
        "--expected-obs-dim",
        dest="expected_obs_dim",
        type=positive_int,
        default=DEFAULT_EXPECTED_OBSERVATIONS,
        metavar="N",
        help=(
            "Expected actor observation width. Defaults to 235 for the rough policy; "
            "use 48 for minich_flat."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file.",
    )
    return parser.parse_args()


def checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def latest_run(logs_root: Path) -> Path:
    runs = sorted(path for path in logs_root.iterdir() if path.is_dir() and path.name != "exported")
    if not runs:
        raise FileNotFoundError(f"No run directories found below: {logs_root}")
    return runs[-1]


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_file is not None:
        path = args.checkpoint_file.expanduser().resolve()
    else:
        logs_root = args.logs_root.expanduser().resolve()
        run_dir = logs_root / args.load_run if args.load_run else latest_run(logs_root)
        if args.checkpoint < 0:
            candidates = sorted(
                (path for path in run_dir.glob("model_*.pt") if checkpoint_number(path) >= 0),
                key=checkpoint_number,
            )
            if not candidates:
                raise FileNotFoundError(f"No model_*.pt checkpoints found in: {run_dir}")
            path = candidates[-1]
        else:
            path = run_dir / f"model_{args.checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def activation_module(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    return activations[name]()


def actor_state_dict(checkpoint: object) -> Dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict containing model_state_dict.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("model_state_dict is not a state-dict.")
    actor_state = {
        # Python 3.8 is used by the Isaac environment, so avoid str.removeprefix.
        key[len("actor."):]: value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise KeyError("No actor.* weights found; this is not a feed-forward RSL-RL checkpoint.")
    if not all(isinstance(value, Tensor) for value in actor_state.values()):
        raise TypeError("Actor state dict contains a non-tensor value.")
    return actor_state


def build_actor(state_dict: Dict[str, Tensor], activation: str) -> Tuple[nn.Sequential, List[int]]:
    indexed_weights = []
    for key, weight in state_dict.items():
        match = re.fullmatch(r"(\d+)\.weight", key)
        if match:
            indexed_weights.append((int(match.group(1)), weight))
    indexed_weights.sort(key=lambda item: item[0])
    if len(indexed_weights) < 2:
        raise ValueError("Expected at least an input and output Linear layer in actor state dict.")

    linear_indices = [index for index, _ in indexed_weights]
    expected_indices = list(range(0, 2 * len(indexed_weights) - 1, 2))
    if linear_indices != expected_indices:
        raise ValueError(
            "Unsupported actor module layout: expected Linear/activation alternation at "
            f"indices {expected_indices}, found {linear_indices}."
        )

    dimensions: List[int] = []
    previous_output = None
    for index, weight in indexed_weights:
        bias = state_dict.get(f"{index}.bias")
        if weight.ndim != 2 or bias is None or bias.ndim != 1:
            raise ValueError(f"Actor layer {index} must have a 2-D weight and 1-D bias.")
        if weight.shape[0] != bias.shape[0]:
            raise ValueError(f"Actor layer {index} weight/bias output dimensions disagree.")
        if previous_output is not None and weight.shape[1] != previous_output:
            raise ValueError(f"Actor layer {index} input does not match the preceding output.")
        if not dimensions:
            dimensions.append(int(weight.shape[1]))
        dimensions.append(int(weight.shape[0]))
        previous_output = int(weight.shape[0])

    modules: List[nn.Module] = []
    for layer_index in range(len(dimensions) - 1):
        modules.append(nn.Linear(dimensions[layer_index], dimensions[layer_index + 1]))
        if layer_index != len(dimensions) - 2:
            modules.append(activation_module(activation))
    actor = nn.Sequential(*modules)
    actor.load_state_dict(state_dict, strict=True)
    actor.eval()
    return actor, dimensions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_jit_available() -> None:
    if os.environ.get("PYTORCH_JIT") == "0":
        raise RuntimeError("PYTORCH_JIT=0 disables TorchScript. Run with: env -u PYTORCH_JIT ...")
    probe = torch.jit.script(nn.Identity())
    if not hasattr(probe, "save"):
        raise RuntimeError("TorchScript is unavailable in this PyTorch runtime.")


def main() -> int:
    args = parse_args()
    check_jit_available()
    checkpoint_path = resolve_checkpoint(args)
    output_path = args.output.expanduser().resolve() if args.output else (
        checkpoint_path.parent / "exported" / "policies" / f"{checkpoint_path.stem}.jit"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {output_path} (pass --overwrite)")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor, dimensions = build_actor(actor_state_dict(checkpoint), args.activation)
    if dimensions[0] != args.expected_obs_dim or dimensions[-1] != EXPECTED_ACTIONS:
        raise ValueError(
            f"Deployment contract mismatch: actor is [N, {dimensions[0]}] -> [N, {dimensions[-1]}], "
            f"expected [N, {args.expected_obs_dim}] -> [N, {EXPECTED_ACTIONS}]."
        )

    with torch.inference_mode():
        example_obs = torch.randn(1, args.expected_obs_dim, dtype=torch.float32)
        eager_actions = actor(example_obs)
        scripted_actor = torch.jit.script(actor)
        scripted_actions = scripted_actor(example_obs)
    torch.testing.assert_close(scripted_actions, eager_actions, rtol=1e-6, atol=1e-6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    scripted_actor.save(str(temporary_path))
    temporary_path.replace(output_path)

    loaded_actor = torch.jit.load(str(output_path), map_location="cpu").eval()
    with torch.inference_mode():
        loaded_actions = loaded_actor(example_obs)
    torch.testing.assert_close(loaded_actions, eager_actions, rtol=1e-6, atol=1e-6)
    if not torch.isfinite(loaded_actions).all():
        raise RuntimeError("TorchScript verification produced non-finite actions.")

    # Older RSL-RL runners store a stale ``iter`` field in intermediate
    # checkpoints, while their model_<N>.pt filename is the user-visible
    # checkpoint number. Prefer that stable filename for export reporting.
    iteration = checkpoint_number(checkpoint_path)
    if iteration < 0:
        iteration = checkpoint.get("iter", "unknown") if isinstance(checkpoint, dict) else "unknown"
    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint_iteration={iteration}")
    print(f"actor_shape=[N, {dimensions[0]}] -> [N, {dimensions[-1]}]")
    print(f"torchscript={output_path}")
    print(f"checkpoint_sha256={sha256(checkpoint_path)}")
    print(f"torchscript_sha256={sha256(output_path)}")
    print("verification=passed (CPU eager actor == reloaded TorchScript actor)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
