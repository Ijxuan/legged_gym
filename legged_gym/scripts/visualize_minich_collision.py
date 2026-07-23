#!/usr/bin/env python3
"""Visualize the Mini Cheetah with the calf collision mesh shown as red geometry.

The normal URDF keeps the original visual mesh.  For this inspection script we
create a temporary URDF beside the real one and replace only the calf visual
mesh with the collision STL.  This makes the collision model visible in the
Isaac Gym viewer without changing the permanent visual model.
"""

import argparse
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import isaacgym  # noqa: F401: Isaac Gym must be imported before legged_gym.

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403: registers the minich task.
from legged_gym.utils import get_args, task_registry


URDF_PATH = Path(LEGGED_GYM_ROOT_DIR) / "resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
COLLISION_MESH = "meshes/mini_calf_collision.STL"


def _parse_vector(value, length, name):
    if value is None:
        return None
    if len(value) != length:
        raise ValueError(f"{name} requires {length} values")
    return tuple(float(item) for item in value)


def _read_collision_transform(collision):
    if collision is None:
        raise ValueError("No collision element found on a calf link")

    origin = collision.find("origin")
    rpy = tuple(float(item) for item in origin.get("rpy", "0 0 0").split()) if origin is not None else (0., 0., 0.)
    xyz = tuple(float(item) for item in origin.get("xyz", "0 0 0").split()) if origin is not None else (0., 0., 0.)

    mesh = collision.find("geometry/mesh")
    if mesh is None:
        raise ValueError("No mesh collision geometry found on a calf link")
    scale = tuple(float(item) for item in mesh.get("scale", "1 1 1").split())
    return rpy, xyz, scale


def _get_calf_collision_transform(urdf_path):
    root = ET.parse(urdf_path).getroot()
    calf = next(link for link in root.findall("link") if link.get("name", "").endswith("_calf"))
    return _read_collision_transform(calf.find("collision"))


def _set_visual_to_collision(urdf_path, output_path, rpy_override, xyz_override, scale_override):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        if not link.get("name", "").endswith("_calf"):
            continue

        link_rpy, link_xyz, link_scale = _read_collision_transform(link.find("collision"))
        rpy = rpy_override if rpy_override is not None else link_rpy
        xyz = xyz_override if xyz_override is not None else link_xyz
        scale = scale_override if scale_override is not None else link_scale

        visual = link.find("visual")
        if visual is None:
            visual = ET.SubElement(link, "visual")

        geometry = visual.find("geometry")
        if geometry is None:
            geometry = ET.SubElement(visual, "geometry")
        for child in list(geometry):
            geometry.remove(child)
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": COLLISION_MESH, "scale": " ".join(f"{item:.12g}" for item in scale)},
        )

        origin = visual.find("origin")
        if origin is None:
            origin = ET.SubElement(visual, "origin")
        origin.set("rpy", " ".join(f"{item:.12g}" for item in rpy))
        origin.set("xyz", " ".join(f"{item:.12g}" for item in xyz))

        old_material = visual.find("material")
        if old_material is not None:
            visual.remove(old_material)
        material = ET.SubElement(visual, "material", {"name": "collision_preview_red"})
        ET.SubElement(material, "color", {"rgba": "1 0.05 0.02 1"})

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def _read_binary_stl_bounds(path):
    """Return raw STL bounds and triangle count when the file is binary STL."""
    data = path.read_bytes()
    if len(data) < 84:
        return None
    triangle_count = int.from_bytes(data[80:84], byteorder="little")
    if 84 + 50 * triangle_count != len(data):
        return None

    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for index in range(triangle_count):
        offset = 84 + index * 50 + 12
        for vertex in range(3):
            values = struct.unpack_from("<3f", data, offset + vertex * 12)
            for axis, value in enumerate(values):
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
    return mins, maxs, triangle_count


def _parse_preview_options():
    """Parse script-only options, then leave Isaac Gym options for get_args()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--collision-rpy", nargs=3, type=float, default=None)
    parser.add_argument("--collision-xyz", nargs=3, type=float, default=None)
    parser.add_argument("--collision-scale", nargs=3, type=float, default=None)
    parser.add_argument(
        "--base-height",
        type=float,
        default=2.75,
        help="Height of the fixed robot base above the plane in meters.",
    )
    preview_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return preview_args


def main():
    preview_args = _parse_preview_options()
    args = get_args()
    if args.headless:
        raise SystemExit("This script needs a viewer; remove --headless.")

    if not URDF_PATH.is_file():
        raise FileNotFoundError(URDF_PATH)
    collision_path = URDF_PATH.parent / COLLISION_MESH
    if not collision_path.is_file():
        raise FileNotFoundError(collision_path)

    default_rpy, default_xyz, default_scale = _get_calf_collision_transform(URDF_PATH)
    rpy_override = _parse_vector(preview_args.collision_rpy, 3, "--collision-rpy")
    xyz_override = _parse_vector(preview_args.collision_xyz, 3, "--collision-xyz")
    scale_override = _parse_vector(preview_args.collision_scale, 3, "--collision-scale")

    bounds = _read_binary_stl_bounds(collision_path)
    print(f"Collision mesh: {collision_path}")
    if bounds is not None:
        mins, maxs, triangles = bounds
        size = [maxs[i] - mins[i] for i in range(3)]
        print(f"Raw STL: {triangles} triangles, bounds min={mins}, max={maxs}, size={size}")
    print(f"First calf collision rpy={default_rpy}, xyz={default_xyz}, scale={default_scale}")
    if rpy_override is not None or xyz_override is not None or scale_override is not None:
        print(f"Preview overrides rpy={rpy_override}, xyz={xyz_override}, scale={scale_override}")
    print(f"Preview base height: {preview_args.base_height:.3f} m")
    print("The calf collision mesh is displayed as red geometry in the viewer.")
    print("Close the viewer window or press Esc to exit.")

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix="_collision_preview.urdf", dir=URDF_PATH.parent, delete=False
    ) as temporary_urdf:
        preview_urdf = Path(temporary_urdf.name)
    try:
        _set_visual_to_collision(
            URDF_PATH,
            preview_urdf,
            rpy_override,
            xyz_override,
            scale_override,
        )

        env_cfg, _ = task_registry.get_cfgs(name="minich")
        env_cfg.env.num_envs = 1
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.measure_heights = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.noise.add_noise = False
        env_cfg.commands.ranges.lin_vel_x = [0., 0.]
        env_cfg.commands.ranges.lin_vel_y = [0., 0.]
        env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
        env_cfg.commands.heading_command = False
        base_height = preview_args.base_height
        env_cfg.init_state.pos = [0., 0., base_height]
        env_cfg.asset.file = str(preview_urdf)
        env_cfg.asset.fix_base_link = True
        env_cfg.asset.disable_gravity = True
        env_cfg.viewer.pos = [1.4, -1.4, base_height + 0.45]
        env_cfg.viewer.lookat = [0., 0., base_height - 0.15]

        # Force this dedicated script to create exactly one minich environment.
        args.task = "minich"
        args.num_envs = 1
        env, _ = task_registry.make_env(name="minich", args=args, env_cfg=env_cfg)
        try:
            while not env.gym.query_viewer_has_closed(env.viewer):
                env.render(sync_frame_time=True)
        finally:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
    finally:
        preview_urdf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
