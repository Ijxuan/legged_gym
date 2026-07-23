"""Mini Cheetah task implementation hooks."""

import os
import tempfile

import numpy as np

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.minich.leg_length_randomization import write_leg_length_variant


class MiniCheetah(LeggedRobot):
    """LeggedRobot with startup-time Mini Cheetah morphology randomization."""

    def _load_robot_assets(self, asset_root, asset_file, asset_options):
        domain_rand = self.cfg.domain_rand
        if not getattr(domain_rand, "randomize_leg_lengths", False):
            return super()._load_robot_assets(asset_root, asset_file, asset_options)

        thigh_range = np.asarray(domain_rand.thigh_link_length_range_m, dtype=float)
        calf_range = np.asarray(domain_rand.calf_link_length_range_m, dtype=float)
        if thigh_range.shape != (2,) or calf_range.shape != (2,):
            raise ValueError("Mini Cheetah leg-length ranges must each contain [min_m, max_m]")
        if np.any(thigh_range <= 0.0) or np.any(calf_range <= 0.0):
            raise ValueError("Mini Cheetah leg-length ranges must be positive")
        if thigh_range[0] > thigh_range[1] or calf_range[0] > calf_range[1]:
            raise ValueError("Mini Cheetah leg-length range minimum cannot exceed maximum")

        num_variants = int(domain_rand.leg_length_randomization_num_variants)
        if num_variants < 1:
            raise ValueError("leg_length_randomization_num_variants must be at least 1")
        num_variants = min(num_variants, self.num_envs)

        source_path = os.path.join(asset_root, asset_file)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)

        # Keep temporary URDFs outside the source checkout.  The meshes remain
        # visible through this symlink because their URDF paths are relative.
        self._leg_length_urdf_directory = tempfile.TemporaryDirectory(
            prefix="legged_gym_minich_leg_lengths_"
        )
        os.symlink(os.path.join(asset_root, "meshes"), os.path.join(self._leg_length_urdf_directory.name, "meshes"))

        sampled_lengths = np.column_stack((
            np.random.uniform(thigh_range[0], thigh_range[1], num_variants),
            np.random.uniform(calf_range[0], calf_range[1], num_variants),
        ))
        robot_assets = []
        for index, (thigh_length_m, calf_length_m) in enumerate(sampled_lengths):
            variant_file = f"mini_cheetah_leg_length_{index:02d}.urdf"
            variant_path = os.path.join(self._leg_length_urdf_directory.name, variant_file)
            write_leg_length_variant(source_path, variant_path, thigh_length_m, calf_length_m)
            robot_assets.append(
                self.gym.load_asset(
                    self.sim, self._leg_length_urdf_directory.name, variant_file, asset_options
                )
            )

        # Distribute every variant across the parallel environments, then
        # shuffle assignment.  A morphology remains fixed for an environment's
        # lifetime because PhysX joint anchors cannot be changed on reset.
        asset_indices = np.arange(self.num_envs, dtype=np.int64) % num_variants
        np.random.shuffle(asset_indices)
        self.leg_length_samples_m = sampled_lengths[asset_indices]
        print(
            "Mini Cheetah leg-length randomization: "
            f"{num_variants} variants, thigh={thigh_range.tolist()} m, calf={calf_range.tolist()} m"
        )
        return robot_assets, asset_indices
