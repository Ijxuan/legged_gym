from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.minich.leg_length_randomization import write_leg_length_variant


URDF_PATH = (
    Path(LEGGED_GYM_ROOT_DIR)
    / "resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
)


def _z_origin(root, joint_name):
    origin = root.find(f"joint[@name='{joint_name}']/origin")
    return float(origin.get("xyz").split()[2])


class TestMiniChLegLengthRandomization(unittest.TestCase):
    def test_variant_changes_all_leg_anchors_and_geometry_together(self):
        with TemporaryDirectory() as temporary_dir:
            variant_path = Path(temporary_dir) / "mini_cheetah_variant.urdf"
            write_leg_length_variant(URDF_PATH, variant_path, thigh_length_m=0.215, calf_length_m=0.184)
            root = ET.parse(variant_path).getroot()

        for prefix in ("FR", "FL", "RR", "RL"):
            self.assertAlmostEqual(_z_origin(root, f"{prefix}_calf_joint"), -0.215)
            self.assertAlmostEqual(_z_origin(root, f"{prefix}_foot_fixed"), -0.184)

            thigh = root.find(f"link[@name='{prefix}_thigh']")
            thigh_mesh = thigh.find("visual/geometry/mesh")
            self.assertAlmostEqual(float(thigh_mesh.get("scale").split()[0]), 0.215 / 0.209)
            thigh_box = thigh.find("collision/geometry/box")
            self.assertAlmostEqual(float(thigh_box.get("size").split()[0]), 0.17 * 0.215 / 0.209)

            calf = root.find(f"link[@name='{prefix}_calf']")
            for mesh in calf.findall("visual/geometry/mesh") + calf.findall("collision/geometry/mesh"):
                self.assertAlmostEqual(float(mesh.get("scale").split()[2]), 0.184 / 0.19)

    def test_rejects_non_positive_leg_length(self):
        with TemporaryDirectory() as temporary_dir:
            variant_path = Path(temporary_dir) / "mini_cheetah_variant.urdf"
            with self.assertRaises(ValueError):
                write_leg_length_variant(URDF_PATH, variant_path, thigh_length_m=0.0, calf_length_m=0.19)
