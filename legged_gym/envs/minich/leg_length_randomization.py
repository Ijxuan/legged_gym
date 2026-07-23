"""Build geometry-consistent Mini Cheetah leg-length URDF variants.

Isaac Gym fixes joint anchors when an actor is created.  The vectorized
training environment therefore uses a small set of URDF variants, sampled once
at startup, instead of trying to mutate an actor's link length during reset.
"""

import xml.etree.ElementTree as ET


LEG_PREFIXES = ("FR", "FL", "RR", "RL")


def _parse_xyz(element):
    origin = element.find("origin")
    if origin is None:
        raise ValueError(f"{element.tag} named {element.get('name')} has no origin")
    return origin, [float(value) for value in origin.get("xyz", "0 0 0").split()]


def _write_vector(element, attribute, values):
    element.set(attribute, " ".join(f"{value:.12g}" for value in values))


def _scale_origin_axis(element, axis, scale):
    origin, xyz = _parse_xyz(element)
    xyz[axis] *= scale
    _write_vector(origin, "xyz", xyz)


def _multiply_mesh_scale(mesh, axis, scale):
    mesh_scale = [float(value) for value in mesh.get("scale", "1 1 1").split()]
    mesh_scale[axis] *= scale
    _write_vector(mesh, "scale", mesh_scale)


def _scale_link_inertial_properties(link, scale):
    """Scale a slender link that is elongated along its local z axis."""
    inertial = link.find("inertial")
    if inertial is None:
        return

    _scale_origin_axis(inertial, axis=2, scale=scale)
    mass = inertial.find("mass")
    if mass is not None:
        mass.set("value", f"{float(mass.get('value')) * scale:.12g}")

    inertia = inertial.find("inertia")
    if inertia is None:
        return
    # Cross-section is unchanged: m scales with length, transverse moments
    # scale with length^3, and the long-axis moment scales with length.
    for attribute in ("ixx", "iyy", "ixy"):
        if attribute in inertia.attrib:
            inertia.set(attribute, f"{float(inertia.get(attribute)) * scale ** 3:.12g}")
    for attribute in ("izz",):
        if attribute in inertia.attrib:
            inertia.set(attribute, f"{float(inertia.get(attribute)) * scale:.12g}")
    for attribute in ("ixz", "iyz"):
        if attribute in inertia.attrib:
            inertia.set(attribute, f"{float(inertia.get(attribute)) * scale ** 2:.12g}")


def _scale_thigh_geometry(link, scale):
    # The upper-link OBJ's longitudinal axis is local x before its URDF
    # visual rotation; the actual link's longitudinal axis is local z.
    for mesh in link.findall("visual/geometry/mesh"):
        _multiply_mesh_scale(mesh, axis=0, scale=scale)

    for collision in link.findall("collision"):
        box = collision.find("geometry/box")
        if box is not None:
            size = [float(value) for value in box.get("size").split()]
            size[0] *= scale
            _write_vector(box, "size", size)
            _scale_origin_axis(collision, axis=2, scale=scale)


def _scale_calf_geometry(link, scale):
    # The calf collision STL and its visual use the link-local z axis as the
    # longitudinal direction.
    for mesh in link.findall("visual/geometry/mesh") + link.findall("collision/geometry/mesh"):
        _multiply_mesh_scale(mesh, axis=2, scale=scale)


def _named_element(root, tag, name):
    element = root.find(f"{tag}[@name='{name}']")
    if element is None:
        raise ValueError(f"Mini Cheetah URDF is missing {tag} '{name}'")
    return element


def _joint_length(joint):
    _, xyz = _parse_xyz(joint)
    length = abs(xyz[2])
    if length <= 0.0:
        raise ValueError(f"joint '{joint.get('name')}' has no z-axis link length")
    return length


def write_leg_length_variant(source_path, output_path, thigh_length_m, calf_length_m):
    """Write one Mini Cheetah URDF with all four legs at the requested lengths.

    ``thigh_length_m`` is the hip-to-knee distance and ``calf_length_m`` is
    the knee-to-foot distance.  The function scales both visual and collision
    geometries as well as the corresponding inertial properties.
    """
    if thigh_length_m <= 0.0 or calf_length_m <= 0.0:
        raise ValueError("leg lengths must be positive")

    tree = ET.parse(source_path)
    root = tree.getroot()
    first_calf_joint = _named_element(root, "joint", "FR_calf_joint")
    first_foot_joint = _named_element(root, "joint", "FR_foot_fixed")
    thigh_scale = thigh_length_m / _joint_length(first_calf_joint)
    calf_scale = calf_length_m / _joint_length(first_foot_joint)

    for prefix in LEG_PREFIXES:
        calf_joint = _named_element(root, "joint", f"{prefix}_calf_joint")
        foot_joint = _named_element(root, "joint", f"{prefix}_foot_fixed")
        thigh_link = _named_element(root, "link", f"{prefix}_thigh")
        calf_link = _named_element(root, "link", f"{prefix}_calf")

        _scale_origin_axis(calf_joint, axis=2, scale=thigh_scale)
        _scale_thigh_geometry(thigh_link, thigh_scale)
        _scale_link_inertial_properties(thigh_link, thigh_scale)

        _scale_origin_axis(foot_joint, axis=2, scale=calf_scale)
        _scale_calf_geometry(calf_link, calf_scale)
        _scale_link_inertial_properties(calf_link, calf_scale)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
