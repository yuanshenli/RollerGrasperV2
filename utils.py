import numpy as np
import math
from scipy.spatial.transform import Rotation as R


###############################################
# Helper functions
###############################################


def project_a_onto_b(vec_a, vec_b):
    """
    :param vec_a:
    :param vec_b:
    :return: the projected vector a onto vector b
    """
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_b) ** 2) * vec_b


def project_vec_onto_plane(vec_a, n_b):
    """
    :param vec_a:
    :param n_b:
    :return: the projected vector a onto plane defined by its normal vector b
    """
    return vec_a - project_a_onto_b(vec_a, n_b)


def normalize_vec(vec_a):
    """
    :param vec_a: original vec
    :return: normalized vector
    """
    mag_vec = np.linalg.norm(vec_a)
    if mag_vec == 0:
        print("Cannot normalize vector of length 0")
    else:
        vec_a = vec_a / mag_vec
    return vec_a


def quat_to_quat(mj_quat):
    """
    Convert Mujoco quaternion (w, x, y, z) to Scipy quaternion (x, y, z, w)
    :param my_quat: Mujoco quaternion (w, x, y, z)
    :return: Scipy quaternion (x, y, z, w)
    """
    sp_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])
    return sp_quat


def quat_to_rot(mj_quat):
    """
    Convert Mujoco quaternion (w, x, y, z) to Scipy rotvec
    :param mj_quat: Mujoco quaternion (w, x, y, z)
    :return: Scipy rotvec
    """
    sp_quat = quat_to_quat(mj_quat)
    r = R.from_quat(sp_quat).as_rotvec()
    return r


def angle(v1, v2):
    """
    Calculate the angle between two vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: angle in rad between v1 and v2
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_in_rad = np.arccos(cos_angle)
    return angle_in_rad


def rad_to_deg(val):
    """
    :param val: angle in radians
    :return: angle in degrees
    """
    return val / np.pi * 180.


def deg_to_rad(val):
    """
    :param val: angle in degrees
    :return: angle in radians
    """
    return val / 180. * np.pi


def my_sign(raw):
    """
    :param raw: input value
    :return: 1 if input >= 0; -1 otherwise.
    """
    return (raw >= 0) * 2 - 1


def get_non_orthogonal_projections(v1, v2, v3):
    """
    Project v1 onto v2 and v3 (v2 and v3 might not be orthogonal)
    so that v1 = alpha * v2 + beta * v3
    :param v1: vector being projected
    :param v2: first projection direction
    :param v3: second projection direction
    :return: alpha * v2, beta * v3
    """
    n_z = normalize_vec(np.cross(v2, v3))      # normal to all three vectors
    alpha = np.dot(n_z, np.cross(v3, v1)) / np.dot(n_z, np.cross(v3, v2))
    beta = np.dot(n_z, np.cross(v2, v1)) / np.dot(n_z, np.cross(v2, v3))
    return alpha * v2, beta * v3


def distance_between_lines(p1, r1, p2, r2):
    """
    Find the distance between two lines in 3D space
    :param p1: a point in space where line 1 passes through
    :param r1: direction of line 1
    :param p2: a point in space where line 2 passes through
    :param r2: direction of line 2
    :return: distance between line 1 and line 2
    """
    n_vec = normalize_vec(np.cross(r1, r2))
    dist = 0
    if n_vec is not None:
        dist = np.dot(n_vec, (p1 - p2))
    return dist

###############################################
# Gripper related functions
###############################################


def base_dof_pos_convert(val):
    """
    solve the discrepancy in Mujoco base dof zero position and hardware real zero position
    :param val: base dof position on hardware (0 to 90 degrees)
    :return: base dof position in Mujoco (45 to -45 degrees)
    """
    val = 180. - val    # 45 to -45 degree
    return val


def simulation_to_real_base(val):
    """
    :param val: 45 to -45 degree (plus 135 degree offset)
    :return: 0 to 90 degree
    """
    return 45. - val


def get_contact_vel_omega(pi, po, axis_o, kw, r_roller):
    """
    :param pi: roller pos
    :param po: object pos
    :param axis_o: object pos
    :param kw: object rolling speed  (scalar)
    :param r_roller: roller radius
    :return: velocity at contact point due to object rotation (assuming rotating about the desired axis)
    """
    p_io = po - pi                                      # roller center to object center
    pc = pi + r_roller * p_io / np.linalg.norm(p_io)    # contact position
    p_co = po - pc                                      # contact point to object center along p_io
    v_omega = np.cross(p_co, axis_o) * kw               #
    return v_omega


def get_contact_vel_v(po_curr, po_target, kv):
    """
    :param po_curr: current obj pos
    :param po_target: target obj pos
    :param kv: speed to move towards target
    :return: velocity at contact point due to object translation
    """
    dp_o = (po_target - po_curr)
    dp_o_mag = np.linalg.norm(dp_o)
    dir_o = dp_o
    if dp_o_mag != 0:
        dir_o = dir_o / dp_o_mag
    v_vel = dir_o * kv
    return v_vel


def get_finger_dir(pivot_axis, base_axis):
    """
    :param pivot_axis: pivot axis
    :param base_axis: base axis
    :return: direction along the finger
    """
    finger_dir = np.cross(pivot_axis, base_axis)
    finger_dir *= my_sign(finger_dir[2])
    return normalize_vec(finger_dir)


def get_pivot_axis(finger_normal, base_angle):
    """
    :param finger_normal: direction from base position pointing towards global z-axis
    :param base_angle: angle of base dof (from the horizontal direction)
    :return: pivot axis
    """
    if finger_normal[2] != 0:
        raise ValueError("axis_horizontal must have z component being 0.")
    pivot_axis = np.array([finger_normal[0] * np.cos(base_angle),   # projection to get rotated x-comp
                           finger_normal[1] * np.cos(base_angle),   # projection to get rotated y-comp
                           np.sin(base_angle)])                     # projection to get rotated z-comp
    return normalize_vec(pivot_axis)


def get_rolling_axis(base_axis, pivot_axis, contact_vel_dir):
    """
    :param base_axis: axis of the base dof
    :param pivot_axis: axis of the pivot dof (middle joint)
    :param contact_vel_dir: direction of the contact point between the roller and the object
    :return: axis of the roller
    """
    rolling_axis = np.cross(pivot_axis, contact_vel_dir)  # rolling axis vec; are these in the same plane?
    if_pointing_up = my_sign(rolling_axis[2])  # if the axis is pointing up
    rolling_axis *= if_pointing_up  # helps to avoid change-point?
    pivot_angle = angle(rolling_axis, base_axis)    #
    pivot_angle = np.pi / 2 - pivot_angle  # pivot angle (used for controlling pivot joint)
    return normalize_vec(rolling_axis), pivot_angle, if_pointing_up


def get_obj_axis(curr_orientation, target_orientation):
    """
    Get the normalized rotation axis of the object
    :param curr_orientation: current orientation of object in quaternion (w, x, y, z)
    :param target_orientation: target orientation of object in quaternion (w, x, y, z)
    :return: normalized rotation axis
    """
    curr_rot_matrix = R.from_quat(quat_to_quat(curr_orientation)).as_matrix()   # rotation matrix for current ori in global
    tgt_rot_matrix = R.from_quat(quat_to_quat(target_orientation)).as_matrix()  # rotation matrix for target ori in global
    curr_to_tgt_rot = np.matmul(np.transpose(curr_rot_matrix), tgt_rot_matrix)  # rotation matrix from current to target in local coordinates
    curr_to_tgt_axis_angle = R.from_matrix(curr_to_tgt_rot).as_rotvec()         # convert to axis angle (in local coordinates)
    obj_axis_local = normalize_vec(curr_to_tgt_axis_angle)                      # normalize
    obj_axis = np.matmul(curr_rot_matrix, obj_axis_local)                       # convert from current frame to global
    # print("obj_axis: ", obj_axis)
    return obj_axis

def get_base_axis(finger_normal):
    """
    :param finger_normal: direction from base position pointing towards global z-axis
    :return: axis of base dof
    """
    return np.cross(np.array([0, 0, 1]), finger_normal)


def get_quat_error_v1(quat_1, quat_2):
    """
    Find the distance between two quaternions (w, x, y, z)
    :param quat_1: the first quaternion
    :param quat_2: the second quaternion
    :return: distance between the two quats
    """
    skew_mat = np.array([[0, -quat_1[3], quat_1[2]],
                         [quat_1[3], 0, -quat_1[1]],
                         [-quat_1[2], quat_1[1], 0]])
    ori_err = quat_1[0] * quat_2[1:4] - quat_2[0] * quat_1[1:4] - np.matmul(skew_mat, quat_2[1:4])
    return np.linalg.norm(ori_err)

def get_quat_error(quat_1, quat_2):
    """
    Find the distance between two quaternions (w, x, y, z)
    :param quat_1: the first quaternion
    :param quat_2: the second quaternion
    :return: distance between the two quats
    """
    ori_err = min(np.linalg.norm(quat_1 - quat_2),np.linalg.norm(quat_1 + quat_2))
    ori_err = ori_err / math.sqrt(2)
    return ori_err



def get_pos_error(pos1, pos2):
    """
    L2 distance between two points in space
    :param pos1: the first point
    :param pos2: the second point
    :return: distancem between to points
    """
    return np.linalg.norm(pos2 - pos1)


def get_roller_pos_from_joint(base_angle, finger_length, pos_base, finger_normal):
    """
    :param base_angle: angle of the base joint
    :param finger_length: length of finger (from base joint to center of roller)
    :param pos_base: position of the base joint
    :param finger_normal: direction from base position pointing towards global z-axis
    :return: base_axis: base axis
    :return: pivot_axis: pivot axis
    :return: finger_dir: direction along finger
    :return: pos_roller: position of the roller (calculated based on base joint)
    """
    base_axis = get_base_axis(finger_normal)
    pivot_axis = get_pivot_axis(finger_normal, base_angle)
    finger_dir = get_finger_dir(pivot_axis, base_axis)
    pos_roller = pos_base + finger_length * finger_dir
    return base_axis, pivot_axis, finger_dir, pos_roller




def compute_pivot_and_roller(k_vw, k_vv, base_angle, pos_obj_curr, pos_obj_target, ori_obj_curr, ori_obj_target,
                             r_roller, finger_length, pos_base, base_axis, finger_normal):
    """
    Compute output for each joint for the finger (used in sim)
    :param k_vw: scaling factor for contact velocity resulted from object rotation
    :param k_vv: scaling factor for contact velocity resulted from object translation
    :param base_angle: angle of the base joint
    :param pos_obj_curr: current object position
    :param pos_obj_target: target object position
    :param ori_obj_curr: current object orientation
    :param ori_obj_target: target object orientation
    :param r_roller: roller radius
    :param finger_length: length of finger (from base joint to center of roller)
    :param pos_base: position of the base joint
    :param finger_normal: direction from base position pointing towards global z-axis
    :return: q_pivot: absolute position of pivot dof
    :return: dq_roller: relative position of roller dof
    """
    pivot_axis = get_pivot_axis(finger_normal, base_angle)  # axis of rotation of the sphere in global coordinates
    finger_dir = get_finger_dir(pivot_axis, base_axis)      # direction along finger
    pos_roller = pos_base + finger_length * finger_dir      # position of the roller (calculated based on base joint)

    err_quat = get_quat_error_v1(ori_obj_curr, ori_obj_target)
    obj_axis = get_obj_axis(ori_obj_curr, ori_obj_target)   # desired axis of rotation
    err_pos = get_pos_error(pos_obj_curr, pos_obj_target)
    kw = err_quat
    kv = err_pos

    v_cw = get_contact_vel_omega(pos_roller, pos_obj_curr, obj_axis, kw, r_roller)  # desired contact velocity due to rotation
    v_cv = get_contact_vel_v(pos_obj_curr, pos_obj_target, kv)                      # desired contact velocity due to translation
    vc = k_vw * v_cw + k_vv * v_cv     # Need to tune how much v_cw and v_cv contribute to the result (orig = 0.5, 0.05)

    rolling_axis, pivot_angle, if_up = get_rolling_axis(base_axis, pivot_axis, normalize_vec(vc))
    q_pivot = pivot_angle

    dq_roller = np.linalg.norm(vc) * if_up
    return q_pivot, dq_roller


def get_circumcenter(pos_a, pos_b, pos_c):
    """
    Identify the circumcenter of a triangle in 3D space
    referred from:
    https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
    :param pos_a: corner of the triangle
    :param pos_b: corner of the triangle
    :param pos_c: corner of the triangle
    :return: circumsphere_center: circumcenter
    :return: circumsphere_radius: circumsphere radius
    """
    vec_ac = pos_c - pos_a
    vec_ab = pos_b - pos_a
    ab_x_ac = np.cross(vec_ab, vec_ac)
    to_circumsphere_center = (np.cross(ab_x_ac, vec_ab) * np.inner(vec_ac, vec_ac)
                          + np.cross(vec_ac, ab_x_ac) * np.inner(vec_ab, vec_ab)) / (2. * np.inner(ab_x_ac, ab_x_ac))
    circumsphere_radius = np.linalg.norm(to_circumsphere_center)
    circumsphere_center = pos_a + to_circumsphere_center
    return circumsphere_center, circumsphere_radius


def get_obj_pos_from_roller_positions(pos_a, pos_b, pos_c, r_obj, r_roller):
    """
    Estimate object (assuming to be a sphere) position based on roller positions
    :param pos_a: center of roller 1
    :param pos_b: center of roller 2
    :param pos_c: center of roller 3
    :param r_obj: radius of object (sphere)
    :param r_roller: radius of roller
    :return: position of object (center of sphere)
    """
    circumsphere_center, circumsphere_radius = get_circumcenter(pos_a, pos_b, pos_c)
    vec_ac = pos_c - pos_a
    vec_ab = pos_b - pos_a
    plane_normal = normalize_vec(np.cross(vec_ab, vec_ac))
    plane_normal *= my_sign(plane_normal[2])        # Always take the positive z direction
    r_cc = r_obj + r_roller
    obj_pos = None
    if r_cc >= circumsphere_radius:     # if rollers in contact with the obj
        plane_offset = np.sqrt(r_cc ** 2 - circumsphere_radius ** 2)
        obj_pos = circumsphere_center + plane_normal * plane_offset

    return obj_pos


def compute_base_and_rolling_vel(v_contact, pos_roller, pos_obj, pivot_axis):
    """
    :param v_contact: contact velocity
    :param pos_roller: position of roller
    :param pos_obj: position of object
    :param pivot_axis: pivot axis
    :return: velocity of roller resulted from (1) base dof motion, and (2) roller rolling motion
    """
    vec_tan = normalize_vec(pos_obj - pos_roller)   # unit vector normal to the contact plane
    vec_vc = normalize_vec(v_contact)               # unit vector of contact velocity direction
    vec_cb = normalize_vec(pivot_axis)              # unit vector pointing to the pivot axis
    vec_cr = normalize_vec(np.cross(np.cross(vec_cb, vec_vc), vec_tan)) # this could be zero if vec_vc aligns with vec_cb

    v_base = vec_vc
    v_roller_resulted = 0
    if np.linalg.norm(vec_cr) != 0:
        v_base, v_roller_resulted = get_non_orthogonal_projections(v_contact, vec_cb, vec_cr)
    return v_base, v_roller_resulted


def compute_joints(k_vw, k_vv, base_angle, pos_obj_curr, pos_obj_target, ori_obj_curr, ori_obj_target,
                   r_roller, finger_length, pos_base, finger_normal):
    """
    Compute output for each joint for the finger (used on hardware)
    :param k_vw: scaling factor for contact velocity resulted from object rotation
    :param k_vv: scaling factor for contact velocity resulted from object translation
    :param base_angle: angle of the base joint
    :param pos_obj_curr: current object position
    :param pos_obj_target: target object position
    :param ori_obj_curr: current object orientation
    :param ori_obj_target: target object orientation
    :param r_roller: roller radius
    :param finger_length: length of finger (from base joint to center of roller)
    :param pos_base: position of the base joint
    :param finger_normal: direction from base position pointing towards global z-axis
    :return: dq_base: relative position of base dof
    :return: q_pivot: absolute position of pivot dof
    :return: dq_roller: relative position of roller dof
    """

    base_axis, pivot_axis, finger_dir, pos_roller = get_roller_pos_from_joint(base_angle=base_angle,
                                                                              finger_length=finger_length,
                                                                              pos_base=pos_base,
                                                                              finger_normal=finger_normal)

    err_quat = get_quat_error(ori_obj_curr, ori_obj_target)
    obj_axis = get_obj_axis(ori_obj_curr, ori_obj_target)   # desired axis of rotation
    err_pos = get_pos_error(pos_obj_curr, pos_obj_target)
    kw = err_quat
    kv = err_pos

    v_cw = get_contact_vel_omega(pos_roller, pos_obj_curr, obj_axis, kw, r_roller)  # desired contact velocity due to rotation
    v_cv = get_contact_vel_v(pos_obj_curr, pos_obj_target, kv)                      # desired contact velocity due to translation
    vc = k_vw * v_cw + k_vv * v_cv     # Need to tune how much v_cw and v_cv contribute to the result (orig = 0.5, 0.05)

    v_base, v_roller_resulted = compute_base_and_rolling_vel(v_contact=vc,
                                                             pos_roller=pos_roller,
                                                             pos_obj=pos_obj_curr,
                                                             pivot_axis=pivot_axis)

    p_ro = pos_obj_curr - pos_roller    # roller center to object center
    pos_contact = pos_roller + r_roller * p_ro / np.linalg.norm(p_ro)  # contact position

    rolling_axis, pivot_angle, if_up = get_rolling_axis(base_axis, pivot_axis, normalize_vec(v_roller_resulted))

    r_effective_base = distance_between_lines(pos_base, base_axis, pos_contact, v_base)
    r_effective_roller = distance_between_lines(pos_roller, rolling_axis, pos_contact, v_roller_resulted)

    q_pivot = pivot_angle
    dq_base = 0
    dq_roller = 0
    if r_effective_base != 0:
        dq_base = np.linalg.norm(v_base) / r_effective_base
    if r_effective_roller != 0:
        dq_roller = np.linalg.norm(v_roller_resulted) / r_effective_roller * if_up
    return dq_base, q_pivot, dq_roller

