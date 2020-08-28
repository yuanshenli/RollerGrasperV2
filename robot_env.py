from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
from utils import *

XML_PATH = "xml/roller_grasper_v2.xml"
np.set_printoptions(precision=4, suppress=False)

# Constants
MAX_EPISODES = int(1e6)
SCALE_ERROR_ROT = 100
SCALE_ERROR_POS = 100


class RollerFinger(object):
    base_gear_ratio = 3
    pivot_gear_ratio = 10
    roller_radius = 0.0215
    dist_pivot_to_base = 0.1215
    n_fingers = 3

    def __init__(self, idx, name, pos_base, base_normal, base_horizontal):

        self.idx = idx
        self.name = name

        # Finger model
        self.pos_base = pos_base                # position of finger base
        self.base_normal = base_normal          # direction of finger base pointing towards the z-axis of global frame
        self.base_horizontal = base_horizontal  # orthogonal to base_normal, on global x-y plane
        self.q_pivot_prev = 0.0                 # previous position of the pivot joint
        self.q_pivot = 0.0                      # current position of the pivot joint
        self.q_pivot_limit = 3 * np.pi / 180    # speed limit for pivot joint
        self.dq_roller = 0.0                    # relative position of roller joint
        self.init_base_angle = 0.1              # initial base position


class RobotEnv(object):
    def __init__(self):
        self.model = load_model_from_path(XML_PATH)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        # finger instances
        self.front_finger = RollerFinger(idx=0,
                                         name='front', 
                                         pos_base=np.array([0, 0.048, 0.085]),
                                         base_normal=np.array([0., -1., 0.]), 
                                         base_horizontal=np.array([1., 0., 0.]))
        self.left_finger = RollerFinger(idx=1,
                                        name='left', 
                                        pos_base=np.array([-0.04157, -0.024, 0.085]),
                                        base_normal=np.array([0.8660, 0.5, 0.]), 
                                        base_horizontal=np.array([-0.5, 0.8660, 0.]))
        self.right_finger = RollerFinger(idx=2,
                                         name='right', 
                                         pos_base=np.array([0.04157, -0.024, 0.085]),
                                         base_normal=np.array([-0.8660, 0.5, 0.]), 
                                         base_horizontal=np.array([-0.5, -0.8660, 0.]))

        self.r_fingers = [self.front_finger, self.left_finger, self.right_finger]

        # object target orientation in angle-axis representation
        self.axis = normalize_vec(np.array([0., 1., 1.]))
        self.angle = deg_to_rad(90)

        self.init_box_pos = np.array([0.0, 0.0, 0.2])       # obj initial pos
        self.target_box_pos = np.array([0.0, 0.0, 0.2])     # obj target pos

        self.max_step = 1500
        self.termination = False
        self.success = False
        self.timestep = 0

        self.k_vw = 0.5
        self.k_vv = 0.3

        self.reset()

    def reset(self, target_axis=np.array([0., 1., 1.]), target_angle=90):

        self.sim.reset()

        # Reset finger positions
        for fg in self.r_fingers:
            self.sim.data.qpos[fg.idx*3] = fg.init_base_angle   # base angles
            self.sim.data.qpos[fg.idx*3+1] = 0                  # pivot angles

        my_map = [0, 3, 6, 1, 4, 7, 2, 5, 8]     # qpos and ctrl have different joint orders (see xml)
        for ii in range(9):
            self.sim.data.ctrl[ii] = self.sim.data.qpos[my_map[ii]]
        self.sim.step()

        self.termination = False
        self.success = False
        self.timestep = 0

        # reset target
        if target_axis is not None:
            self.axis = normalize_vec(target_axis)
        if target_angle is not None:
            self.angle = deg_to_rad(target_angle)
        self.quat_target = np.array([np.cos(self.angle / 2),
                                     self.axis[0] * np.sin(self.angle / 2),
                                     self.axis[1] * np.sin(self.angle / 2),
                                     self.axis[2] * np.sin(self.angle / 2)])
        self.curr = self.sim.data.sensordata[-7:]  # quat_to_rot(self.sim.data.sensordata[-4:])
        self.rot_matrix_target = R.from_quat(quat_to_quat(self.quat_target)).as_matrix()
        self.test_min_err = 100

        self.session_name = str(self.axis) + ', ' + str(rad_to_deg(self.angle))
        return self.sim.data.sensordata


    def get_expert_action(self):
        # Sensor data
        curr_data = self.sim.data.sensordata
        self.cube_pos = curr_data[-7:-4]        # obj position
        self.cube_orientation = curr_data[-4:]  # obj orientation

        # compute each finger
        for fg in self.r_fingers:
            fg.base_rad = self.sim.data.qpos[fg.idx*3]
            fg.q_pivot_prev = self.sim.data.qpos[fg.idx*3+1]
            fg.q_pivot, fg.dq_roller = compute_pivot_and_roller(k_vw=self.k_vw,
                                                                k_vv=self.k_vv,
                                                                base_angle=fg.base_rad,
                                                                pos_obj_curr=self.cube_pos,
                                                                pos_obj_target=self.target_box_pos,
                                                                ori_obj_curr=self.cube_orientation,
                                                                ori_obj_target=self.quat_target,
                                                                r_roller=fg.roller_radius,
                                                                finger_length=fg.dist_pivot_to_base,
                                                                pos_base=fg.pos_base,
                                                                base_axis=fg.base_horizontal,
                                                                finger_normal=fg.base_normal)

            dq_pivot = np.clip(fg.q_pivot-fg.q_pivot_prev, -fg.q_pivot_limit, fg.q_pivot_limit)
            fg.q_pivot = fg.q_pivot_prev + dq_pivot

            self.sim.data.ctrl[fg.idx] = -fg.init_base_angle
            self.sim.data.ctrl[3+fg.idx] = fg.pivot_gear_ratio * fg.q_pivot
            self.sim.data.ctrl[6+fg.idx] += fg.dq_roller

        action = np.copy(self.sim.data.ctrl)
        return action 

    def step(self, action):
        for i in range(len(action)):
            self.sim.data.ctrl[i] = action[i]

        self.viewer.add_overlay(const.GRID_TOPRIGHT, " ", self.session_name)
        self.viewer.render()
        self.sim.step()
        self.timestep += 1

        obs = self.sim.data.sensordata
        curr = obs[-7:]         # object pose and orientation

        err_curr_rot = get_quat_error(curr[3:], self.quat_target)
        err_curr_pos = get_pos_error(curr[:3], self.target_box_pos)
        err_curr = SCALE_ERROR_ROT * err_curr_rot + SCALE_ERROR_POS * err_curr_pos
        self.test_min_err = min(self.test_min_err, err_curr_rot)
        if err_curr < 15:
            self.termination = True
            self.success = True
        else:
            if self.timestep > self.max_step or err_curr_pos > 0.05:
                self.termination = True
                self.success = False
        #if self.termination:
        #    print("target axis",self.axis,"success",self.success,"min_error",self.test_min_err)
        return obs, self.termination, self.success

