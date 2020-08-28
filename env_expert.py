from utils import *
from robot_env import RobotEnv, SCALE_ERROR_ROT, SCALE_ERROR_POS
import math
import numpy as np

np.set_printoptions(precision=4, suppress=False)


class EnvExpert:
    def __init__(self, env=RobotEnv(), state_dim=35):
        self.state_dim = state_dim
        self.train_target_axis = np.loadtxt('./expert_targets/task_pose_dict.txt', delimiter=',')
        self.train_target_position = np.zeros_like(self.train_target_axis)
        self.test_target_axis = np.loadtxt('./expert_targets/test_pose_dict.txt', delimiter=',')
        self.test_target_position = np.zeros_like(self.test_target_axis)
        self.demos = [i for i in range(len(self.train_target_axis))]
        self.test_task = [i for i in range(len(self.test_target_axis))]
        self.env = env
        self.task_id = 0
        self.noise = False
        self.theta = np.pi / 2
        self.state_dim = state_dim
        self.min_err = 1000.0

    def reset(self, task_id, test=False):
        self.min_err = 10000.0
        self.termination = False
        self.success = False
        self.task_id = task_id
        if not test:
            self.axis = self.train_target_axis[self.task_id]
            self.transl = self.train_target_position[self.task_id]
        else:
            self.axis = self.test_target_axis[self.task_id]
            self.transl = self.test_target_position[self.task_id]

        axis = normalize_vec(self.axis)

        sensordata = self.env.reset(target_axis=axis, target_angle=90)
        self.quat_target = self.env.quat_target

        # current state, previous state
        self.curr = sensordata[-7:]
        self.prev = sensordata[-7:]

        # set target object pose
        self.interl = self.env.axis * self.env.angle
        self.target_obj_pos = self.transl * 100.0
        self.init_obj_ori = np.array([0.0, 0.0, 0.0])
        self.init_obj_pos = np.array([0.0, 0.0, 0.0])

        return self.get_observation(sensordata, self.prev)

    def get_observation(self, sensordata, prev):
        obs = np.zeros((self.state_dim,))
        obs[:16] = sensordata
        if self.noise:
            obs[9:9 + 3] += np.random.normal(0, 0.05, size=(3,))
            obs[12:16] += np.random.normal(0, 0.01, size=(4,))
            new_norm = np.linalg.norm(np.array(obs[12:16]))
            obs[12:16] /= new_norm

        obs[16:16 + 7] = prev
        obs[16 + 7:16 + 7 + 3] = self.interl
        obs[26:29] = self.target_obj_pos  # (self.pos_target - self.pos_init) * 100.0
        obs[29:32] = self.init_obj_ori
        obs[32:35] = self.init_obj_pos
        return obs

    def get_quat_error(self, quat1, quat2):
        ori_err = min(np.linalg.norm(quat1 - quat2), np.linalg.norm(quat1 + quat2))
        ori_err = ori_err / math.sqrt(2)
        return ori_err

    def get_pos_error(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def step_expert(self):
        expert_action = self.env.get_expert_action()
        sensordata, self.termination, self.success = self.env.step(expert_action)
        obs = self.get_observation(sensordata, self.prev)
        self.prev = np.copy(obs[9:16])
        err_curr_rot = self.get_quat_error(self.prev[3:], self.quat_target)
        if self.min_err > err_curr_rot:
            self.min_err = err_curr_rot
        return obs, self.termination, self.success, expert_action

    def step(self, action, show=False):
        self.env.timestep += 1

        sensordata, self.termination, self.success = self.env.step(action)
        obs = self.get_observation(sensordata, self.prev)

        self.prev = np.copy(obs[9:16])
        self.curr = sensordata[-7:]

        err_curr_rot = self.get_quat_error(self.curr[3:], self.quat_target)
        err_curr_pos = self.get_pos_error(self.curr[:3], self.env.target_box_pos)
        err_curr = SCALE_ERROR_ROT * err_curr_rot + SCALE_ERROR_POS * err_curr_pos
        if self.min_err > err_curr_rot:
            self.min_err = err_curr_rot

        if err_curr < 15:
            self.termination = True
            self.success = True
        else:
            if self.env.timestep > self.env.max_step or err_curr_pos > 0.05:
                self.termination = True
                self.success = False
        return obs, self.termination, self.success

    def get_expert_action(self):
        return self.env.get_expert_action()


if __name__ == '__main__':
    env_expert = EnvExpert()
    for i in env_expert.demos:
        env_expert.reset(i)
        print(i)
        while True:
            _, done, _, _ = env_expert.step_expert()
            if done:
                print(env_expert.env.test_min_err)
                break
