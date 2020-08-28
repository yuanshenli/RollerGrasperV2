from robot_env import *
import time

with open('expert_targets/target_axis_list.npy', 'rb') as f:
    target_axis_list = np.load(f)
env = RobotEnv()

def demo_loop():
    # for ep_ in range(MAX_EPISODES):
    for target_axis in target_axis_list:
        print(target_axis)
        env.reset(target_axis=target_axis)
        while not env.termination:
            action = env.get_expert_action()
            obs, done, suc = env.step(action)
            time.sleep(0.02)  # https://github.com/openai/mujoco-py/issues/340
        print(env.test_min_err)

if __name__ == '__main__':
    demo_loop()



