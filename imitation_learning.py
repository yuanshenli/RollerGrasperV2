import time
from utils import *
import os
import datetime
from scipy.special import softmax
from tensorboardX import SummaryWriter
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from env_expert import EnvExpert

np.set_printoptions(precision=4, suppress=False)


class Learner(object):
    def __init__(self, env_expert, actor_lr, mem_size, state_dim, action_dim, batch_size, save_top_dir="./save_folder",
                 log_writer="./runs_log"):
        self.actor_lr = actor_lr
        self.env_expert = env_expert
        self.mem_size = mem_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.save_top_dir = save_top_dir
        self.max_action_b = 0.05
        self.max_action_m = 0.2
        self.max_action_t = 0.01
        if not os.path.exists(self.save_top_dir):
            os.makedirs(self.save_top_dir)

        self.memory = np.zeros((self.mem_size, self.state_dim + self.action_dim), dtype=np.float32)
        self.pointer = 0
        self.timestep = 0
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.writer_dir = log_writer

        self.actor_lr = actor_lr
        if not os.path.exists(self.writer_dir):
            os.makedirs(self.writer_dir)

        self.writer = SummaryWriter(
            logdir=(self.writer_dir + '/{}_{}_{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                          "env_roller_grasper_v2", "Dagger"))

        self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
        self.gt_action = tf.placeholder(tf.float32, [None, self.action_dim], 'action_imitate')

        with tf.variable_scope('Actor'):
            self.action = self._build_actor(self.state, scope='policy', trainable=True)
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/policy')
        self.iloss = tf.reduce_mean(tf.squared_difference(self.action, self.gt_action))
        self.itrain = tf.train.AdamOptimizer(self.actor_lr).minimize(self.iloss, var_list=self.ae_params)

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.sess.run(tf.global_variables_initializer())

    def _build_actor(self, state, scope, trainable=True):
        action_max = np.ones((1, self.action_dim))
        ####
        action_max[0, :3] = self.max_action_b
        action_max[0, 3:6] = self.max_action_m
        action_max[0, 6:9] = self.max_action_t

        with tf.variable_scope(scope):
            action = tf.layers.dense(state, 256, tf.nn.leaky_relu, trainable=trainable)
            action = tf.layers.dense(action, 256, tf.nn.leaky_relu, trainable=trainable)
            action = tf.layers.dense(action, 256, tf.nn.leaky_relu, trainable=trainable)
            action_base = tf.layers.dense(action, 3, trainable=trainable)
            action_second = tf.layers.dense(action, 3, trainable=trainable)
            action_third = tf.layers.dense(action, 3, trainable=trainable)
            action = tf.concat([action_base, action_second, action_third], axis=-1)
        return action

    def save_model(self, step):
        ckpt_path = os.path.join(self.save_top_dir, str(step) + 'model.ckpt')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_path = self.saver.save(self.sess, ckpt_path, write_meta_graph=False)
        if step == 0:
            self.saver.save(self.sess, ckpt_path, write_meta_graph=True)
        else:
            self.saver.save(self.sess, ckpt_path, write_meta_graph=False)
        print("Saving model at step %d to %s" % (step, save_path))

    def restore(self, step):
        ckpt_path = os.path.join(self.save_top_dir, str(step) + 'model.ckpt')
        print("restoring from %s" % ckpt_path)
        self.saver.restore(self.sess, ckpt_path)

    def choose_action(self, state):
        state = state[np.newaxis, :]  # single state
        action = self.sess.run(self.action, feed_dict={self.state: state})[0]  # single action
        return action

    def store_transition(self, s, gt_a, stage2=0):
        transition = np.hstack((s, gt_a))
        if stage2 == 0:
            index = self.pointer % self.mem_size
            self.memory[index, :] = transition
            self.pointer += 1
        else:
            index = self.pointer % (self.mem_size - stage2)
            self.memory[index + stage2, :] = transition
            self.pointer += 1

    def learn_imitation(self):  # batch update
        self.timestep += 1
        if self.pointer > self.mem_size:
            indices = np.random.choice(self.mem_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)

        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        ba_gt = bt[:, -self.action_dim:]

        _, iloss, pred_action = self.sess.run([self.itrain, self.iloss, self.action],
                                              feed_dict={self.state: bs, self.gt_action: ba_gt})
        return iloss, self.timestep

    def evaluate_expert(self):
        evaluation_result = {}
        for eval_id in self.env_expert.demos:
            observation = self.env_expert.reset(eval_id)
            observation = np.reshape(observation, (-1,))
            reset_flag = True
            while True:
                action = self.choose_action(observation)
                observation_next, done, suc, gt_action = self.env_expert.step_expert()
                observation_next = np.reshape(observation_next, (-1,))
                observation = observation_next
                if done:
                    break
            evaluation_result[eval_id] = self.env_expert.min_err
        print(evaluation_result)
        import json
        json = json.dumps(evaluation_result)
        f = open(os.path.join(self.writer_dir, "expert_performance.json"), "w")
        f.write(json)
        f.close()

    def evaluate(self, restore_step=0, count_test=0):
        total_step = 0
        import json
        with open(os.path.join(self.writer_dir, 'expert_performance.json')) as f:
            expert_perf = json.load(f)
        print("expert_perf", expert_perf)
        self.save_model(0)
        if restore_step > 0:
            self.restore(restore_step)

        evaluation_result = {}
        eval_list = []
        for eval_id in self.env_expert.demos:
            observation = self.env_expert.reset(eval_id)
            observation = np.reshape(observation, (-1,))
            reset_flag = True
            while True:
                action = self.choose_action(observation)

                # delay to help with NaN in simulation from here: https://github.com/openai/mujoco-py/issues/340
                time.sleep(0.02)
                observation_next, done, suc = self.env_expert.step(action)
                # print("gt_action",gt_action,"predicated",action)
                observation_next = np.reshape(observation_next, (-1,))
                observation = observation_next
                if done:
                    break
            evaluation_result[eval_id] = self.env_expert.min_err
            eval_list.append(self.env_expert.min_err / expert_perf[str(eval_id)])
        print(evaluation_result)
        import json
        json = json.dumps(evaluation_result)
        f = open(os.path.join(self.writer_dir, "learn_performance" + str(count_test) + ".json"), "w")
        f.write(json)
        f.close()
        print("return eaval", np.array(eval_list))
        return np.array(eval_list)

    def test(self, restore_step=0, count=0):
        total_step = 0
        import json
        self.save_model(0)
        if restore_step > 0:
            self.restore(restore_step)

        test_result = {}
        test_list = []
        for test_id in self.env_expert.test_task:
            observation = self.env_expert.reset(test_id, test=True)
            observation = np.reshape(observation, (-1,))
            reset_flag = True
            while True:
                action = self.choose_action(observation)
                # delay to help with NaN in simulation from here: https://github.com/openai/mujoco-py/issues/340
                time.sleep(0.02)

                observation_next, done, suc = self.env_expert.step(action)
                observation_next = np.reshape(observation_next, (-1,))
                observation = observation_next
                if done:
                    break
            test_result[test_id] = self.env_expert.min_err
            test_list.append(test_result[test_id])
        print(test_result)
        import json
        json = json.dumps(test_result)
        f = open(os.path.join(self.writer_dir, "learn_test_performance" + str(count) + ".json"), "w")
        f.write(json)
        f.close()
        return np.mean(np.array(test_list))

    def train(self, training_step_stage_one=30000, training_step_stage_two=100000):
        ### generate expert transitions
        for ep_, train_id in enumerate(self.env_expert.demos):
            observation = self.env_expert.reset(train_id)
            self.writer.add_scalar('train/train_id', train_id, ep_)
            observation = np.reshape(observation, (-1,))
            reset_flag = True
            while True:
                action = self.choose_action(observation)

                # delay to help with NaN in simulation from here: https://github.com/openai/mujoco-py/issues/340
                time.sleep(0.02)
                observation_next, done, suc, gt_action = self.env_expert.step_expert()
                observation_next = np.reshape(observation_next, (-1,))

                if not reset_flag:
                    self.store_transition(observation, gt_action)

                reset_flag = False
                observation = observation_next

                if done:
                    break
        print("finish generating expert demonstrations!")
        # stage 1: behavior clone of the expert
        count_test = 0
        total_step = 0
        for iter_i in range(training_step_stage_one):
            total_step += 1
            iloss, _ = self.learn_imitation()
            self.writer.add_scalar('train/iloss', iloss, total_step)

            if total_step % 5000 == 0:
                self.save_model(total_step)
                res_list = self.evaluate(total_step, total_step)
                self.writer.add_scalar('train/err', np.mean(res_list), total_step)
                res_test = self.test(total_step, total_step)
                count_test += 1
                self.writer.add_scalar('test/err', res_test, total_step)

        memory_base = min(self.pointer, int(self.mem_size / 2))
        res_list = self.evaluate()

        # stage 2: learn on-line
        for ep_ in range(training_step_stage_two):
            prob = softmax(res_list)
            print("prob", prob)
            print("a", self.env_expert.demos)
            train_id = np.random.choice(self.env_expert.demos, p=prob)
            print("train_id", train_id)
            observation = self.env_expert.reset(train_id)
            observation = np.reshape(observation, (-1,))
            reset_flag = True
            while True:
                action = self.choose_action(observation)
                # delay to help with NaN in simulation from here: https://github.com/openai/mujoco-py/issues/340
                time.sleep(0.02)

                action_expert = self.env_expert.get_expert_action()
                observation_next, done, suc = self.env_expert.step(action)
                observation_next = np.reshape(observation_next, (-1,))
                self.store_transition(observation, action_expert, stage2=memory_base)
                self.writer.add_scalar("dagger_v2", np.linalg.norm(action - action_expert), total_step)
                total_step += 1
                iloss, timestep = self.learn_imitation()
                self.writer.add_scalar('train/iloss', iloss, timestep)
                observation = observation_next
                if total_step % 1000 == 0:
                    self.save_model(total_step)
                    res_list = self.evaluate(total_step, total_step)
                    self.writer.add_scalar('train/err', np.mean(res_list), total_step)
                    res_test = self.test(total_step, total_step)
                    count_test += 1
                    self.writer.add_scalar('test/err', res_test, total_step)
                if done:
                    break


if __name__ == "__main__":
    env_expert = EnvExpert()
    learner = Learner(env_expert, actor_lr=1e-5, mem_size=100000, state_dim=35, action_dim=9, batch_size=256)
    learner.evaluate_expert()
    learner.train(training_step_stage_two=10000)
    learner.test()
