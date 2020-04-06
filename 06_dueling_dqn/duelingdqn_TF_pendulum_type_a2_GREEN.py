import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import sys

env = gym.make('Pendulum-v0')
# env = env.unwrapped
env.seed(1)

state_size = env.observation_space.shape[0]
action_size = 25

np.random.seed(1)
tf.set_random_seed(1)

class DQN_network:
    def __init__(self,sess=None):
        self.action_size         = action_size
        self.state_size          = state_size
        self.learning_rate       = 0.001
        self.discount_factor     = 0.9
        self.epsilon_max         = 0.9
        self.target_update_cycle = 200
        self.memory_size         = 3000
        self.batch_size          = 32
        self.epsilon_increment   = 0.001
        self.epsilon             = 0 
        self.memory = np.zeros((self.memory_size, state_size*2+2))
        self.build_model()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        
        self.cost_his = []

    def build_model(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_size, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            # Dueling DQN
            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, self.action_size], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.action_size], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)

            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.state_size], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.action_size], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.next_state = tf.placeholder(tf.float32, [None, self.state_size], name='next_state')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.next_state, c_names, n_l1, w_initializer, b_initializer)

    def append_sample(self, s, a, r, next_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.action_size)
        return action

    def train_model(self):
        
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        
        mini_batch = self.memory[sample_index, :]

        q_next = self.sess.run(self.q_next, feed_dict={self.next_state: mini_batch[:, -self.state_size:]}) # next state
        q_eval = self.sess.run(self.q_eval, {self.s: mini_batch[:, :self.state_size]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = mini_batch[:, self.state_size].astype(int)
        reward = mini_batch[:, self.state_size + 1]

        selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.discount_factor * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: mini_batch[:, :self.state_size],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment    
        else: 
            self.epsilon_max
            
sess = tf.Session()
with tf.variable_scope('dueling'):
    dueling_DQN = DQN_network(sess=sess)

sess.run(tf.global_variables_initializer())

def train(agent):
    acc_r = [0]
    total_steps = 0    
    episode = 0
    start_time = time.time()
    
    while total_steps < agent.memory_size:
        done = False
        score = 0
        rewards = 0
        state = env.reset()
        
        while not done:
            action = agent.get_action(state)

            f_action = (action-(action_size-1)/2)/((action_size-1)/4)   # convert to [-2 ~ 2] float actions
            next_state, reward, done, _ = env.step(np.array([f_action]))

            reward /= 10
            rewards += reward
            acc_r.append(reward + acc_r[-1])  # accumulated reward

            agent.append_sample(state, action, reward, next_state)

            state = next_state
            score += 1 
            total_steps += 1
            
            if done:
                print("episode :{:>4} last step :{:>3} rewards :{:>6.2f}".format(episode, score, rewards),
                      "last angle :{:>5.1f}".format(np.arccos(state[0])*180/np.pi),"degree")
                break
    
    while time.time() - start_time < 4*60:
        done = False
        score = 0
        rewards = 0
        state = env.reset()
        
        while not done:
            # if np.arccos(state[0])*180/np.pi < 8:
            env.render()

            action = agent.get_action(state)

            f_action = (action-(action_size-1)/2)/((action_size-1)/4)   # convert to [-2 ~ 2] float actions
            next_state, reward, done, _ = env.step(np.array([f_action]))

            reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
            # the Q target at upright state will be 0, because Q_target = r + discount_factor * Qmax(s', a') = 0 + discount_factor * 0
            # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.
            rewards += reward
            acc_r.append(reward + acc_r[-1])  # accumulated reward

            agent.append_sample(state, action, reward, next_state)

            agent.train_model()
                
            state = next_state
            score += 1
            total_steps += 1
            
            if done:
                episode += 1
                print("episode :{:>4} last step :{:>3} rewards :{:>6.2f}".format(episode, score, rewards),
                      "last angle :{:>5.1f}".format(np.arccos(state[0])*180/np.pi),"degree")
                agent.sess.run(agent.replace_target_op)
                break

    return agent.cost_his, acc_r

c_dueling, r_dueling = train(dueling_DQN)
