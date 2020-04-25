import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
from typing import List

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

discount_factor = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
target_update_cycle = 10
memory_size = 50000
batch_size = 32

scores = []
MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

class DQN:

    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:
        self.session = session
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = name
        
        self.build_model()

    def build_model(self, H_SIZE_01=200,Alpha=0.001) -> None:
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            net_0 = self._X

            net_1 = tf.layers.dense(net_0, H_SIZE_01, activation=tf.nn.relu)
            net_16 = tf.layers.dense(net_1, self.action_size)
            self._Qpred = net_16

            self._LossValue = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=Alpha)
            self._train = optimizer.minimize(self._LossValue)

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = np.reshape(state, [-1, self.state_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._LossValue, self._train], feed)

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

def Copy_Weights(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder
                 
def train_model(agent: DQN, target_agent: DQN, minibatch: list) -> float:
    states      = np.vstack([batch[0] for batch in minibatch])
    actions     = np.array( [batch[1] for batch in minibatch])
    rewards     = np.array( [batch[2] for batch in minibatch])
    next_states = np.vstack([batch[3] for batch in minibatch])
    dones       = np.array( [batch[4] for batch in minibatch])

    X_batch = states
    Y_batch = agent.predict(states)

    Q_Global = rewards + discount_factor * np.max(target_agent.predict(next_states), axis=1) * ~dones

    Y_batch[np.arange(len(X_batch)), actions] = Q_Global

    return agent.update(X_batch, Y_batch)

def main():
    last_n_game_reward = deque(maxlen=100)
    last_n_game_reward.append(0)
    memory = deque(maxlen=memory_size)

    with tf.Session() as sess:
        agent        = DQN(sess, state_size, action_size, name="main")
        target_agent = DQN(sess, state_size, action_size, name="target")
        
        init = tf.global_variables_initializer()
        sess.run(init)

        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        for episode in range(N_EPISODES):
            state = env.reset()
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            rall = 0
            done = False
            count = 0

            while not done and count < 10000 :
                count += 1
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(agent.predict(state))

                nextstate, reward, done, _ = env.step(action)

                if done: 
                    reward = -100
                memory.append((state, action, reward, nextstate, done))

                if len(memory) > memory_size:
                    memory.popleft()
                    
                state = nextstate
                
            if episode % target_update_cycle ==0:
                
                for _ in range (batch_size):
                    minibatch = random.sample(memory, 10)
                    LossValue,_ = train_model(agent,target_agent, minibatch)
                print("LossValue : ",LossValue)
                sess.run(copy_ops)
                print("Episode {:>5} reward:{:>5} recent N Game reward:{:>5.2f} memory length:{:>5}"
                      .format(episode, count, np.mean(last_n_game_reward),len(memory)))
                
            last_n_game_reward.append(count)

            if len(last_n_game_reward) == last_n_game_reward.maxlen:
                avg_reward = np.mean(last_n_game_reward)

                if avg_reward > 199.0:
                    print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                    break

        for episode in range(N_train_result_replay):
            state = env.reset()
            rall = 0
            done = False
            count = 0
            
            while not done :
                env.render()
                count += 1
                Q_Global = agent.predict(state)
                action = np.argmax(Q_Global)
                state, reward, done, _ = env.step(action)
                rall += reward

            scores.append(rall)
            print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, count, rall, np.mean(scores)))

if __name__ == "__main__":
    main()