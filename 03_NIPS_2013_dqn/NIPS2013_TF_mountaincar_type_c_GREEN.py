import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import time
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

# In case of CartPole-v1, maximum length of episode is 500
env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

discount_factor = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
PRINT_CYCLE = 10
memory_size = 50000
batch_size = 32

rlist=[]
MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

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


def train_minibatch(mainDQN: DQN, minibatch: list) -> float:
    states      = np.vstack([batch[0] for batch in minibatch])
    actions     = np.array( [batch[1] for batch in minibatch])
    rewards     = np.array( [batch[2] for batch in minibatch])
    next_states = np.vstack([batch[3] for batch in minibatch])
    dones       = np.array( [batch[4] for batch in minibatch])

    X_batch = states
    Y_batch = mainDQN.predict(states)

    Q_Global = rewards + discount_factor * np.max(mainDQN.predict(next_states), axis=1) * ~dones
    Y_batch[np.arange(len(X_batch)), actions] = Q_Global

    LossValue, _ = mainDQN.update(X_batch, Y_batch)

    return LossValue

def main():
    last_n_game_reward = deque(maxlen=30)
    last_n_game_reward.append(10000)
    replay_buffer = deque(maxlen=memory_size)

    with tf.Session() as sess:
        mainDQN = DQN(sess, state_size, action_size, name="main")
        init = tf.global_variables_initializer()
        sess.run(init)
        
        episode = 0
        step = 0
        start_time = time.time()

        while time.time() - start_time < 30*60:
            
            state = env.reset()
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            score = 0
            done = False
            count = 0
            progress = ""
            
            while not done and count < 10000 :
                if step < memory_size:
                    progress = "exploration"
                else:
                    progress = "training"
                
                count += 1
                step += 1
                
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > memory_size:
                    replay_buffer.popleft()
                    
                state = next_state
                score += 1
                
                if progress == "training":
                    minibatch = random.sample(replay_buffer, batch_size)
                    train_minibatch(mainDQN, minibatch)
                    
                if done or score == 10000:
                    if progress == "training":
                        episode += 1
                    print("Episode {:>5}/ reward:{:>5}/ recent n game reward:{:>5.2f}/ memory length:{:>5}"
                      .format(episode, count, np.mean(last_n_game_reward),len(replay_buffer)),"/ Progress :",progress)
            
                    break
            last_n_game_reward.append(count)

            if len(last_n_game_reward) == last_n_game_reward.maxlen:
                avg_reward = np.mean(last_n_game_reward)

                if avg_reward < 200:
                    print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                    sys.exit()

if __name__ == "__main__":
    main()
