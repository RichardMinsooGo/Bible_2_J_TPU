import tensorflow as tf
import gym
import numpy as np
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

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)    

# 5.	Define algorithm parameters (hyperparameters)
# Learning Rate = Alpha
Alpha = 0.001
# Discount Factor = discount_factor
discount_factor = 0.99
N_EPISODES = 500
N_train_result_replay = 20
PRINT_CYCLE = 10

# Hidden Layer 01 Size 
H_SIZE_01 = 200

rlist=[]
MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

# Main Network Initialization / 네트워크 구성
X=tf.placeholder(dtype=tf.float32, shape=(None, state_size), name="input_X")
Y=tf.placeholder(dtype=tf.float32, shape=(None, action_size), name="output_Y")

W01_m=tf.get_variable('W01_m',shape=[state_size,H_SIZE_01],initializer=tf.contrib.layers.xavier_initializer())
W16_m=tf.get_variable('W16_m',shape=[H_SIZE_01,action_size],initializer=tf.contrib.layers.xavier_initializer())

LAY01_m=tf.nn.relu(tf.matmul(X,W01_m))
Qpred_m = tf.matmul(LAY01_m,W16_m)

LossValue = tf.reduce_sum(tf.square(Y - Qpred_m))
train = tf.train.AdamOptimizer(learning_rate=Alpha).minimize(LossValue)

step_history = []

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

episode = 0
start_time = time.time()
while time.time() - start_time < 20*60:

    state = env.reset()
    e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
    done = False
    count = 0

    while not done and count < 10000:
        count += 1
        state_reshape = np.reshape(state, [1, state_size])
        Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_Global)

        nextstate, reward, done, _ = env.step(action)
        
        if not done:
            nextstate_reshape = np.reshape(nextstate, [1, state_size])
            Q_next = sess.run(Qpred_m, feed_dict={X: nextstate_reshape})
            Q_Global[0, action] = reward + discount_factor * np.max(Q_next)

        sess.run(train, feed_dict={X: state_reshape, Y: Q_Global})
        state = nextstate

        if done or count == 10000:
            episode += 1
            print("Episode {:>5}/ Episode count:{:>5}"
                          .format(episode, count))
    step_history.append(count)
    if len(step_history) > 100 and np.mean(step_history[-30:]) < 300:
        print("\n\n Training Finished \n")
        # sys.exit()
        break
        
for episode in range(N_train_result_replay):
    state = env.reset()
    rall = 0
    done = False
    count = 0
    while not done and count < 10000:
        env.render()
        count += 1
        state_reshape = np.reshape(state, [1, state_size])
        Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
        action = np.argmax(Q_Global)
        state, reward, done, _ = env.step(action)
        rall += reward

    rlist.append(rall)
    print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, count, rall,
                                                                    np.mean(rlist)))

        