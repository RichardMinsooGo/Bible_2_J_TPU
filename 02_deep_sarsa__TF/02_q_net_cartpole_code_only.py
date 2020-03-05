import tensorflow as tf
import gym
import numpy as np
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()
env = gym.make('CartPole-v0')
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

# 5.	Define algorithm parameters (hyperparameters)
# Learning Rate = Alpha
Alpha = 0.001
# Discount Factor = Gamma
Gamma = 0.99
N_EPISODES = 5000
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
X=tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE), name="input_X")
Y=tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_SIZE), name="output_Y")

W01_m=tf.get_variable('W01_m',shape=[INPUT_SIZE,H_SIZE_01],initializer=tf.contrib.layers.xavier_initializer())
W16_m=tf.get_variable('W16_m',shape=[H_SIZE_01,OUTPUT_SIZE],initializer=tf.contrib.layers.xavier_initializer())

LAY01_m=tf.nn.relu(tf.matmul(X,W01_m))
Qpred_m = tf.matmul(LAY01_m,W16_m)

LossValue = tf.reduce_sum(tf.square(Y - Qpred_m))
train = tf.train.AdamOptimizer(learning_rate=Alpha).minimize(LossValue)

step_history = []

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for episode in range(N_EPISODES):
        
    state = env.reset()
    e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
    done = False
    count = 0

    while not done and count < 10000 :
        count += 1
        state_reshape = np.reshape(state, [1, INPUT_SIZE])
        Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_Global)

        nextstate, reward, done, _ = env.step(action)
        if done:
            Q_Global[0, action] = -100
        else:
            nextstate_reshape = np.reshape(nextstate, [1, INPUT_SIZE])
            Q_next = sess.run(Qpred_m, feed_dict={X: nextstate_reshape})
            Q_Global[0, action] = reward + Gamma * np.max(Q_next)

        sess.run(train, feed_dict={X: state_reshape, Y: Q_Global})
        state = nextstate

    step_history.append(count)
    if episode % PRINT_CYCLE == 0 :
        print("Episode {:>5} reward:{:>5} recent N Game reward:{:>5.2f} memory length:{:>5}"
                      .format(episode, count, np.mean(last_N_game_reward),len(replay_buffer)))
    if len(step_history) > 100 and np.mean(step_history[-100:]) > 199:
        break
        
for episode in range(N_train_result_replay):
    state = env.reset()
    rall = 0
    done = False
    count = 0
    while not done :
        env.render()
        count += 1
        state_reshape = np.reshape(state, [1, INPUT_SIZE])
        Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
        action = np.argmax(Q_Global)
        state, reward, done, _ = env.step(action)
        rall += reward

    rlist.append(rall)
    print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, count, rall,
                                                                    np.mean(rlist)))

        