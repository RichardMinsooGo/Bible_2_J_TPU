import tensorflow as tf
import gym
import numpy as np
import random as ran
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

learning_rate = 0.001
disscore_factor = 0.99
N_EPISODES = 2000
N_train_result_replay = 20
H_SIZE_01 = 256
UPDATE_CYCLE = 10

replay_buffer = []
memory_size = 50000
batch_size = 32

MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

X=tf.placeholder(dtype=tf.float32, shape=(None, state_size), name="input_X")
Y=tf.placeholder(dtype=tf.float32, shape=(None, action_size), name="output_Y")
dropout = tf.placeholder(dtype=tf.float32)

W01_m = tf.get_variable('W01_m',shape=[state_size, H_SIZE_01]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W16_m = tf.get_variable('W16_m',shape=[H_SIZE_01, action_size]
                        ,initializer=tf.contrib.layers.xavier_initializer())

B01_m = tf.Variable(tf.zeros([1],dtype=tf.float32))

_LAY01_m = tf.nn.relu(tf.matmul(X,W01_m)+B01_m)
LAY01_m = tf.nn.dropout(_LAY01_m,dropout)
Qpred_m = tf.matmul(LAY01_m,W16_m)

rlist=[0]
last_n_game_reward=[0]

episode = 0

LossValue = tf.reduce_sum(tf.square(Y-Qpred_m))
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.01)
train = optimizer.minimize(LossValue)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    last_n_game_reward = deque(maxlen=100)
    last_n_game_reward.append(10000)
    replay_buffer = deque(maxlen=memory_size)
    
    episode = 0
    step = 0
    start_time = time.time()
        
    while time.time() - start_time < 600*60:
        
        state = env.reset()

        e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)

        done = False
        score = 0
        progress = ""

        while not done and score < 10000 :
            if step < memory_size:
                progress = "exploration"
            else:
                progress = "training"
            
            score += 1
            step += 1
            
            state_reshape = np.reshape(state,[1,state_size])

            Q_m = sess.run(Qpred_m, feed_dict={X:state_reshape, dropout: 1})

            if e > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_m)

            next_state, reward, done, _ = env.step(action)

            replay_buffer.append([state_reshape,action,reward,next_state,done,score])
            
            state = next_state

            # if episode % UPDATE_CYCLE == 0 and len(replay_buffer) > batch_size:
            if progress == "training":
                for sample in ran.sample(replay_buffer, batch_size):
                    state, action, reward, next_state, done, score = sample

                    Q_Global = sess.run(Qpred_m, feed_dict={X: state, dropout: 1})
                    
                    next_state = np.reshape(next_state,[1,state_size])                    
                    Q_m = sess.run(Qpred_m, feed_dict={X: next_state, dropout: 1})                    
                    Q_Global[0, action] = reward + disscore_factor * np.max(Q_m)

                    _, loss = sess.run([train, LossValue], feed_dict={X: state, Y: Q_Global, dropout:1})

            if done or score == 10000:
                episode += 1                
                print("Episode {:>5}/ reward:{:>5}/ recent n Game reward:{:>5.2f}/ memory length:{:>5}"
                      .format(episode, score, np.mean(last_n_game_reward),len(replay_buffer)),"/ Progress :",progress)

        last_n_game_reward.append(score)
        
        if len(last_n_game_reward) == last_n_game_reward.maxlen:
            avg_reward = np.mean(last_n_game_reward)
            if avg_reward < 200:
                print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                break

    save_path = saver.save(sess, model_path + "/model.ckpt")
    print("Model saved in file: ",save_path)