import tensorflow as tf
import gym
import numpy as np
import random as ran
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.framework import ops
ops.reset_default_graph()

env = gym.make('CartPole-v0')

INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

Alpha = 0.001
Gamma = 0.99
N_EPISODES = 2000
N_train_result_replay = 20
H_SIZE_01 = 256
TARGET_UPDATE_CYCLE = 10

replay_buffer = []
SIZE_R_M = 50000
MINIBATCH = 64

MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01
def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

X=tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE), name="input_X")
Y=tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_SIZE), name="output_Y")
dropout = tf.placeholder(dtype=tf.float32)

W01_m = tf.get_variable('W01_m',shape=[INPUT_SIZE, H_SIZE_01]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W16_m = tf.get_variable('W16_m',shape=[H_SIZE_01, OUTPUT_SIZE]
                        ,initializer=tf.contrib.layers.xavier_initializer())

B01_m = tf.Variable(tf.zeros([1],dtype=tf.float32))

_LAY01_m = tf.nn.relu(tf.matmul(X,W01_m)+B01_m)
LAY01_m = tf.nn.dropout(_LAY01_m,dropout)
Qpred_m = tf.matmul(LAY01_m,W16_m)

W01_t = tf.get_variable('W01_t',shape=[INPUT_SIZE, H_SIZE_01])
W16_t = tf.get_variable('W16_t',shape=[H_SIZE_01, OUTPUT_SIZE])

B01_t = tf.Variable(tf.zeros([1],dtype=tf.float32))

LAY01_t = tf.nn.relu(tf.matmul(X ,W01_t)+B01_t)
Qpred_t = tf.matmul(LAY01_t,W16_t)

rlist=[0]
last_N_game_reward=[0]

episode = 0

LossValue = tf.reduce_sum(tf.square(Y-Qpred_m))
optimizer = tf.train.AdamOptimizer(Alpha, epsilon=0.01)
train = optimizer.minimize(LossValue)

LOG_DIR_Model = "/tmp/RL/05_DoubleDQN/Model"
LOG_DIR_Graph = "/tmp/RL/05_DoubleDQN/Graph"
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    sess.run(W01_t.assign(W01_m))
    sess.run(W16_t.assign(W16_m))
    sess.run(B01_t.assign(B01_m))

    if not os.path.exists(LOG_DIR_Model):
        os.makedirs(LOG_DIR_Model)
    if not os.path.exists(LOG_DIR_Graph):
        os.makedirs(LOG_DIR_Graph)

    last_N_game_reward = deque(maxlen=100)
    last_N_game_reward.append(0)
    replay_buffer = deque(maxlen=SIZE_R_M)
    
    for episode in range(N_EPISODES):
        
        state = env.reset()

        e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)

        rall = 0
        done = False
        count = 0

        while not done and count < 10000 :

            #env.render()
            count += 1

            state_reshape = np.reshape(state,[1,INPUT_SIZE])

            Q_m = sess.run(Qpred_m, feed_dict={X:state_reshape, dropout: 1})

            if e > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_m)

            nextstate, reward, done, _ = env.step(action)

            replay_buffer.append([state_reshape,action,reward,nextstate,done,count])
            
            rall += reward
            state = nextstate

        # if episode % TARGET_UPDATE_CYCLE == 0 and len(replay_buffer) > MINIBATCH:
        if (episode+1) % TARGET_UPDATE_CYCLE == 0 and len(replay_buffer) > MINIBATCH:

            for sample in ran.sample(replay_buffer, MINIBATCH):

                state_R_M, action_R_M, reward_R_M, nextstate_R_M, done_R_M ,count_R_M = sample

                Q_Global = sess.run(Qpred_m, feed_dict={X: state_R_M, dropout: 1})

                if done_R_M:
                    if count_R_M < env.spec.timestep_limit :
                        Q_Global[0, action_R_M] = -100
                else:
                    nextstate_reshape_R_M = np.reshape(nextstate_R_M,[1,INPUT_SIZE])
                    Q_target, Q_m = sess.run([Qpred_t,Qpred_m], feed_dict={X: nextstate_reshape_R_M, dropout:1})
                    Q_Global[0, action_R_M] = reward_R_M + Gamma * Q_target[0, np.argmax(Q_m)]

                _, loss = sess.run([train, LossValue], feed_dict={X: state_R_M, Y: Q_Global, dropout:1})

            sess.run(W01_t.assign(W01_m))
            sess.run(W16_t.assign(W16_m))
            sess.run(B01_t.assign(B01_m))

            print("Episode {:>5} reward:{:>5} average reward:{:>5.2f} recent N Game reward:{:>5.2f} Loss:{:>5.2f} memory length:{:>5}"
                  .format(episode, rall, np.mean(rlist), np.mean(last_N_game_reward),loss,len(replay_buffer)))

        last_N_game_reward.append(rall)
        rlist.append(rall)
        
        if len(last_N_game_reward) == last_N_game_reward.maxlen:
            avg_reward = np.mean(last_N_game_reward)
            if avg_reward > 199.0:
                print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                break

    save_path = saver.save(sess, LOG_DIR_Model + "/model.ckpt")
    print("Model saved in file: ",save_path)
    
# Replay the result
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, LOG_DIR_Model+ "/model.ckpt")
    print("Play Cartpole!")
    
    rlist=[]
    
    for episode in range(N_train_result_replay):
        
        state = env.reset()
        rall = 0
        done = False
        count = 0
        
        while not done :
            env.render()
            count += 1
            state_reshape = np.reshape(state, [1, INPUT_SIZE])
            Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape,dropout: 1})            
            action = np.argmax(Q_Global)
            state, reward, done, _ = env.step(action)
            rall += reward
            
        rlist.append(rall)
        print("Episode : {:>5} rewards ={:>5}. averge reward : {:>5.2f}".format(episode, rall,
                                                                        np.mean(rlist)))
    sess.close()
