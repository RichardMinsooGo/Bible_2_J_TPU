import tensorflow as tf
import gym
import numpy as np
import random as ran
from collections import deque

env = gym.make('CartPole-v0')
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
Alpha = 0.001
Gamma = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
H_SIZE_01 = 512
H_SIZE_02 = 511
H_SIZE_03 = 510
H_SIZE_04 = 511
H_SIZE_05 = 512

H_SIZE_06 = 513
H_SIZE_07 = 514
H_SIZE_08 = 515
H_SIZE_09 = 516
H_SIZE_10 = 517

H_SIZE_11 = 518
H_SIZE_12 = 519
H_SIZE_13 = 518
H_SIZE_14 = 517
H_SIZE_15 = 516
TARGET_UPDATE_CYCLE = 10

# [Dueling DQN] Define Parameters for state-net and action-net
H_SIZE_15_state = 256
H_SIZE_15_action = 256
H_SIZE_16_state = OUTPUT_SIZE
H_SIZE_16_action = OUTPUT_SIZE

# 6.	[DQN] Initialize replay memory D to capacity N
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

W01_m = tf.get_variable('W01_m',shape=[INPUT_SIZE, H_SIZE_01]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W02_m = tf.get_variable('W02_m',shape=[H_SIZE_01, H_SIZE_02]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W03_m = tf.get_variable('W03_m',shape=[H_SIZE_02, H_SIZE_03]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W04_m = tf.get_variable('W04_m',shape=[H_SIZE_03, H_SIZE_04]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W05_m = tf.get_variable('W05_m',shape=[H_SIZE_04, H_SIZE_05]
                        ,initializer=tf.contrib.layers.xavier_initializer())

W06_m = tf.get_variable('W06_m',shape=[H_SIZE_05, H_SIZE_06]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W07_m = tf.get_variable('W07_m',shape=[H_SIZE_06, H_SIZE_07]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W08_m = tf.get_variable('W08_m',shape=[H_SIZE_07, H_SIZE_08]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W09_m = tf.get_variable('W09_m',shape=[H_SIZE_08, H_SIZE_09]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W10_m = tf.get_variable('W10_m',shape=[H_SIZE_09, H_SIZE_10]
                        ,initializer=tf.contrib.layers.xavier_initializer())

W11_m = tf.get_variable('W11_m',shape=[H_SIZE_10, H_SIZE_11]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W12_m = tf.get_variable('W12_m',shape=[H_SIZE_11, H_SIZE_12]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W13_m = tf.get_variable('W13_m',shape=[H_SIZE_12, H_SIZE_13]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W14_m = tf.get_variable('W14_m',shape=[H_SIZE_13, H_SIZE_14]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W15_m_state = tf.get_variable('W15_m_state',shape=[H_SIZE_14, H_SIZE_15_state]
                              ,initializer=tf.contrib.layers.xavier_initializer())
W15_m_action = tf.get_variable('W15_m_action',shape=[H_SIZE_14, H_SIZE_15_action]
                               ,initializer=tf.contrib.layers.xavier_initializer())
W16_m_state = tf.get_variable('W16_m_state',shape=[H_SIZE_15_state, H_SIZE_16_state]
                              ,initializer=tf.contrib.layers.xavier_initializer())
W16_m_action = tf.get_variable('W16_m_action',shape=[H_SIZE_15_action, H_SIZE_16_action]
                               ,initializer=tf.contrib.layers.xavier_initializer())

B01_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B02_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B03_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B04_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B05_m = tf.Variable(tf.zeros([1],dtype=tf.float32))

B06_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B07_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B08_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B09_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B10_m = tf.Variable(tf.zeros([1],dtype=tf.float32))

B11_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B12_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B13_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B14_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_m_state = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_m_action = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_m_state = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_m_action = tf.Variable(tf.zeros([1],dtype=tf.float32))

_LAY01_m = tf.nn.relu(tf.matmul(X,W01_m)+B01_m)
_LAY02_m = tf.nn.relu(tf.matmul(_LAY01_m,W02_m)+B02_m)
_LAY03_m = tf.nn.relu(tf.matmul(_LAY02_m,W03_m)+B03_m)
_LAY04_m = tf.nn.relu(tf.matmul(_LAY03_m,W04_m)+B04_m)
_LAY05_m = tf.nn.relu(tf.matmul(_LAY04_m,W05_m)+B05_m)

_LAY06_m = tf.nn.relu(tf.matmul(_LAY05_m,W06_m)+B06_m)
_LAY07_m = tf.nn.relu(tf.matmul(_LAY06_m,W07_m)+B07_m)
_LAY08_m = tf.nn.relu(tf.matmul(_LAY07_m,W08_m)+B08_m)
_LAY09_m = tf.nn.relu(tf.matmul(_LAY08_m,W09_m)+B09_m)
_LAY10_m = tf.nn.relu(tf.matmul(_LAY09_m,W10_m)+B10_m)

_LAY11_m = tf.nn.relu(tf.matmul(_LAY10_m,W11_m)+B11_m)
_LAY12_m = tf.nn.relu(tf.matmul(_LAY11_m,W12_m)+B12_m)
_LAY13_m = tf.nn.relu(tf.matmul(_LAY12_m,W13_m)+B13_m)
_LAY14_m = tf.nn.relu(tf.matmul(_LAY13_m,W14_m)+B14_m)

_LAY15_m_state = tf.nn.relu(tf.matmul(_LAY14_m,W15_m_state)+B15_m_state)
_LAY15_m_action = tf.nn.relu(tf.matmul(_LAY14_m,W15_m_action)+B15_m_action)

_LAY16_m_state = tf.matmul(_LAY15_m_state,W16_m_state) + B16_m_state
_LAY16_m_action = tf.matmul(_LAY15_m_action,W16_m_action) + B16_m_action

LAY16_m_advantage = tf.subtract(_LAY16_m_action, tf.reduce_mean(_LAY16_m_action))

Qpred_m = tf.add(_LAY16_m_state, LAY16_m_advantage)

W01_t = tf.get_variable('W01_t',shape=[INPUT_SIZE, H_SIZE_01])
W02_t = tf.get_variable('W02_t',shape=[H_SIZE_01, H_SIZE_02])
W03_t = tf.get_variable('W03_t',shape=[H_SIZE_02, H_SIZE_03])
W04_t = tf.get_variable('W04_t',shape=[H_SIZE_03, H_SIZE_04])
W05_t = tf.get_variable('W05_t',shape=[H_SIZE_04, H_SIZE_05])

W06_t = tf.get_variable('W06_t',shape=[H_SIZE_05, H_SIZE_06])
W07_t = tf.get_variable('W07_t',shape=[H_SIZE_06, H_SIZE_07])
W08_t = tf.get_variable('W08_t',shape=[H_SIZE_07, H_SIZE_08])
W09_t = tf.get_variable('W09_t',shape=[H_SIZE_08, H_SIZE_09])
W10_t = tf.get_variable('W10_t',shape=[H_SIZE_09, H_SIZE_10])

W11_t = tf.get_variable('W11_t',shape=[H_SIZE_10, H_SIZE_11])
W12_t = tf.get_variable('W12_t',shape=[H_SIZE_11, H_SIZE_12])
W13_t = tf.get_variable('W13_t',shape=[H_SIZE_12, H_SIZE_13])
W14_t = tf.get_variable('W14_t',shape=[H_SIZE_13, H_SIZE_14])
W15_t_state = tf.get_variable('W15_t_state',shape=[H_SIZE_14, H_SIZE_15_state])
W15_t_action = tf.get_variable('W15_t_action',shape=[H_SIZE_14, H_SIZE_15_action])
W16_t_state = tf.get_variable('W16_t_state',shape=[H_SIZE_15_state, H_SIZE_16_state])
W16_t_action = tf.get_variable('W16_t_action',shape=[H_SIZE_15_action, H_SIZE_16_action])

B01_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B02_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B03_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B04_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B05_t = tf.Variable(tf.zeros([1],dtype=tf.float32))

B06_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B07_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B08_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B09_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B10_t = tf.Variable(tf.zeros([1],dtype=tf.float32))

B11_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B12_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B13_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B14_t = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_t_state = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_t_action = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_t_state = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_t_action = tf.Variable(tf.zeros([1],dtype=tf.float32))

LAY01_t = tf.nn.relu(tf.matmul(X ,W01_t)+B01_t)
LAY02_t = tf.nn.relu(tf.matmul(LAY01_t ,W02_t)+B02_t)
LAY03_t = tf.nn.relu(tf.matmul(LAY02_t ,W03_t)+B03_t)
LAY04_t = tf.nn.relu(tf.matmul(LAY03_t ,W04_t)+B04_t)
LAY05_t = tf.nn.relu(tf.matmul(LAY04_t ,W05_t)+B05_t)

LAY06_t = tf.nn.relu(tf.matmul(LAY05_t ,W06_t)+B06_t)
LAY07_t = tf.nn.relu(tf.matmul(LAY06_t ,W07_t)+B07_t)
LAY08_t = tf.nn.relu(tf.matmul(LAY07_t ,W08_t)+B08_t)
LAY09_t = tf.nn.relu(tf.matmul(LAY08_t ,W09_t)+B09_t)
LAY10_t = tf.nn.relu(tf.matmul(LAY09_t ,W10_t)+B10_t)

LAY11_t = tf.nn.relu(tf.matmul(LAY10_t ,W11_t)+B11_t)
LAY12_t = tf.nn.relu(tf.matmul(LAY11_t ,W12_t)+B12_t)
LAY13_t = tf.nn.relu(tf.matmul(LAY12_t ,W13_t)+B13_t)
LAY14_t = tf.nn.relu(tf.matmul(LAY13_t ,W14_t)+B14_t)
LAY15_t_state = tf.nn.relu(tf.matmul(LAY14_t,W15_t_state)+B15_t_state)
LAY15_t_action = tf.nn.relu(tf.matmul(LAY14_t,W15_t_action)+B15_t_action)

LAY16_t_state = tf.matmul(LAY15_t_state,W16_t_state) + B16_t_state
LAY16_t_action = tf.matmul(LAY15_t_action,W16_t_action) + B16_t_action

LAY16_t_advantage = tf.subtract(LAY16_t_action, tf.reduce_mean(LAY16_t_action))

Qpred_t = tf.add(LAY16_t_state, LAY16_t_advantage)

rlist=[0]
last_N_game_reward=[0]

episode = 0

LossValue = tf.reduce_sum(tf.square(Y-Qpred_m))
optimizer = tf.train.AdamOptimizer(Alpha, epsilon=0.01)
train = optimizer.minimize(LossValue)

model_path = "/tmp/RL/save/06_DuelingDQN/model.ckpt"
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    sess.run(W01_t.assign(W01_m))
    sess.run(W02_t.assign(W02_m))
    sess.run(W03_t.assign(W03_m))
    sess.run(W04_t.assign(W04_m))
    sess.run(W05_t.assign(W05_m))

    sess.run(W06_t.assign(W06_m))
    sess.run(W07_t.assign(W07_m))
    sess.run(W08_t.assign(W08_m))
    sess.run(W09_t.assign(W09_m))
    sess.run(W10_t.assign(W10_m))
    
    sess.run(W11_t.assign(W11_m))
    sess.run(W12_t.assign(W12_m))
    sess.run(W13_t.assign(W13_m))
    sess.run(W14_t.assign(W14_m))
    sess.run(W15_t_state.assign(W15_m_state))
    sess.run(W15_t_action.assign(W15_m_action))
    sess.run(W16_t_state.assign(W16_m_state))
    sess.run(W16_t_action.assign(W16_m_action))
    
    sess.run(B01_t.assign(B01_m))
    sess.run(B02_t.assign(B02_m))
    sess.run(B03_t.assign(B03_m))
    sess.run(B04_t.assign(B04_m))
    sess.run(B05_t.assign(B05_m))
    
    sess.run(B06_t.assign(B06_m))
    sess.run(B07_t.assign(B07_m))
    sess.run(B08_t.assign(B08_m))
    sess.run(B09_t.assign(B09_m))
    sess.run(B10_t.assign(B10_m))
    
    sess.run(B11_t.assign(B11_m))
    sess.run(B12_t.assign(B12_m))
    sess.run(B13_t.assign(B13_m))
    sess.run(B14_t.assign(B14_m))
    sess.run(B15_t_state.assign(B15_m_state))
    sess.run(B15_t_action.assign(B15_m_action))
    sess.run(B16_t_state.assign(B16_m_state))
    sess.run(B16_t_action.assign(B16_m_action))
        

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
            Q_m = sess.run(Qpred_m, feed_dict={X:state_reshape})

            if e > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_m)

            nextstate, reward, done, _ = env.step(action)
            replay_buffer.append([state_reshape,action,reward,nextstate,done,count])
            rall += reward
            state = nextstate

        if episode % TARGET_UPDATE_CYCLE == 0 and len(replay_buffer) > MINIBATCH:

            for sample in ran.sample(replay_buffer, MINIBATCH):

                state_R_M, action_R_M, reward_R_M, nextstate_R_M, done_R_M ,count_R_M = sample
                Q_Global = sess.run(Qpred_m, feed_dict={X: state_R_M})

                if done_R_M:
                    if count_R_M < env.spec.timestep_limit :
                        Q_Global[0, action_R_M] = -100
                else:
                    nextstate_reshape_R_M = np.reshape(nextstate_R_M,[1,INPUT_SIZE])
                    Q_target = sess.run(Qpred_t, feed_dict={X: nextstate_reshape_R_M})
                    Q_Global[0, action_R_M] = reward_R_M + Gamma * np.max(Q_target)
                
                _, loss = sess.run([train, LossValue], feed_dict={X: state_R_M, Y: Q_Global})

            sess.run(W01_t.assign(W01_m))
            sess.run(W02_t.assign(W02_m))
            sess.run(W03_t.assign(W03_m))
            sess.run(W04_t.assign(W04_m))
            sess.run(W05_t.assign(W05_m))

            sess.run(W06_t.assign(W06_m))
            sess.run(W07_t.assign(W07_m))
            sess.run(W08_t.assign(W08_m))
            sess.run(W09_t.assign(W09_m))
            sess.run(W10_t.assign(W10_m))

            sess.run(W11_t.assign(W11_m))
            sess.run(W12_t.assign(W12_m))
            sess.run(W13_t.assign(W13_m))
            sess.run(W14_t.assign(W14_m))
            sess.run(W15_t_state.assign(W15_m_state))
            sess.run(W15_t_action.assign(W15_m_action))
            sess.run(W16_t_state.assign(W16_m_state))
            sess.run(W16_t_action.assign(W16_m_action))
            sess.run(B01_t.assign(B01_m))
            sess.run(B02_t.assign(B02_m))
            sess.run(B03_t.assign(B03_m))
            sess.run(B04_t.assign(B04_m))
            sess.run(B05_t.assign(B05_m))

            sess.run(B06_t.assign(B06_m))
            sess.run(B07_t.assign(B07_m))
            sess.run(B08_t.assign(B08_m))
            sess.run(B09_t.assign(B09_m))
            sess.run(B10_t.assign(B10_m))

            sess.run(B11_t.assign(B11_m))
            sess.run(B12_t.assign(B12_m))
            sess.run(B13_t.assign(B13_m))
            sess.run(B14_t.assign(B14_m))
            sess.run(B15_t_state.assign(B15_m_state))
            sess.run(B15_t_action.assign(B15_m_action))
            sess.run(B16_t_state.assign(B16_m_state))
            sess.run(B16_t_action.assign(B16_m_action))            
            
            print("Episode {:>5} reward:{:>5} average reward:{:>5.2f} recent N Game reward:{:>5.2f} Loss:{:>5.2f} memory length:{:>5}"
                  .format(episode, rall, np.mean(rlist), np.mean(last_N_game_reward),loss,len(replay_buffer)))

        last_N_game_reward.append(rall)
        rlist.append(rall)
        
        if len(last_N_game_reward) == last_N_game_reward.maxlen:
            avg_reward = np.mean(last_N_game_reward)
            if avg_reward > 199.0:
                print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                break

    save_path = saver.save(sess, model_path)
    print("Model saved in file: ",save_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
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
            Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
            action = np.argmax(Q_Global)
            state, reward, done, _ = env.step(action)
            rall += reward

        rlist.append(rall)
        print("Episode : {:>5} rewards ={:>5}. averge reward : {:>5.2f}".format(episode, rall,
                                                                        np.mean(rlist)))