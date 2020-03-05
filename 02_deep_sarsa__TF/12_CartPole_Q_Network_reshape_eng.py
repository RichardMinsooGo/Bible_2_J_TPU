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

Alpha = 0.001
Gamma = 0.99
N_EPISODES = 2000
N_train_result_replay = 20
H_SIZE_01 = 256

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

rlist=[0]
last_N_game_reward=[0]

episode = 0

LossValue = tf.reduce_sum(tf.square(Y-Qpred_m))
optimizer = tf.train.AdamOptimizer(Alpha, epsilon=0.01)
train = optimizer.minimize(LossValue)

# 13.	(option) Model Save path define and Initialization
LOG_DIR_Model = "/tmp/RL/02_Cartpole_Reshape/Model"
LOG_DIR_Graph = "/tmp/RL/02_Cartpole_Reshape/Graph"
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # if folders are not exist, crate model Save path
    if not os.path.exists(LOG_DIR_Model):
        os.makedirs(LOG_DIR_Model)
    if not os.path.exists(LOG_DIR_Graph):
        os.makedirs(LOG_DIR_Graph)
        
    last_N_game_reward = deque(maxlen=100)
    last_N_game_reward.append(0)    
    
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
            Q_Global = sess.run(Qpred_m, feed_dict={X:state_reshape, dropout: 1})  
            
            if e > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_Global)

            nextstate, reward, done, _ = env.step(action)     
            
            if done:
                Q_Global[0, action] = -100
            else:
                nextstate_reshape= np.reshape(nextstate,[1,INPUT_SIZE])
                Q_next = sess.run(Qpred_m, feed_dict={X: nextstate_reshape, dropout: 1})
                Q_Global[0, action] = reward + Gamma * np.max(Q_next)

            _, loss = sess.run([train, LossValue], feed_dict={X: state_reshape, Y: Q_Global, dropout: 1})
            
            rall += reward
            state = nextstate
        print("Episode {:>5} reward:{:>5} average reward:{:>5.2f} recent N Game reward:{:>5.2f} Loss:{:>5.2f}"
                  .format(episode, rall, np.mean(rlist), np.mean(last_N_game_reward),loss))
            
        last_N_game_reward.append(rall)
        rlist.append(rall)
        
        if len(last_N_game_reward) == last_N_game_reward.maxlen:
            avg_reward = np.mean(last_N_game_reward)
            if avg_reward > 199.0:
                print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                break

    # 18.	(option) Save the train result at the path that is defined at 13.
    save_path = saver.save(sess, LOG_DIR_Model + "/model.ckpt")
    print("Model saved in file: ",save_path)

    # rlist=[]
    # last_N_game_reward=[]

# 19.	Replay with training results.
# 19.1.	Session Initialization
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 19.2.	Restore the trained data
    saver.restore(sess, LOG_DIR_Model+ "/model.ckpt")

    print("Play Cartpole!")
    
    # 19.3.	Define and Initialize the total reward list
    rlist=[]
    
    # 19.4.	Replay the model
    # For episode = 1, N_train_result_replay do
    for episode in range(N_train_result_replay):
        
        # 19.4.1.	State initialization
        state = env.reset()
        rall = 0
        done = False
        count = 0
        
        # 19.4.2.	For t = 1, T do 
        #        Execute the epidose till terminal

        while not done :
            # Plotting
            env.render()
            count += 1
            # 19.4.3.	State reshape and Calulate Q value
            state_reshape = np.reshape(state, [1, INPUT_SIZE])
            Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape,dropout: 1})
            
            # 19.4.4.	Select a_t = argmax_a Q*(Pi(s_t), a ; 𝜃)
            action = np.argmax(Q_Global)

            # 19.4.5.	Execute action a_t in emulator and observe reward r_t and image x_t+1
            state, reward, done, _ = env.step(action)

            # total reward
            rall += reward

        # 19.4.6.	(option) Save total reward
        rlist.append(rall)

        print("Episode : {:>5} rewards ={:>5}. averge reward : {:>5.2f}".format(episode, rall,
                                                                        np.mean(rlist)))