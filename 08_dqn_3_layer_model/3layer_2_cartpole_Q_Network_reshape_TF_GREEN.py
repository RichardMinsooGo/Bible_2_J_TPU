import tensorflow as tf
import gym
import numpy as np
from collections import deque
env = gym.make('CartPole-v0')
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

Alpha = 0.001
Gamma = 0.99
N_EPISODES = 2000
N_train_result_replay = 20
H_SIZE_01 = 512
H_SIZE_02 = 511
H_SIZE_03 = 510
PRINT_CYCLE = 10

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
W16_m = tf.get_variable('W16_m',shape=[H_SIZE_03, OUTPUT_SIZE]
                        ,initializer=tf.contrib.layers.xavier_initializer())

B01_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B02_m = tf.Variable(tf.zeros([1],dtype=tf.float32))
B03_m = tf.Variable(tf.zeros([1],dtype=tf.float32))

_LAY01_m = tf.nn.relu(tf.matmul(X,W01_m)+B01_m)
_LAY02_m = tf.nn.relu(tf.matmul(_LAY01_m,W02_m)+B02_m)
_LAY03_m = tf.nn.relu(tf.matmul(_LAY02_m,W03_m)+B03_m)
Qpred_m = tf.matmul(_LAY03_m,W16_m)

rlist=[0]
last_N_game_reward=[0]

episode = 0

LossValue = tf.reduce_sum(tf.square(Y-Qpred_m))
optimizer = tf.train.AdamOptimizer(Alpha, epsilon=0.01)
train = optimizer.minimize(LossValue)

model_path = "/tmp/RL/save/02_DQN_reshape/model.ckpt"
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
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

            # 현재 상태의 Q값을 예측
            Q_Global = sess.run(Qpred_m, feed_dict={X:state_reshape})

            # Action decision with e-greedy policy
            if e > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_Global
                
            nextstate, reward, done, _ = env.step(action)
            
            if done:
                Q_Global[0, action] = -100
            else:
                nextstate_reshape= np.reshape(nextstate,[1,INPUT_SIZE])
                Q_next = sess.run(Qpred_m, feed_dict={X: nextstate_reshape})
                Q_Global[0, action] = reward + Gamma * np.max(Q_next)

            sess.run(train, feed_dict={X: state_reshape, Y: Q_Global})
            
            state = nextstate
            rall += reward

        if episode % PRINT_CYCLE == 0:
            print("Episode {:>5} reward:{:>5} average reward:{:>5.2f} recent N Game reward:{:>5.2f}"
                      .format(episode, rall, np.mean(rlist), np.mean(last_N_game_reward)))
            
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