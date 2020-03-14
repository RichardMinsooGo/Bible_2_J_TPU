import tensorflow as tf
import gym
import numpy as np
import random as ran

env = gym.make('CartPole-v0')

INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
Alpha = 0.001
Gamma = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
H_SIZE_01 = 512
UPDATE_CYCLE = 10
MAX_LENGTH_N_Game_Rewards = 100

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
    
    for episode in range(N_EPISODES):
        state = env.reset()
        
        if len(last_N_game_reward) > 100:
            del last_N_game_reward[0]

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
            
            if len(replay_buffer) > SIZE_R_M:
                del replay_buffer[0]
                
            rall += reward
            state = nextstate

        if episode % UPDATE_CYCLE == 0 and len(replay_buffer) > MINIBATCH:
            for sample in ran.sample(replay_buffer, MINIBATCH):

                state_R_M, action_R_M, reward_R_M, nextstate_R_M, done_R_M ,count_R_M = sample
                Q_Global = sess.run(Qpred_m, feed_dict={X: state_R_M, dropout: 1})
                if done_R_M:
                    if count_R_M < env.spec.timestep_limit :
                        Q_Global[0, action_R_M] = -100
                else:
                    nextstate_reshape_R_M = np.reshape(nextstate_R_M,[1,INPUT_SIZE])
                    Q_m = sess.run(Qpred_m, feed_dict={X: nextstate_reshape_R_M, dropout: 1})                    
                    Q_Global[0, action_R_M] = reward_R_M + Gamma * np.max(Q_m)
                    
                _, loss = sess.run([train, LossValue], feed_dict={X: state_R_M, Y: Q_Global, dropout:1})
                
            print("Episode {:>5} reward:{:>5} average reward:{:>5.2f} recent N Game reward:{:>5.2f} Loss:{:>5.2f} memory length:{:>5}"
                  .format(episode, rall, np.mean(rlist), np.mean(last_N_game_reward),loss,len(replay_buffer)))

        last_N_game_reward.append(rall)
        rlist.append(rall)
        
        # 17.6.	If episode is terminal condition, break the training.
        if len(last_N_game_reward) == MAX_LENGTH_N_Game_Rewards:
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
            Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape,dropout: 1})            
            action = np.argmax(Q_Global)
            state, reward, done, _ = env.step(action)
            rall += reward

        rlist.append(rall)
        print("Episode : {:>5} rewards ={:>5}. averge reward : {:>5.2f}".format(episode, rall,
                                                                        np.mean(rlist)))