import tensorflow as tf
import gym
import numpy as np
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 1.	Import or generate datasets
env = gym.make('CartPole-v0')
# 4.	(DQN) Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

# 5.	Define algorithm parameters (hyperparameters)
# Learning Rate = Alpha
Alpha = 0.001
# Discount Factor = Gamma
Gamma = 0.99
N_EPISODES = 200
N_train_result_replay = 20
# Hidden Layer 01 Size 
H_SIZE_01 = 256

#7.	(option) e-greedy define
# minimum epsilon for epsilon greedy
MIN_E = 0.0
# epsilon will be `MIN_E` at `EPSILON_DECAYING_EPISODE`
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

# 8.	Initialize variables and placeholders = Network Initializations
# ex). x_input = tf.placeholder(tf.float32, [None, input_size])
# ex). y_input = tf.placeholder(tf.float32, [None, num_classes])
X=tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE), name="input_X")
Y=tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_SIZE), name="output_Y")
dropout = tf.placeholder(dtype=tf.float32)

# 9.	Define the model structure – Main Network
W01_m = tf.get_variable('W01_m',shape=[INPUT_SIZE, H_SIZE_01]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W16_m = tf.get_variable('W16_m',shape=[H_SIZE_01, OUTPUT_SIZE]
                        ,initializer=tf.contrib.layers.xavier_initializer())

B01_m = tf.Variable(tf.zeros([1],dtype=tf.float32))

_LAY01_m = tf.nn.relu(tf.matmul(X,W01_m)+B01_m)
LAY01_m = tf.nn.dropout(_LAY01_m,dropout)
Qpred_m = tf.matmul(LAY01_m,W16_m)

# 11.	총 reward를 저장해놓을 리스트, 최근 N 게임의 reward를 저장할 리스트
# Define and Initialize the total reward list and last N game reward
rlist=[0]
last_N_game_reward=[0]

episode = 0

# 12.	Define the loss functions
LossValue = tf.reduce_sum(tf.square(Y-Qpred_m))
optimizer = tf.train.AdamOptimizer(Alpha, epsilon=0.01)
train = optimizer.minimize(LossValue)

# 14.	Initialize and train the model
# Initialize action-value function Q with random weight 𝜃
# 세션 정의 및 변수 초기화/ Session define and variable initialization

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Variable initialization
    sess.run(init)
    
        
    # 16. (option) Define the size of last N game reward and initialization / 지난 N game reward의 사이즈 지정 및 초기화
    # if the reward is sufficiently trained, with this valiable, make a exit condition/ 지난 N game에서 충분한 학습이되면 완료시키기 위함

    last_N_game_reward = deque(maxlen=100)
    last_N_game_reward.append(0)    
    
    # 17.	Evaluate the model/ Start episode / 에피소드 시작
    # For episode = 1, M do
    # Or can define the sufficiently trained condition using “while”.
    # 에피소드 마지막까지 학습 시키거나 아니면 충분히 학습되는 조건을 정할수 있음
    # episode의 마지막까지 학슬을 위해서 사용. Use for loop to train till the end of episodes.
    
    for episode in range(N_EPISODES):
        
    # 혹은 while loop를 사용하여 학습이 될때까지 지정할수도 있음. 복잡한 보델의 경우 학습의 target 을 정하기 어려우므로 추천하지는 않음
    # Or you can use while loop till it is trained. But it is not recommended for the high complex models.
    # while np.mean(last_N_game_reward) < 195 :
        #episode += 1

        # 17.1. State initialization
        #    Initialize sequence s1 = {x1} and preprocessed sequence Pi1 = Pi(s1)
        state = env.reset()

        # 17.2.	e-greedy
        # e-greedy option 1
        # e = 1. / ((episode/50)+10)
        
        # e-greedy option 2
        e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)

        rall = 0
        done = False
        count = 0
        s_t = sess.run(tf.expand_dims(state, axis=0))
        # 17.3.	For t = 1, T do
        # 에피소드가 끝나기 전까지 반복 혹은 충분한 step 까지 반복
        # Execute each episode till finish or do it till sufficient steps(10000)
        while not done and count < 10000 :

            #env.render()
            count += 1

            # state 값의 전처리

            # 현재 상태의 Q값을 예측
            Q_Global = sess.run(Qpred_m, feed_dict={X:s_t, dropout: 1})

            # 17.3.1.	 With probability e select a random action a_t, otherwise select a_t = argmax_a Q*(Pi(s_t), a ; 𝜃)
            # e-greedy 정책으로 랜덤하게 action 결정
            # Action decision with e-greedy policy
            if e > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_Global)

            # 17.3.2.	Execute action a_t in emulator and observe reward r_t and image x_t+1
            # 17.3.3.	Set s_t+1 = s_t, a_t, x_t+1 and preprocess Pi_t+1 = Pi(s_t+1)
            # 결정된 action으로 Environment에 입력
            # with decided action exucute in emulator and observe reward ... , set ...
            nextstate, reward, done, _ = env.step(action)
            
            # 17.3.6.	Set y_j = r_j                                   for terminal Pi_j+1
            #           y_j = r_j + gamma * max_a' Q^(Pi_j+1, a'; 𝜃_main)   for non-terminal Pi_j+1


            if done:
                # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                # If the status of replay sample is terminal, assign negative reward
                Q_Global[0, action] = -100
            else:
                # 끝나지 않았다면 Q값을 업데이트
                # If the status of replay sample is non-terminal, update Q value
                nextstate = sess.run(tf.expand_dims(nextstate, axis=0))
                # Obtain the Q' values by feeding the new state through our network
                Q_next = sess.run(Qpred_m, feed_dict={X: nextstate, dropout: 1})
                Q_Global[0, action] = reward + Gamma * np.max(Q_next)

            # 17.3.7.	Perform a gradient descent step on (y_j - Q(Pi_j, a_j; 𝜃))2
            # 업데이트 된 Q값으로 main네트워크를 학습
            # Train main network with updated Q value 
            _, loss = sess.run([train, LossValue], feed_dict={X: s_t, Y: Q_Global, dropout: 1})
            
            rall += reward
            s_t = nextstate

        print("Episode {:>5} reward:{:>5} average reward:{:>5.2f} recent N Game reward:{:>5.2f} Loss:{:>5.2f}"
                  .format(episode, rall, np.mean(rlist), np.mean(last_N_game_reward),loss))
            
        # (option) 17.5.	Save total reward and last N game rewards.
        # 총 reward의 합을 list에 저장하고 이를 또한 지난 N game list에 저장
        # Save total rewards at reward list and save it at the last N game list
        last_N_game_reward.append(rall)
        rlist.append(rall)
        
        # 17.6.	If episode is terminal condition, break the training.
        if len(last_N_game_reward) == last_N_game_reward.maxlen:
            avg_reward = np.mean(last_N_game_reward)
            if avg_reward > 199.0:
                print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                break
