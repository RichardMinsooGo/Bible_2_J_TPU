import tensorflow as tf
import gym
import numpy as np
from collections import deque
import os

# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 1.	Import or generate datasets
env = gym.make('CartPole-v0')

# 2.	[Not for DQN] Transform and normalize data 
# 3.	[Not for DQN] Partition datasets into train, test, and validation sets

# 4.	(DQN) Constants defining our neural network
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

#7.	(option) e-greedy define
# minimum epsilon for epsilon greedy
MIN_E = 0.0
# epsilon will be `MIN_E` at `EPSILON_DECAYING_EPISODE`
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

# 손실 함수 정의
# LossValue function
LossValue = tf.reduce_sum(tf.square(Y - Qpred_m))
# Learning
train = tf.train.AdamOptimizer(learning_rate=Alpha).minimize(LossValue)

step_history = []

# Setting up our environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

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

    done = False
    count = 0

    # 17.3.	For t = 1, T do
    # 에피소드가 끝나기 전까지 반복 혹은 충분한 step 까지 반복
    # Execute each episode till finish or do it till sufficient steps(10000)
    while not done and count < 10000 :

        count += 1
        state_reshape = np.reshape(state, [1, INPUT_SIZE])
        # Choose an action by greedily (with e chance of random action) from
        # the Q-network
        Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_Global)

        #  action 수행 / Get new state and reward from environment
        nextstate, reward, done, _ = env.step(action)
        if done:
            # 에피소드가 끝났을때 Negative reward 부여
            Q_Global[0, action] = -100
        else:
            # next_state값의 전처리 후 Q-learning
            nextstate_reshape = np.reshape(nextstate, [1, INPUT_SIZE])
            # Obtain the Q' values by feeding the new state through our network
            Q_next = sess.run(Qpred_m, feed_dict={X: nextstate_reshape})
            Q_Global[0, action] = reward + Gamma * np.max(Q_next)

        # Train our network using target and predicted Q values on each episode
        sess.run(train, feed_dict={X: state_reshape, Y: Q_Global})
        state = nextstate

    step_history.append(count)
    if episode % PRINT_CYCLE == 0 :
        # print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, count, e))
        print("Episode {:>5} reward:{:>5} recent N Game reward:{:>5.2f} memory length:{:>5}"
                      .format(episode, count, np.mean(last_N_game_reward),len(replay_buffer)))
    # If last 10's avg steps are 500, it's good enough
    if len(step_history) > 100 and np.mean(step_history[-100:]) > 199:
        break
        
for episode in range(N_train_result_replay):
    # state 초기화
    state = env.reset()

    rall = 0
    done = False
    count = 0
    # 에피소드가 끝나기 전까지 반복
    while not done :
        env.render()
        count += 1
        # state 값의 전처리
        state_reshape = np.reshape(state, [1, INPUT_SIZE])

        # 현재 상태의 Q값을 에측
        Q_Global = sess.run(Qpred_m, feed_dict={X: state_reshape})
        action = np.argmax(Q_Global)

        # 결정된 action으로 Environment에 입력
        state, reward, done, _ = env.step(action)

        # 총 reward 합
        rall += reward

    rlist.append(rall)

    print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, count, rall,
                                                                    np.mean(rlist)))

        