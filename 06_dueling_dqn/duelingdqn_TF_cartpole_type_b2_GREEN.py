"""
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(nextstate, action; 𝜃_target))
    LossValue: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃

"""
import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import Duelingdqn
from typing import List
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

# Discount Factor = Gamma
Gamma = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
TARGET_UPDATE_CYCLE = 10

# 6.	[DQN] Initialize replay memory D to capacity N
SIZE_R_M = 50000
MINIBATCH = 64

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

def Copy_Weights(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`

    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)

    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder
                 
def train_minibatch(mainDQN: Duelingdqn.DQN, targetDQN: Duelingdqn.DQN, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`

    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained
        targetDQN (dqn.DQN): Target DQN that will predict Q_Global
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, nextstate, done), ...]

    Returns:
        float: After updating `mainDQN`, it returns a `LossValue`
    """
    state_array = np.vstack([x[0] for x in train_batch])
    action_array = np.array([x[1] for x in train_batch])
    reward_array = np.array([x[2] for x in train_batch])
    nextstate_array = np.vstack([x[3] for x in train_batch])
    done_array = np.array([x[4] for x in train_batch])

    X_batch = state_array
    Y_batch = mainDQN.predict(state_array)

    Q_Global = reward_array + Gamma * np.max(targetDQN.predict(nextstate_array), axis=1) * ~done_array

    Y_batch[np.arange(len(X_batch)), action_array] = Q_Global

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X_batch, Y_batch)

def main():
    
    # 16.	Define replay buffer size / Replay buffer 사이즈 지정
    # (option) Define the size of last N game reward and initialization / 지난 N game reward의 사이즈 지정 및 초기화
    # if the reward is sufficiently trained, with this valiable, make a exit condition/ 지난 N game에서 충분한 학습이되면 완료시키기 위함

    last_N_game_reward = deque(maxlen=100)
    last_N_game_reward.append(0)
    replay_buffer = deque(maxlen=SIZE_R_M)
    
    with tf.Session() as sess:
        mainDQN = Duelingdqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = Duelingdqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        init = tf.global_variables_initializer()
        sess.run(init)

        # initial copy q_net -> target_net
        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

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

            # 17.3.	For t = 1, T do
            # 에피소드가 끝나기 전까지 반복 혹은 충분한 step 까지 반복
            # Execute each episode till finish or do it till sufficient steps(10000)
            while not done and count < 10000 :
                count += 1
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # 17.3.2.	Execute action a_t in emulator and observe reward r_t and image x_t+1
                # 17.3.3.	Set s_t+1 = s_t, a_t, x_t+1 and preprocess Pi_t+1 = Pi(s_t+1)
                # 결정된 action으로 Environment에 입력
                # with decided action exucute in emulator and observe reward ... , set ...
                nextstate, reward, done, _ = env.step(action)

                if done:  # Penalty
                    reward = -100

                #Save the experience to our buffer
                replay_buffer.append((state, action, reward, nextstate, done))

                if len(replay_buffer) > SIZE_R_M:
                    replay_buffer.popleft()
                    
                    #minibatch = random.sample(replay_buffer, MINIBATCH)
                    #LossValue, _ = train_minibatch(mainDQN, targetDQN, minibatch)

#                if count % TARGET_UPDATE_CYCLE == 0:
#                    sess.run(copy_ops)
                state = nextstate
                #count += 1
        
            #train every 10 episodes
            if episode % TARGET_UPDATE_CYCLE ==0:
                #Get a random batch of experiences.
                for _ in range (MINIBATCH):
                    #Minibatch works better                
                    minibatch = random.sample(replay_buffer, 10)
                    LossValue,_ = train_minibatch(mainDQN,targetDQN, minibatch)
                print("LossValue : ",LossValue)
                # copy q_net --> target_net
                sess.run(copy_ops)
                    
#            if episode % TARGET_UPDATE_CYCLE == 0 :
                # print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, count, e))
                print("Episode {:>5} reward:{:>5} recent N Game reward:{:>5.2f} memory length:{:>5}"
                      .format(episode, count, np.mean(last_N_game_reward),len(replay_buffer)))
            
            # CartPole-v0 Game Clear Checking Logic
            last_N_game_reward.append(count)

            if len(last_N_game_reward) == last_N_game_reward.maxlen:
                avg_reward = np.mean(last_N_game_reward)

                if avg_reward > 199.0:
                    print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                    break

        # 19.	Replay with training results.
        # 19.1.	Session Initialization
        
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
                # 19.4.4.	Select a_t = argmax_a Q*(Pi(s_t), a ; 𝜃)
                Q_Global = mainDQN.predict(state)
                action = np.argmax(Q_Global)

                # 19.4.5.	Execute action a_t in emulator and observe reward r_t and image x_t+1
                state, reward, done, _ = env.step(action)

                # 총 reward 합
                rall += reward

            rlist.append(rall)

            print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, count, rall, np.mean(rlist)))

if __name__ == "__main__":
    main()