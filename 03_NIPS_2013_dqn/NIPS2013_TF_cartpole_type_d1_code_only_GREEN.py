import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import dqn
from typing import List

env = gym.make('CartPole-v0')
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

Gamma = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
PRINT_CYCLE = 10

SIZE_R_M = 50000
MINIBATCH = 64

rlist=[]


MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

def train_minibatch(mainDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.INPUT_SIZE)
    y_stack = np.empty(0).reshape(0, mainDQN.OUTPUT_SIZE)

    for state, action, reward, nextstate, done in train_batch:
        Q_Global = mainDQN.predict(state)
        
        #terminal?
        if done:
            Q_Global[0,action] = reward
            
        else:
            #Obtain the Q' values by feeding the new state through our network
            Q_Global[0,action] = reward + Gamma * np.max(mainDQN.predict(nextstate))

        y_stack = np.vstack([y_stack, Q_Global])
        x_stack = np.vstack([x_stack, state])
    
    return mainDQN.update(x_stack, y_stack)

def main():
    last_N_game_reward = deque(maxlen=100)
    last_N_game_reward.append(0)
    replay_buffer = deque(maxlen=SIZE_R_M)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        init = tf.global_variables_initializer()
        sess.run(init)

        # 17.	Evaluate the model/ Start episode / 에피소드 시작
        # For episode = 1, M do
        # 에피소드 마지막까지 학습 시키거나 아니면 충분히 학습되는 조건을 정할수 있음


        for episode in range(N_EPISODES):        
            state = env.reset()
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            rall = 0
            done = False
            count = 0

            while not done and count < 10000 :
                count += 1
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                nextstate, reward, done, _ = env.step(action)
                if done:
                    reward = -100

                replay_buffer.append((state, action, reward, nextstate, done))

                if len(replay_buffer) > SIZE_R_M:
                    replay_buffer.popleft()
                state = nextstate
                if count > 10000:
                    break

            if count > 10000:
                pass
            #train every 10 episodes
            if episode % PRINT_CYCLE == 0 :
                for _ in range (MINIBATCH):
                    minibatch = random.sample(replay_buffer, 10)
                    train_minibatch(mainDQN, minibatch)
                print("Episode {:>5} reward:{:>5} recent N Game reward:{:>5.2f} memory length:{:>5}"
                      .format(episode, count, np.mean(last_N_game_reward),len(replay_buffer)))
            
            last_N_game_reward.append(count)

            if len(last_N_game_reward) == last_N_game_reward.maxlen:
                avg_reward = np.mean(last_N_game_reward)

                if avg_reward > 199.0:
                    print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                    break

        for episode in range(N_train_result_replay):            
            state = env.reset()
            rall = 0
            done = False
            count = 0
            
            while not done :
                env.render()
                count += 1
                Q_Global = mainDQN.predict(state)
                action = np.argmax(Q_Global)
                state, reward, done, _ = env.step(action)
                rall += reward

            rlist.append(rall)
            print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, count, rall, np.mean(rlist)))

if __name__ == "__main__":
    main()