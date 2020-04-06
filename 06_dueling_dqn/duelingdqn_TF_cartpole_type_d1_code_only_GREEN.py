import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import Duelingdqn
from typing import List

env = gym.make('CartPole-v0')
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

Gamma = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
TARGET_UPDATE_CYCLE = 10

SIZE_R_M = 50000
MINIBATCH = 64

rlist=[]


MIN_E = 0.0
EPSILON_DECAYING_EPISODE = N_EPISODES * 0.01

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)

def Copy_Weights(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder
                 
def train_minibatch(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.INPUT_SIZE)
    y_stack = np.empty(0).reshape(0, mainDQN.OUTPUT_SIZE)

    for state, action, reward, nextstate, done in train_batch:
        Q_Global = mainDQN.predict(state)
        
        #terminal?
        if done:
            Q_Global[0,action] = reward
            
        else:
            #Obtain the Q' values by feeding the new state through our network
            Q_Global[0,action] = reward + Gamma * np.max(targetDQN.predict(nextstate))

        y_stack = np.vstack([y_stack, Q_Global])
        x_stack = np.vstack([x_stack, state])
    
    return mainDQN.update(x_stack, y_stack)

def main():
    last_N_game_reward = deque(maxlen=100)
    last_N_game_reward.append(0)
    replay_buffer = deque(maxlen=SIZE_R_M)

    with tf.Session() as sess:
        mainDQN = Duelingdqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = Duelingdqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        init = tf.global_variables_initializer()
        sess.run(init)

        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

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
            if episode % TARGET_UPDATE_CYCLE ==0:
                for _ in range (MINIBATCH):
                    minibatch = random.sample(replay_buffer, 10)
                    LossValue,_ = train_minibatch(mainDQN,targetDQN, minibatch)
                print("LossValue : ",LossValue)
                sess.run(copy_ops)
                    
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