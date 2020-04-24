import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import time
import dqn
from typing import List
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.framework import ops
ops.reset_default_graph()

env_name = "MountainCar-v0"
env = gym.make(env_name)
# env.seed(1)     # reproducible, general Policy gradient has high variance
# np.random.seed(123)
# tf.set_random_seed(456)  # reproducible
env = env.unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
discount_factor = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
target_update_cycle = 200
SIZE_R_M = 50000
batch_size = 32

scores = []
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
                 
def train_model(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.state_size)
    y_stack = np.empty(0).reshape(0, mainDQN.action_size)

    for state, action, reward, nextstate, done in train_batch:
        Q_Global = mainDQN.predict(state)
        
        #terminal?
        if done:
            Q_Global[0,action] = reward
            
        else:
            #Obtain the Q' values by feeding the new state through our network
            Q_Global[0,action] = reward + discount_factor * np.max(targetDQN.predict(nextstate))

        y_stack = np.vstack([y_stack, Q_Global])
        x_stack = np.vstack([x_stack, state])
    
    return mainDQN.update(x_stack, y_stack)

def main():
    last_n_game_reward = deque(maxlen=30)
    last_n_game_reward.append(0)
    replay_buffer = deque(maxlen=SIZE_R_M)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, state_size, action_size, name="main")
        targetDQN = dqn.DQN(sess, state_size, action_size, name="target")
        init = tf.global_variables_initializer()
        sess.run(init)
        
        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        start_time = time.time()
        time_step = 0
        episode = 0

        while time.time() - start_time < 119 * 60:
            state = env.reset()
            rall = 0
            done = False
            ep_step = 0
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            
            while not done and ep_step < 10000 :
                ep_step += 1
                time_step += 1
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                nextstate, reward, done, _ = env.step(action)

                # if done:
                #     reward = -100

                replay_buffer.append((state, action, reward, nextstate, done))

                if len(replay_buffer) > SIZE_R_M:
                    replay_buffer.popleft()
                    
                state = nextstate
                
                if time_step > SIZE_R_M :
                    # for _ in range (batch_size):
                    minibatch = random.sample(replay_buffer, batch_size)
                    LossValue,_ = train_model(mainDQN,targetDQN, minibatch)
                        
                    if done or ep_step % target_update_cycle == 0:
                        sess.run(copy_ops)
                        
                if done or ep_step == 10000:
                    episode += 1
                    print("Episode :{:>5} / Epsidoe step :{:>5} / Recent n Game reward :{:>5.2f} / memory length :{:>5}"
                          .format(episode, ep_step, np.mean(last_n_game_reward),len(replay_buffer)))
                    break
            
            last_n_game_reward.append(ep_step)

            if len(last_n_game_reward) == last_n_game_reward.maxlen:
                avg_reward = np.mean(last_n_game_reward)
                if avg_reward < 199.0:
                    print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                    sys.exit()

        for episode in range(N_train_result_replay):
            
            state = env.reset()
            rall = 0
            done = False
            ep_step = 0
            
            while not done :
                env.render()
                ep_step += 1
                Q_Global = mainDQN.predict(state)
                action = np.argmax(Q_Global)
                state, reward, done, _ = env.step(action)
                rall += reward

            scores.append(rall)
            print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, ep_step, rall, np.mean(scores)))

if __name__ == "__main__":
    main()