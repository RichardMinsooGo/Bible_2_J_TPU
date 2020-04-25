import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import dqn
from typing import List
import time
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

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

discount_factor = 0.99
N_EPISODES = 5000
N_train_result_replay = 20
target_update_cycle = 200
memory_size = 50000
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
                 
def train_model(agent: dqn.DQN, target_agent: dqn.DQN, minibatch: list) -> float:
    states      = np.vstack([batch[0] for batch in minibatch])
    actions     = np.array( [batch[1] for batch in minibatch])
    rewards     = np.array( [batch[2] for batch in minibatch])
    next_states = np.vstack([batch[3] for batch in minibatch])
    dones       = np.array( [batch[4] for batch in minibatch])

    X_batch = states
    Y_batch = agent.predict(states)

    Q_Global = rewards + discount_factor * np.max(target_agent.predict(next_states), axis=1) * ~dones

    Y_batch[np.arange(len(X_batch)), actions] = Q_Global

    return agent.update(X_batch, Y_batch)

def main():
    last_n_game_reward = deque(maxlen=30)
    last_n_game_reward.append(10000)
    memory = deque(maxlen=memory_size)

    with tf.Session() as sess:
        agent        = dqn.DQN(sess, state_size, action_size, name="main")
        target_agent = dqn.DQN(sess, state_size, action_size, name="target")
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        start_time = time.time()
        step = 0
        episode = 0

        while time.time() - start_time < 29 * 60:
            
            state = env.reset()
            rall = 0
            done = False
            ep_step = 0
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            progress = " "
            
            while not done and ep_step < 10000 :
                ep_step += 1
                step += 1
                
                if step < memory_size:
                    progress = "Exploration"
                else :
                    progress = "Training" 
                
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(agent.predict(state))

                next_state, reward, done, _ = env.step(action)
                
                memory.append((state, action, reward, next_state, done))

                if len(memory) > memory_size:
                    memory.popleft()
                    
                state = next_state
                
                if progress == "Training":
                    # for _ in range (batch_size):
                    minibatch = random.sample(memory, batch_size)
                    LossValue,_ = train_model(agent,target_agent, minibatch)
                        
                    if done or ep_step % target_update_cycle == 0:
                        sess.run(copy_ops)
                        
                if done or ep_step == 10000:
                    if progress == "Training":
                        episode += 1
                    last_n_game_reward.append(ep_step)
                    print("Episode :{:>5} / Epsidoe step :{:>5} / Recent n Game reward :{:>5.2f} / memory length :{:>5}"
                          .format(episode, ep_step, np.mean(last_n_game_reward),len(memory)))
                    avg_reward = np.mean(last_n_game_reward)
                    if avg_reward < 199.0:
                        print("Game Cleared within {:>5} episodes with avg reward {:>5.2f}".format(episode, avg_reward))
                        # sys.exit()
                    break

        for episode in range(N_train_result_replay):
            
            state = env.reset()
            rall = 0
            done = False
            ep_step = 0
            
            while not done :
                env.render()
                ep_step += 1
                Q_Global = agent.predict(state)
                action = np.argmax(Q_Global)
                state, reward, done, _ = env.step(action)
                rall += reward

            scores.append(rall)
            print("Episode : {:>5} steps : {:>5} r={:>5}. averge reward : {:>5.2f}".format(episode, ep_step, rall, np.mean(scores)))

if __name__ == "__main__":
    main()
