import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import pickle
from typing import List
import matplotlib.pyplot as plt
import time
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

env_name = "MountainCar-v0"
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance
np.random.seed(123)
tf.set_random_seed(456)  # reproducible
env = env.unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

target_update_cycle = 200

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN:
    """ Implementation of deep q learning algorithm """   
    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        
        #HyperParameters
        self.session = session
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 30,30
        self.memory_size = 50000
        self.batch_size = 32
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9997
        self.epsilon = self.epsilon_max
        
        #Experience Replay 
        self.memory = deque(maxlen=self.memory_size)
        self.net_name = name
        self.model = self.build_model()
        
    # def build_model(self, H_SIZE_01 = 256, H_SIZE_02 = 256, H_SIZE_03 = 256, self.learning_rate=0.001) -> None:
    def build_model(self):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            
            net_0 = self._X

            net = tf.layers.dense(net_0, self.hidden1, activation=tf.nn.relu)
            # net = tf.layers.dense(net, self.hidden2, activation=tf.nn.relu)
            net_16 = tf.layers.dense(net, self.action_size)
            self._Qpred = net_16

            self._LossValue = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self._train = optimizer.minimize(self._LossValue)

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = np.reshape(state, [-1, self.state_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._LossValue, self._train], feed)
    
    def get_action(self,state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values  = self.predict(state)
        
        return np.argmax(q_values[0])
    
    def append_sample(self,state,action,reward,next_state,done):
        #in every action put in the memory
        self.memory.append((state,action,reward,next_state,done))

def Copy_Weights(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder    
    
def train_model(agent, target_agent):
    #When the memory is filled up take a batch and train the network
    if len(agent.memory) < agent.memory_size:
        return

    minibatch = random.sample(agent.memory, agent.batch_size)
    for state, action, reward, next_state, done in minibatch:
        q_update = reward
        if not done:
            #Obtain the Q' values by feeding the new state through our network
            q_update = (reward + agent.discount_factor*np.amax(target_agent.predict(next_state)[0]))
        q_values = agent.predict(state)
        q_values[0][action] = q_update
        agent.update(state,q_values)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

def main():
    
    with tf.Session() as sess:
        agent        = DQN(sess, state_size, action_size, name="main")
        target_agent = DQN(sess, state_size, action_size, name="target")
        
        init = tf.global_variables_initializer()
        sess.run(init)

        # initial copy q_net -> target_net
        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        
        episode = 0
        start_time = time.time()
        while time.time() - start_time < 29*60:
        
        # for episode in range(EPISODES):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            step = 0
            scores = []

            while not done and score < 10000:
                if agent.render:
                    env.render()        
                action = agent.get_action(state)
                
                # 선택한 액션으로 1 step을 시행한다
                next_state, reward, done, _ = env.step(action)
                
                """
                #If the car pulls back on the left or right hill he gets a reward of +20
                if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1]>0:
                    reward = 20
                elif next_state[1] < state[0][1] and next_state[1]<=0 and state[0][1]<=0:
                    reward = 20
                #if he finishes with less than 200 steps
                if done and step < 200:
                    reward += 10000
                else:
                    reward += -25
                """
                score += 1
                next_state = np.reshape(next_state, [1, state_size])
                agent.append_sample(state, action, reward, next_state, done)
                state = next_state
                
                # if done or score % 10 == 0:
                train_model(agent, target_agent)
                    
                if done or score % target_update_cycle == 0:
                    if len(agent.memory) == agent.memory_size:
                        # return# copy q_net --> target_net
                        sess.run(copy_ops) 

                if done or score == 10000:
                    episode += 1
                    scores.append(score)
                    print ("Run: {:>4}".format(episode),"/ epsilon: {:2.5f}".format(agent.epsilon),
                           "/ score: {:>5}".format(score))
                    if np.mean(scores[-min(30, len(scores)):]) < 200:
                        e = int(time.time() - start_time)
                        print('Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
                        sys.exit()
                    break
                                
        # save the model
        if (episode+1) % 50 == 0:
            agent.model.save_weights("./save_model/MountainCar_DQN.h5")
                
if __name__ == "__main__":
    main()
