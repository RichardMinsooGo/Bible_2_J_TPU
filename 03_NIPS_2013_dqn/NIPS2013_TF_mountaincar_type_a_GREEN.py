import os
import sys
import gym
import pylab
import numpy as np
import random
import time
from collections import deque
import tensorflow as tf
import pickle
env_name = "MountainCar-v0"
env = gym.make(env_name)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define Network Parameters
H_SIZE_01 = 128
H_SIZE_02 = 128

EPISODES = 300

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN:
    """ Implementation of deep q learning algorithm """   
    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:
        
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
        self.epsilon_decay = 0.997
        self.epsilon_rate = self.epsilon_max
        
        #Experience Replay 
        self.memory = deque(maxlen=self.memory_size)
        self.net_name = name
        self.model = self.build_model()
        

    def build_model(self, H_SIZE_01 = 128, H_SIZE_02 = 128, Alpha=0.001) -> None:
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            net_0 = self._X

            net = tf.layers.dense(net_0, H_SIZE_01, activation=tf.nn.relu)
            net = tf.layers.dense(net, H_SIZE_02, activation=tf.nn.relu)
            net_16 = tf.layers.dense(net, self.action_size)
            self._Qpred = net_16

            self._LossValue = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate = Alpha)
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
        if np.random.rand() <= self.epsilon_rate:
            return random.randrange(self.action_size)
        
        q_values  = self.predict(state)
        
        return np.argmax(q_values[0])
    
    def append_sample(self,state,action,reward,next_state,done):
        #in every action put in the memory
        self.memory.append((state,action,reward,next_state,done))
    
    
def train_model(agent):
    #When the memory is filled up take a batch and train the network
    if len(agent.memory) < agent.memory_size:
        return

    mini_batch = random.sample(agent.memory, agent.batch_size)
    for state,action,reward,next_state, done in mini_batch:
        q_update = reward
        if not done:
            q_update = (reward + agent.discount_factor*np.amax(agent.predict(next_state)[0]))
        q_values = agent.predict(state)
        q_values[0][action] = q_update
        agent.update(state,q_values)

    if agent.epsilon_rate > agent.epsilon_min:
        agent.epsilon_rate *= agent.epsilon_decay


def main():
    
    with tf.Session() as sess:
        agent = DQN(sess, state_size, action_size, name="main")
        init = tf.global_variables_initializer()
        sess.run(init)

        flag = False

        for episode in range(EPISODES):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            step = 0

            while not done:
                step += 1
         
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

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

                score += reward
                next_state = np.reshape(next_state, [1, state_size])
                agent.append_sample(state, action, reward, next_state, done)
                state = next_state
                train_model(agent)

                if done:
                    if step < 200 :
                        flag = True
                        print("Successful Episode!")
                    print ("Run: {:>4}".format(episode+1),", exploration: {:2.5f}".format(agent.epsilon_rate),
                           ", score: " + str(step), "Rewards :",score)
                    break

if __name__ == "__main__":
    main()
