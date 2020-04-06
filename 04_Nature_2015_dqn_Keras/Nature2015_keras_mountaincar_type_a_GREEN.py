import sys
import gym
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.activations import relu, linear
import matplotlib.pyplot as plt

# In case of CartPole-v1, maximum length of episode is 500
env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

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
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.progress = ''
        self.action_size = action_size
        self.state_size = state_size
        
        # train time define
        self.c_exploit_cycle = 5*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 24, 24
        
        self.score = 0
        self.episode = 0
        
        # Parameter for Experience Replay
        self.size_replay_memory = 50000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 2

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()
        
        if self.load_model:
            self.model.load_weights("./save_model/MountainCar_DQN.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform'))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.size_replay_memory:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states      = np.array([x[0] for x in minibatch])
        actions     = np.array([x[1] for x in minibatch])
        rewards     = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones       = np.array([x[4] for x in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        q_update = rewards + self.discount_factor*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
        q_value = self.model.predict_on_batch(states)

        ind = np.array([x for x in range(self.batch_size)])
        q_value[[ind], [actions]] = q_update

        self.model.fit(states, q_value, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
    
def main():
    
    # DQN 에이전트의 생성
    agent = DQN(state_size, action_size)
    scores, episodes = [], []
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)        
    agent.progress = 'Exploration'
    print("\n\n Exploration started")
    
    while len(agent.memory) < agent.size_replay_memory:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done and score < 30000:
            score += 1
            if agent.render:
                env.render()        
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            #If the car pulls back on the left or right hill he gets a reward of +20
            if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1]>0:
                reward = 20
            elif next_state[1] < state[0][1] and next_state[1]<=0 and state[0][1]<=0:
                reward = 20
            #if he finishes with less than 200 steps
            if done and score < 200:
                reward += 10000
            else:
                reward += -25
            
            next_state = np.reshape(next_state, [1, state_size])
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                agent.episode += 1
                print("Exploration episode :",agent.episode,"Memory size :", len(agent.memory) )
                
    agent.episode = 0            
    agent.progress = 'Exploitation'
    start_time = time.time()
    print("\n\n Exploitation started")

    while time.time() - start_time < agent.c_exploit_cycle:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done and score < 30000:
            score += 1
            if agent.render:
                env.render()        
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            #If the car pulls back on the left or right hill he gets a reward of +20
            if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1]>0:
                reward = 20
            elif next_state[1] < state[0][1] and next_state[1]<=0 and state[0][1]<=0:
                reward = 20
            #if he finishes with less than 200 steps
            if done and score < 200:
                reward += 10000
            else:
                reward += -25
            
            next_state = np.reshape(next_state, [1, state_size])
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state
            agent.train_model()

            if done or score == 30000:
                agent.episode += 1
                print("episode: {}, score: {}".format(agent.episode, score))
                # if the mean of scores of last 30 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(30, len(scores)):]) < 250:
                    agent.model.save_weights("./save_model/MountainCar_DQN.h5")
                    sys.exit()
                    # break
                # save the model
                if agent.episode % 50 == 0:
                     agent.model.save_weights("./save_model/MountainCar_DQN.h5")
                break
                
        if agent.episode % agent.target_update_cycle == 0:
            if len(agent.memory) == agent.size_replay_memory:
                # return# copy q_net --> target_net
                agent.Copy_Weights()
                    
if __name__ == "__main__":
    main()
