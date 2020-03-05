import sys
import gym
import pylab
import numpy as np
import random
import time
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

# In case of CartPole-v1, maximum length of episode is 500
env = gym.make('CartPole-v1')
# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# this is DeepSARSA Agent for the Cartpole
# it uses Neural Network to approximate q function
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        # these is hyper parameters for the DeepSARSA
        self.discount_factor = 0.99         # decay rate
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(30, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, self.action_size])
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(state, target, epochs=1, verbose=0)


def main():
    # PG 에이전트의 생성
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    episode = 0
    recent_rlist = deque(maxlen=100)
    recent_rlist.append(0)
    start_time = time.time()
    while time.time() <= start_time + 300:
    # for episode in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        while not done:
            # fresh env
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)
            # every time step we do train
            score += reward
            # swap observation
            state = next_state

            if done:
                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_sarsa.png")
                print("episode:", (episode+1), "  score:", score, "  epsilon:", agent.epsilon)
        episode += 1
        if (episode+1) % 100 == 0:
            agent.model.save_weights("./save_model/deep_sarsa.h5")
            
if __name__ == "__main__":
    main()
