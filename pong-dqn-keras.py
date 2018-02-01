"""
Notes about gym environment
env - game environment
done - boolean on whether game ended or not
old state information paired with action and next state and reward
    -> needed for training the agent
"""

"""
output layer of nn should reflect # of actions
"""

"""
loss = (target - prediction)**2
discounted reward -> future reward is worth less than immediate
target = reward + gamma * np.amax(model.predict(next_state))

keras handles hard work by subtracting target from neural network output and squaring it
also applied learning rate

fit() function decreases the gap between our prediction to target by the learning rate

DQN tends to forget the previous experiences as it overwrites them with the new experiences
List of previous experiences and observations to retrain the model with the previous experiences

memory = [(state, action, reward, next_state, done)...] -> list of memories
remember() -> memory.append(...)
"""

"""
Replay() -> train neural net with experiences in memory
minibatch = random.sample(self.memory, batch_size)
"""

"""
action -> at first select action by a certain percentage
try more actions before seeing patterns
when not deciding the action randomly -> agent predict the reward value
based on the current state and pick the action that will give the highest reward
act_values
"""

"""
hyperparameters
episodes -> number of games we want agent to play
gamma -> decay or discount rate
epsilon -> exploration rate -> rate at which agent randomly decides action rather than prediction
epsilon decay -> decrease number of explorations as it gets good at playing games
epsilon min -> agent explore atleast this amount
learning rate -> how much neural net learns in each iteration
"""
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
EPISODES = 1000



class DQN:
    # set hyperparameters
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(200, input_dim=6400, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0])) # important step
            target_f = self.model.predict(state) # important step
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # important step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":

    def prepro(I):
        #reduce game frame to just paddle and ball
        I = I[35:195] # cropping
        I = I[::2, ::2, 0]
        I[I == 144] = 0 #erase background
        I[I == 109] = 0
        I[I != 0] = 1 #paddles and balls set to 1
        return I.astype(np.float).ravel() #flatten to 1d array

    env = gym.make('Pong-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # print("!!!")
    # print(state_size)
    # print(action_size)
    # print("!!!")
    agent = DQN(state_size, action_size)
    done = False
    batch_size = 8

    for e in range(EPISODES):
        state = env.reset()
        # print(state.shape)
        # print(state.size)
        state = prepro(state)
        state = np.reshape(state, [1, 6400]) # how to reshape pong?
        reward_sum = 0
        while True:
            env.render()
            action = agent.act(state) # what is act?
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            next_state = prepro(next_state)
            next_state = np.reshape(next_state, [1, 6400])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            reward_sum += reward
            if done:
                print("episode: {}/{}, e: {:.2}, reward_sum: {}"
                      .format(e, EPISODES, agent.epsilon, reward_sum))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
