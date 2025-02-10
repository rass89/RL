

Here’s a simple example of Reinforcement Learning using Q-learning, which is a model-free algorithm for learning how to act optimally in an environment. We'll create a basic environment and demonstrate how an agent can learn to navigate it.

Example: Gridworld Environment
In this example, we will define a 5x5 grid where the agent can move up, down, left, or right. The goal is to reach a specific cell in the grid while avoiding a penalty.

Explanation
Gridworld Environment:
The Gridworld class defines a simple grid environment. The agent starts in the top-left corner and aims to reach the bottom-right corner (goal).
The step method defines how the agent moves in the grid based on its action (up, down, left, right), and returns the new state, reward, and whether the episode has ended.
Q-learning Agent:
The QLearningAgent class implements the Q-learning algorithm. It maintains a Q-table that stores the expected utility of each action in each state.
The choose_action method selects an action using an epsilon-greedy strategy, balancing exploration and exploitation.
The update_q_table method updates the Q-values using the Bellman equation.
The train method runs multiple episodes to help the agent learn the optimal policy.
Training:
The agent is trained over a specified number of episodes, during which it interacts with the environment and updates its Q-table based on the rewards received.
"""

import numpy as np
import random

# Define the grid environment
class Gridworld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.state = (0, 0)  # Start at the top-left corner
        self.goal = (4, 4)   # Goal is at the bottom-right corner
        self.penalty = -1     # Penalty for stepping into certain cells
        self.reward = 0       # Reward for reaching the goal

    def reset(self):
        self.state = (0, 0)  # Reset to starting position
        return self.state

    def step(self, action):
        # Define the action space
        if action == 0:  # Up
            new_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 1:  # Down
            new_state = (min(self.state[0] + 1, self.height - 1), self.state[1])
        elif action == 2:  # Left
            new_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 3:  # Right
            new_state = (self.state[0], min(self.state[1] + 1, self.width - 1))

        self.state = new_state

        # Check for rewards or penalties
        if self.state == self.goal:
            return self.state, 1, True  # Goal reached
        else:
            return self.state, self.penalty, False  # Penalty for stepping

# Q-learning algorithm
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.q_table = np.zeros((env.height, env.width, 4))  # 4 actions (up, down, left, right)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore: random action
        else:
            return np.argmax(self.q_table[self.env.state])  # Exploit: best action

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action()
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

            # Decay exploration rate
            self.exploration_rate *= self.exploration_decay

# Main execution
if __name__ == "__main__":
    env = Gridworld(5, 5)
    agent = QLearningAgent(env)

    # Train the agent
    agent.train(1000)

    # Display the learned Q-values
    print("Learned Q-values:")
    print(agent.q_table)

"""Here's a real-world example of Reinforcement Learning applied to a popular environment: OpenAI Gym's CartPole. In this example, we'll use the Deep Q-Network (DQN) algorithm to train an agent to balance a pole on a cart.

Example: CartPole Environment
In the CartPole environment, the goal is to keep a pole balanced vertically on a cart that can move left or right. The agent receives a reward for every time step the pole remains upright.
Environment Setup:
We use OpenAI Gym to create the CartPole environment. It provides a standard API for RL environments.
Hyperparameters:
Learning rate, discount factor, exploration probability, and batch size are defined to control the training process.
A deque is used to store experiences for replay.
DQN Model:
We define a neural network using TensorFlow's Keras API. It consists of two hidden layers with 24 neurons each and outputs Q-values for two actions: moving left or right.
Training Loop:
For each episode:
The agent chooses actions based on an epsilon-greedy policy (exploring or exploiting).
The agent takes the action, observes the new state and reward, and stores the experience in memory.
The model is trained on a random batch of experiences from memory to learn the Q-values.
The exploration probability decays over time to favor exploitation.
Reward Tracking:
The total rewards for each episode are tracked and plotted at the end.
Model Saving:
Finally, the trained model is saved for future use.
"""

#2
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Set up the parameters
num_episodes = 1000
output_dir = 'model_output/cartpole/'

# Hyperparameters for DQN
learning_rate = 0.001
discount_factor = 0.99
exploration_prob = 1.0
exploration_decay = 0.995
min_exploration_prob = 0.01
batch_size = 32
memory = deque(maxlen=2000)

# Create the DQN model
def create_model():
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))  # 2 actions: left or right
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model

# Train the DQN agent
def train_dqn():
    global exploration_prob # Add this line to indicate exploration_prob is a global variable
    model = create_model()
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action using epsilon-greedy policy
            if random.random() < exploration_prob:
                action = env.action_space.sample()  # Explore
            else:
                q_values = model.predict(state.reshape(1, 4))
                action = np.argmax(q_values[0])  # Exploit

            # Take action and get the next state and reward
            next_state, reward, done, _ = env.step(action)

            # Store the experience in memory
            memory.append((state, action, reward, next_state, done))

            # Update state
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # Train the model using a batch of experiences
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target += discount_factor * np.amax(model.predict(next_state.reshape(1, 4))[0])
                target_f = model.predict(state.reshape(1, 4))
                target_f[0][action] = target
                model.fit(state.reshape(1, 4), target_f, epochs=1, verbose=0)

        # Decay exploration probability
        if exploration_prob > min_exploration_prob:
            exploration_prob *= exploration_decay

        # Print episode info
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Exploration Prob: {exploration_prob:.2f}")

    return rewards

# Run training
if __name__ == "__main__":
    rewards = train_dqn()

    # Plot the rewards
    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Save the model
    model.save(output_dir + 'cartpole_model.h5')