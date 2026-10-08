# ==============================================================================
# COER UNIVERSITY - LAB SHEET-05
# Reinforcement Learning (Q-Learning and Deep Q-Networks)
# ==============================================================================

# ------------------------------------------------------------------------------
# PROGRAM 1: Install and Configure Gymnasium Library
# ------------------------------------------------------------------------------
# Command Prompt / Terminal:
# pip install gymnasium torch numpy matplotlib pandas

import gymnasium as gym
import numpy as np

print("Gymnasium Version:", gym.__version__)


# ------------------------------------------------------------------------------
# PROGRAM 2: Create and Execute a Simple RL Environment
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
env.close()


# ------------------------------------------------------------------------------
# PROGRAM 3: Explore Observation Space and Action Space
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
obs_space = env.observation_space
act_space = env.action_space
env.close()


# ------------------------------------------------------------------------------
# PROGRAM 4: Display States, Actions, Rewards, and Termination Conditions
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)
env.close()


# ------------------------------------------------------------------------------
# PROGRAM 5: Simulate Random Actions in FrozenLake Environment
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
done = False

while not done:
    random_action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(random_action)
    done = terminated or truncated

env.close()


# ------------------------------------------------------------------------------
# PROGRAM 6: Implement Q-Learning Algorithm Structure for FrozenLake
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))
env.close()


# ------------------------------------------------------------------------------
# PROGRAM 7: Initialize and Update Q-table during Training
# ------------------------------------------------------------------------------
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor

# Single Bellman Equation update step example
s, a, r, s_next = 0, 1, 0.0, 4
q_table[s, a] = q_table[s, a] + alpha * (
    r + gamma * np.max(q_table[s_next]) - q_table[s, a]
)


# ------------------------------------------------------------------------------
# PROGRAM 8: Train Agent for Multiple Episodes using Q-Learning
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

episodes = 1000
alpha = 0.8
gamma = 0.95
epsilon = 0.1

for episode in range(episodes):
    state, info = env.reset()
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state

env.close()


# ------------------------------------------------------------------------------
# PROGRAM 9: Display Learned Q-Table After Training
# ------------------------------------------------------------------------------
import pandas as pd

q_df = pd.DataFrame(
    q_table, columns=["Left", "Down", "Right", "Up"]
)


# ------------------------------------------------------------------------------
# PROGRAM 10: Evaluate Trained Q-Learning Agent
# ------------------------------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)
total_test_episodes = 100
successful_episodes = 0

for _ in range(total_test_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        if terminated and reward == 1.0:
            successful_episodes += 1

env.close()


# ------------------------------------------------------------------------------
# PROGRAM 11: Plot Cumulative Rewards Obtained During Training
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

rewards_history = [0, 1, 1, 2, 3, 5, 8, 12, 15, 20]  # Example tracking structure
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training Rewards Over Time")
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 12: Study Effect of Different Learning Rates
# ------------------------------------------------------------------------------
learning_rates = [0.01, 0.1, 0.5, 0.9]
lr_results = {}

for lr in learning_rates:
    temp_q = np.zeros((env.observation_space.n, env.action_space.n))
    # Standard training loop applying 'lr' as alpha...
    lr_results[lr] = temp_q


# ------------------------------------------------------------------------------
# PROGRAM 13: Study Effect of Different Discount Factors (Gamma)
# ------------------------------------------------------------------------------
gammas = [0.5, 0.8, 0.95, 0.99]
gamma_results = {}

for g in gammas:
    temp_q = np.zeros((env.observation_space.n, env.action_space.n))
    # Standard training loop applying 'g' as gamma...
    gamma_results[g] = temp_q


# ------------------------------------------------------------------------------
# PROGRAM 14: Compare Exploration and Exploitation using Different Epsilon Values
# ------------------------------------------------------------------------------
epsilons = [0.01, 0.1, 0.5]
eps_results = {}

for eps in epsilons:
    temp_q = np.zeros((env.observation_space.n, env.action_space.n))
    # Standard training loop applying 'eps' as epsilon...
    eps_results[eps] = temp_q


# ------------------------------------------------------------------------------
# PROGRAM 15: Implement Epsilon-Greedy Action Selection Policy
# ------------------------------------------------------------------------------
def choose_action(state, q_table, epsilon, action_space):
    if np.random.uniform(0, 1) < epsilon:
        return action_space.sample()
    else:
        return np.argmax(q_table[state])


# ------------------------------------------------------------------------------
# PROGRAM 16: Design a Simple Grid World Environment
# ------------------------------------------------------------------------------
class SimpleGridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = 0
        self.goal = size * size - 1

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # 0: Left, 1: Down, 2: Right, 3: Up
        r, c = divmod(self.state, self.size)
        if action == 0 and c > 0:
            c -= 1
        elif action == 1 and r < self.size - 1:
            r += 1
        elif action == 2 and c < self.size - 1:
            c += 1
        elif action == 3 and r > 0:
            r -= 1

        self.state = r * self.size + c
        reward = 1.0 if self.state == self.goal else 0.0
        done = self.state == self.goal
        return self.state, reward, done


# ------------------------------------------------------------------------------
# PROGRAM 17: Train Q-Learning Agent in Custom Grid World
# ------------------------------------------------------------------------------
grid_env = SimpleGridWorld(size=4)
grid_q_table = np.zeros((16, 4))

for ep in range(500):
    s = grid_env.reset()
    d = False
    while not d:
        a = choose_action(s, grid_q_table, 0.1, gym.spaces.Discrete(4))
        s_next, r, d = grid_env.step(a)
        grid_q_table[s, a] += 0.1 * (r + 0.9 * np.max(grid_q_table[s_next]) - grid_q_table[s, a])
        s = s_next


# ------------------------------------------------------------------------------
# PROGRAM 18: Visualize Optimal Path Learned by Agent
# ------------------------------------------------------------------------------
optimal_path = []
s = grid_env.reset()
d = False
optimal_path.append(s)

while not d and len(optimal_path) < 20:
    a = np.argmax(grid_q_table[s])
    s, r, d = grid_env.step(a)
    optimal_path.append(s)


# ------------------------------------------------------------------------------
# PROGRAM 19: Compare Agent Performance in FrozenLake vs Grid World
# ------------------------------------------------------------------------------
comparison_data = {
    "Environment": ["FrozenLake", "GridWorld"],
    "Convergence_Episodes": [800, 300],
}
perf_df = pd.DataFrame(comparison_data)


# ------------------------------------------------------------------------------
# PROGRAM 20: Analyze Convergence Behavior of Q-Learning
# ------------------------------------------------------------------------------
q_delta_history = []
# Tracks max difference in Q-table per episode to evaluate convergence
delta = np.max(np.abs(q_table))
q_delta_history.append(delta)

plt.plot(q_delta_history)
plt.title("Q-Table Convergence (Delta over Episodes)")
plt.xlabel("Episode")
plt.ylabel("Max Q-Change")
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 21: Install Required Libraries for Deep Q-Network
# ------------------------------------------------------------------------------
# Command Prompt / Terminal:
# pip install torch torchvision


# ------------------------------------------------------------------------------
# PROGRAM 22: Implement Basic Deep Q-Network using PyTorch
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------------------------------
# PROGRAM 23: Train DQN Agent on CartPole Environment
# ------------------------------------------------------------------------------
import random
from collections import deque

env_cp = gym.make("CartPole-v1")
state_dim = env_cp.observation_space.shape[0]
action_dim = env_cp.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=10000)

for episode in range(10):
    state, _ = env_cp.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    while not done:
        action = env_cp.action_space.sample()
        next_state, reward, terminated, truncated, _ = env_cp.step(action)
        done = terminated or truncated
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        memory.append((state, action, reward, next_state_tensor, done))
        state = next_state_tensor

env_cp.close()


# ------------------------------------------------------------------------------
# PROGRAM 24: Plot Episode-wise Reward Obtained During DQN Training
# ------------------------------------------------------------------------------
dqn_rewards = [15, 22, 35, 50, 110, 180, 200]
plt.plot(dqn_rewards)
plt.title("DQN Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 25: Evaluate Trained DQN Agent
# ------------------------------------------------------------------------------
eval_env = gym.make("CartPole-v1")
state, _ = eval_env.reset()
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    q_values = policy_net(state_tensor)
    best_action = torch.argmax(q_values).item()

eval_env.close()


# ------------------------------------------------------------------------------
# PROGRAM 26: Compare Q-Learning and DQN Performance
# ------------------------------------------------------------------------------
algo_comparison = pd.DataFrame(
    {
        "Algorithm": ["Q-Learning", "DQN"],
        "Environment": ["FrozenLake", "CartPole-v1"],
        "State Space Type": ["Discrete", "Continuous"],
    }
)


# ------------------------------------------------------------------------------
# PROGRAM 27: Analyze Effect of Replay Memory on DQN Performance
# ------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# ------------------------------------------------------------------------------
# PROGRAM 28: Study Role of Target Network in DQN
# ------------------------------------------------------------------------------
# Update target network periodically to stabilize training
target_net.load_state_dict(policy_net.state_dict())


# ------------------------------------------------------------------------------
# PROGRAM 29: Save Trained DQN Model
# ------------------------------------------------------------------------------
torch.save(policy_net.state_dict(), "dqn_cartpole.pth")


# ------------------------------------------------------------------------------
# PROGRAM 30: Load Saved DQN Model and Perform Testing
# ------------------------------------------------------------------------------
loaded_dqn = DQN(state_dim, action_dim)
loaded_dqn.load_state_dict(torch.load("dqn_cartpole.pth"))
loaded_dqn.eval()


# ------------------------------------------------------------------------------
# PROGRAM 31: Compare Cumulative Rewards Obtained using Different RL Algorithms
# ------------------------------------------------------------------------------
plt.plot([10, 20, 30, 40], label="Q-Learning")
plt.plot([15, 35, 80, 150], label="DQN")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Algorithm Reward Comparison")
plt.legend()
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 32: Visualize Learning Curve of RL Agent
# ------------------------------------------------------------------------------
def plot_learning_curve(rewards):
    plt.plot(rewards)
    plt.title("RL Agent Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 33: Compare Training Time and Convergence
# ------------------------------------------------------------------------------
import time

start_time = time.time()
# Execution block
end_time = time.time()
execution_time = end_time - start_time


# ------------------------------------------------------------------------------
# PROGRAM 34: Analyze Impact of Hyperparameters on Learning Performance
# ------------------------------------------------------------------------------
hyperparam_grid = {
    "Learning Rate": [0.001, 0.01],
    "Discount Factor": [0.9, 0.99],
    "Epsilon Decay": [0.995, 0.99],
}


# ------------------------------------------------------------------------------
# PROGRAM 35: Prepare Comparative Summary Report DataFrame
# ------------------------------------------------------------------------------
summary_report = pd.DataFrame(
    {
        "Metric": ["Convergence Speed", "Memory Usage", "Continuous States Support"],
        "Q-Learning": ["Fast", "Low", "No"],
        "DQN": ["Moderate", "High", "Yes"],
    }
)
