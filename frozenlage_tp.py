import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()

print(f"Actions space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

q_table = np.zeros((16, 4))
print(q_table)

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 5000

for episode in range(num_episodes):

    state, _ = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        next_state, reward, done, _, _ = env.step(action)

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        state = next_state

        print(f"Épisode {episode} - Action: {action}, État: {state}, Récompense: {reward}")

    epsilon = max(epsilon * epsilon_decay, epsilon_min)





print("Q-Table finale:")
print(q_table)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")


num_test_episodes = 10
successes = 0

for _ in range(num_test_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state, :])
        state, reward, done, _, _ = env.step(action)
        
        if done and reward == 1.0:
            successes += 1

taux_reussite = (successes / num_test_episodes) * 100
print(f"Taux de réussite: {taux_reussite}%")

env.close()
