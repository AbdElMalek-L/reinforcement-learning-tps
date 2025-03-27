import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", 
            #    render_mode = "human"
               )
env.reset()

state_size = env.observation_space.n
action_size = env.action_space.n

print(f"States space: {state_size}")
print(f"Actions space: {action_size}")

policy_table = np.ones((state_size, action_size)) / action_size

value_table = np.zeros(state_size)

print("First 5 rows of Policy table :")
print(policy_table[ 5])

print("\nFirst 5 values of Value table :")
print(value_table[ 5])

num_episodes = 20

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    print(f"\nEpisode {episode + 1}:")
    
    while not done:
        action = env.action_space.sample()  

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        print(f"State: {state}, Action: {action}, Reward: {reward}")
        state = next_state
    
    print(f"Total Reward: {total_reward}")



# *************Q-learning********************

q_table = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.999  
num_episodes = 5000  

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        
        # Exploration
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 

        # Exploitation     
        else:  
            action = np.argmax(q_table[state, :]) 
           
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        state = next_state
    
    epsilon = max(0.01, epsilon * epsilon_decay)  

num_test_episodes = 10
success_count = 0


env = gym.make("Taxi-v3", 
                render_mode = "human"
               )


for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = np.argmax(q_table[state, :]) 
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        
    if reward == 20: 
        success_count += 1

success_rate = success_count / num_test_episodes
print(f"Success Rate: {success_rate * 100:.2f}%")
