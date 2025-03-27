# Reinforcement Learning with OpenAI Gym

In these first three TPs, we focused on learning the fundamentals of Reinforcement Learning. We used the famous **<font color="#00b050">OpenAI Gym</font>** library, starting with **<font color="#ffff00">CartPole-v1</font>** to familiarize ourselves with the essential tools of reinforcement learning. Then, we moved to **<font color="#ffff00">FrozenLake-v1</font>**, implementing Reinforcement Learning using the **<font color="#ffff00">Q-learning Algorithm</font>**. Finally, in **Taxi-v3**, we explored policy-based learning by constructing a **Policy Table** and **Value Table** for the agent's training.

| TP                               | Environment                                | Algorithm/Technique                                              |
| -------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| <font color="#245bdb">1st</font> | <font color="#ffff00">CartPole-v1</font>   | <font color="#002060">Manual interaction (basic RL tools)</font> |
| <font color="#ff0000">2nd</font> | <font color="#ffff00">FrozenLake-v1</font> | <font color="#ff0000">Q-learning</font>                          |
| <font color="#00b050">3rd</font> | <font color="#ffff00">Taxi-v3</font>       | <font color="#00b050">Policy-based learning & Q-learning</font>  |



---

## <font color="#245bdb">1st TP -</font> <font color="#ffff00">CartPole-v1</font>

In this first practical session, we worked with **<font color="#ffff00">CartPole-v1</font>** to understand the basic structure of Reinforcement Learning environments.

### Installation of OpenAI Gym

```bash
pip install --upgrade gymnasium pygame
```

### Environment Initialization

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
```

### Key Concepts:

- **Understanding the environment:** The agent balances a pole on a moving cart.
    
- **Action space:** Discrete with two actions (0: move left, 1: move right).
    
- **Observation space:** Continuous with four state variables (cart position, cart velocity, pole angle, pole angular velocity).
    
- **Reward structure:** +1 for every step the pole remains balanced.
    
- **Episode termination:** The episode ends when the pole falls beyond a certain angle or the cart moves too far.
    
- **Interacting with the environment:** Using keyboard inputs (0 or 1) to move the cart and observing the changes in state, reward, and detecting when an episode is done.
    

---

## <font color="#ff0000">2nd TP -</font> <font color="#ffff00">FrozenLake-v1</font> <font color="#ff0000">(Q-Learning)</font>

In this TP, we implemented **Q-learning**, a fundamental Reinforcement Learning algorithm, on **FrozenLake-v1**.

### Key Steps:

1. **Initialize the environment:**
    
    ```python
    env = gym.make("FrozenLake-v1", is_slippery=False)
    ```
    
2. **Create a Q-table:**
    
    ```python
    import numpy as np
    
    q_table = np.zeros((16, 4))
    ```
    
3. **Understanding the algorithm:**
    
    - **Exploration vs. Exploitation:** The agent initially explores the environment randomly and then starts choosing optimal actions based on learned values.
        
    - **Q-value update:** Using the Bellman equation:
        
        ```python
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        ```
        
4. **Training the agent:**
    
    - Running 5000 episodes to update the Q-table.
        
    - Using an **epsilon-greedy strategy** where the agent chooses random actions with probability `epsilon`, decreasing over time.
        
5. **Testing the trained agent:**
    
    - Running test episodes where the agent chooses the best-known action based on the Q-table.
        
    - Measuring the **success rate** to evaluate performance.
        

---

## <font color="#00b050">3rd TP -</font> <font color="#ffff00">Taxi-v3</font> <font color="#00b050">(Policy-based Learning & Q-learning)</font>

In this TP, we worked on **Taxi-v3**, implementing policy-based learning methods alongside Q-learning.

### Key Concepts:

1. **Understanding the environment:**
    
    - The agent must pick up a passenger and drop them off at the correct location.
        
    - **Action space:** 6 discrete actions (move in four directions, pick up, drop off).
        
    - **State space:** 500 possible states, representing taxi position, passenger location, and destination.
        
    - **Reward structure:**
        
        - +20 for a successful drop-off.
            
        - -1 for each step taken.
            
        - -10 for illegal pickups/drop-offs.
            
2. **Creating a Policy Table and Value Table:**
    
    ```python
    import numpy as np
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    policy_table = np.ones((state_size, action_size)) / action_size
    value_table = np.zeros(state_size)
    ```
    
3. **Training the agent using a policy iteration approach:**
    
    - Selecting actions based on policy.
        
    - Updating value functions based on the observed rewards.
        
4. **Q-learning Implementation:**
    
    - **Exploration-exploitation trade-off:** Using an **epsilon-greedy strategy**.
        
    - **Q-value update using the Bellman equation:**
        
        ```python
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        ```
        
    - **Hyperparameters:**
        
        ```python
        alpha = 0.1  # Learning rate
        gamma = 0.99  # Discount factor
        epsilon = 1.0  # Exploration rate
        epsilon_decay = 0.999  # Decay factor for exploration
        ```
        
5. **Testing and Evaluating the Agent:**
    
    - Running test episodes where the agent selects the best-known action.
        
    - Calculating the **success rate** to measure performance.
        
    - **Success is defined as successfully dropping off the passenger.**
        
