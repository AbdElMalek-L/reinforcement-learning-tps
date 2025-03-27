import numpy as np
import random
import time

# Paramètres de la grille
grid_size = 5  # Taille de la grille (5x5)
trap_positions = [(1, 2), (3, 3), (2, 4)]  # Positions des pièges
treasure_position = (4, 4)  # Position du trésor

# Paramètres d'apprentissage
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur d'actualisation
epsilon = 0.1  # Probabilité d'exploration

actions = ["up", "down", "left", "right"]  # Actions possibles
q_table = np.zeros((grid_size, grid_size, len(actions)))  # Initialisation de la table Q

def get_next_position(state, action):
    x, y = state
    if action == "up" and x > 0:
        x -= 1
    elif action == "down" and x < grid_size - 1:
        x += 1
    elif action == "left" and y > 0:
        y -= 1
    elif action == "right" and y < grid_size - 1:
        y += 1
    return (x, y)

def get_reward(state):
    if state in trap_positions:
        return -10
    elif state == treasure_position:
        return 10
    else:
        return -1

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploration
    else:
        return actions[np.argmax(q_table[state[0], state[1]])]  # Exploitation

def train_agent(episodes=1000):
    for episode in range(episodes):
        state = (0, 0)  # Position initiale de l'agent
        done = False
        total_reward = 0
        
        while not done:
            action = choose_action(state)
            next_state = get_next_position(state, action)
            reward = get_reward(next_state)
            total_reward += reward
            
            # Mise à jour de la table Q
            action_index = actions.index(action)
            best_next_action = np.max(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action_index] = (1 - alpha) * q_table[state[0], state[1], action_index] + alpha * (reward + gamma * best_next_action)
            
            if next_state == treasure_position or next_state in trap_positions:
                done = True
            
            state = next_state
        
        print(f"Épisode {episode + 1}, Score: {total_reward}")
        print_grid(state)

def print_grid(state):
    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            if (i, j) == state:
                row += "😊  "  # Agent
            elif (i, j) in trap_positions:
                row += "💀  "  # Piège
            elif (i, j) == treasure_position:
                row += "🏆  "  # Trésor
            else:
                row += "⬜  "  # Espace vide
        print(row)
    print("\n")
    time.sleep(0.0)

def test_agent():
    state = (0, 0)
    path = [state]
    done = False
    
    while not done:
        print_grid(state)
        action = choose_action(state)
        next_state = get_next_position(state, action)
        path.append(next_state)
        
        if next_state == treasure_position or next_state in trap_positions:
            done = True
        
        state = next_state
    
    print_grid(state)
    return path

# Entraîner l'agent
train_agent(1000)

# Tester l'agent
optimal_path = test_agent()
print("Chemin optimal suivi par l'agent :", optimal_path)
