import gymnasium as gym
import keyboard
import time

env = gym.make("CartPole-v1", 
            #    render_mode="human"
               )
env.reset()

print(f"Espace d'action : {env.action_space}")
print(f"Espace d'observation : {env.observation_space}")
print("********* Enter 0 or 1 ***********")
eteration = 0

try:
    while True:
        if keyboard.is_pressed("0"):
            action = env.action_space.sample()
            observation, reward, done, _, _ = env.step(0)
            print(f"Eteration : {eteration} Action : {0}, Observation : {observation}, Reward : {reward}")
            eteration += 1
            
            if done:
                print("\ndone\n")
                env.reset()
                
            time.sleep(0.1) 
        if keyboard.is_pressed("1"):
            action = env.action_space.sample()
            observation, reward, done, _, _ = env.step(1)
            print(f"Eteration : {eteration} Action : {1}, Observation : {observation}, Reward : {reward}")
            eteration += 1
            
            if done:
                print("\ndone\n")
                env.reset()
                
            time.sleep(0.1) 

except KeyboardInterrupt:
    print("\nArrêt")
finally:
    env.close()

