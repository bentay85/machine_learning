#run python play_game.py
#you must click on the prompt once the window pops up for your keayboard commands to be read
#press a to move left, by default, you will be moving right

import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from time import sleep
import keyboard
import numpy as np

env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda:env])

obs=env.reset()
done=False
score = 0
action = np.array([1])

while not done:
    
    env.render()
    
    if keyboard.is_pressed("a"):
        action = np.array([0])
    else:
        action = np.array([1])
        
    obs,reward,done,info = env.step(action)
    score += reward
    
    sleep(0.1)

print("Score: " + str(score[0]))

env.close()