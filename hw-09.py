# import gym
import gymnasium as gym

import time

# Створюємо середовище FrozenLake
env = gym.make('FrozenLake-v1', is_slippery=True)
state = env.reset()

# Перевіримо базову інформацію
print("Кількість станів:", env.observation_space.n)
print("Кількість дій:", env.action_space.n)

"""
done = False
while not done:
    action = env.action_space.sample()  # випадкова дія
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(0.3)

print("Епізод завершено. Винагорода:", reward)
"""


