import gymnasium as gym
from stable_baselines3 import PPO

import environment


env = gym.make("camera-v3", render_mode="human")

model = PPO.load("models/trained_model")
model.set_env(env)

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
