import gymnasium as gym
import environment as environment
import random

from stable_baselines3 import PPO


env = gym.make("camera-v3", render_mode="human")

model = PPO.load("models/model_1176000_steps")
model.set_env(env)

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        seed = random.randint(0, 100)
        obs, info = env.reset(seed=seed)
