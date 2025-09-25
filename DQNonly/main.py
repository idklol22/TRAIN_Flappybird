import flappy_bird_gymnasium
import gymnasium
import torch
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break

env.close()