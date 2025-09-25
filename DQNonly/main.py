import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import Replaymemo
import itertools
reward_ep =[]
class Agent:
    def run(self,Is_training = True,render = False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        if (Is_training):
            buffer = Replaymemo(maxlen = 10000)
        for i in itertools.count():
            state,_ = env.reset()
            reward1 = 0
            terminated = False
            policy = DQN(env.observation_space.shape[0],env.action_space.n).to("cuda")

            while not terminated:
                    action = env.action_space.sample()
                    newstate, reward, terminated, _, info = env.step(action)
                    reward1 = reward +reward1 
                    if Is_training :
                        buffer.append((state,action,newstate,reward,terminated))
                    if terminated:
                        break
                        env.close()
            print(reward1)
            reward_ep.append(reward1)
x = Agent()
x.run(render = True)