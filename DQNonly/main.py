import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import Replaymemo
import itertools
import yaml
import random
class Agent:
    def __init__(self,hyperparameters_set):
        with open("hyperparameters.yml","rb") as f :
            hyperparameters = yaml.safe_load(f)
        parameters = hyperparameters[hyperparameters_set]
        self.replaysize = parameters["replaysize"]
        self.minibatch = parameters["minibatch"]
        self.epsilon = parameters["epsilon"]
        self.decay = parameters["decay"]
        self.min_epsilon  = parameters["min_epsilon"]
    def run(self,Is_training = True,render = False):        
    #    env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1",render_mode = "human"if render else None)
        reward_ep =[]
        epsilon_history = []

        if (Is_training):
            k = self.replaysize
            d = self.minibatch
            buffer = Replaymemo(maxlen = k,sample_size= d)
        policy = DQN(env.observation_space.shape[0],env.action_space.n).to("cuda")

        for i in itertools.count():
            state,_ = env.reset()
            reward1 = 0
            terminated = False
            truncated = False
            state = torch.tensor(state,dtype = torch.float,device= "cuda")


            while not terminated or truncated:
                    if Is_training and random.random()<self.epsilon:
                        action = env.action_space.sample()
                        action1 = torch.tensor(action,dtype = torch.int,device= "cuda").squeeze().item()

                    else:
                        action1 = policy(state.unsqueeze(0)).squeeze().argmax().item()
                        print(action1)
                    newstate, reward, terminated, truncated, info = env.step(action1)
                    reward1 = reward +reward1 
                    newstate = torch.tensor(newstate,dtype = torch.float,device= "cuda")
                    reward =  torch.tensor(reward,dtype = torch.float,device= "cuda")
                    if Is_training :
                        buffer.append((state,action1,newstate,reward,terminated))
                    if terminated:
                        break
                        env.close()
                    state  = newstate
            self.epsilon -= self.decay
            self.epsilon = max(0.01 , self.epsilon)
            print(reward1)
            reward_ep.append(reward1)
            epsilon_history.append(self.epsilon)
            print(self.epsilon)
x = Agent("cartpole1")
x.run(render = False)