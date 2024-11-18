import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Architecture(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Architecture, self).__init__()
        self.input = nn.Linear(input_dim, 256)
        self.layer = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.layer(x))
        return F.softmax(self.output(x), dim=-1)  

class REINFORCEAgent:
    def __init__(self, input_dim, output_dim, learning_rate, gamma, entropy):
        self.output_dim = output_dim
        self.actor_net = Architecture(input_dim, output_dim)
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy = entropy
    
    def evaluate(self,environment,n_eval_episodes=5):
        episode_returns = []
        eval_env = gym.make('LunarLander-v2')
        for i in range(n_eval_episodes):
            s, _ = eval_env.reset()
            R_ep = 0
            done = False
            while not done:
                a = self.choose_action(s, greedy=False)
                s_next, r, done, *extra = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_next
            episode_returns.append(R_ep)
        mean_return = np.mean(episode_returns)
        return mean_return
    
    def choose_action(self, state, greedy=False):
        with torch.no_grad():
            input_state = torch.FloatTensor(state).unsqueeze(0)
            probabilities = self.actor_net(input_state).squeeze(0)
            if greedy:
                action = torch.argmax(probabilities).item()
            else:     
                action = np.random.choice(np.arange(self.output_dim), p=probabilities.detach().numpy())
        return action   
    
    def update_policy(self, states_collected, actions_performed, rewards_received):
        
        cumulative_return = 0.
        episode_length = len(rewards_received)
        episode_returns = np.zeros((episode_length,))
        for i in range(episode_length - 1, -1, -1):
            cumulative_return = rewards_received[i] + self.gamma * cumulative_return
            episode_returns[i] = cumulative_return

        tensor_states = torch.FloatTensor(states_collected)
        tensor_actions = torch.LongTensor(actions_performed).view(-1, 1)
        tensor_returns = torch.FloatTensor(episode_returns).view(-1, 1)

        probabilities = self.actor_net(tensor_states)
        log_probabilities = torch.log(probabilities.gather(1, tensor_actions))
        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        loss = -torch.mean(log_probabilities * tensor_returns) - self.entropy * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return  cumulative_return   

def Reinforce_Main(num_timesteps, learning_rate, gamma, entropy ):
    environment = gym.make('LunarLander-v2')
    # environment = gym.make('LunarLander-v2', render_mode='human')
    input_dim = environment.observation_space.shape[0]
    output_dim = environment.action_space.n
    timesteps = 0
    learning_rate = learning_rate
    gamma = gamma
    entropy = entropy
    num_timestep = num_timesteps
    eval_interval = 5000
    eval_returns = []
    eval_timesteps = []

    agent = REINFORCEAgent(input_dim, output_dim, learning_rate, gamma, entropy)

    while timesteps < num_timestep:
        state, _ = environment.reset()
        done = False
        states_collected = []
        actions_performed = []
        rewards_received = []
        
        while not done: 
            timesteps += 1
            action = agent.choose_action(state)
            next_state, reward, done, *extra = environment.step(action)
            actions_performed.append(action)
            rewards_received.append(reward)
            states_collected.append(state)
            state = next_state
            eval_interation = timesteps
            if  (eval_interation) % eval_interval == 0:
                mean_return = agent.evaluate(environment)
                eval_returns.append(mean_return)
                eval_timesteps.append(eval_interation)
                print(f"Evaluation Timestep {eval_interation}: Reward {mean_return}")
            
            if done:
                returns = agent.update_policy(states_collected, actions_performed, rewards_received)
                print(f"Timestep {timesteps + 1}: Reward {returns}")
                break

    environment.close()  
    print(eval_returns)  
    return eval_returns, eval_timesteps
    
def test():
    learning_rate = 0.001
    gamma = 0.99
    entropy = 0.1
    num_timesteps = 2000000
    eval_returns, eval_timesteps = Reinforce_Main(num_timesteps,learning_rate, gamma, entropy)
    print(eval_returns,eval_timesteps)

if __name__ == "__main__":
    test()
