import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# The critic network evaluates the value of being in a particular state.
class Network_Critic(nn.Module):
    def __init__(self, input_dim):
        super(Network_Critic, self).__init__()
        # Three fully connected layers form the network architecture.
        self.inputlayer = nn.Linear(input_dim, 256)  # First layer receives the state input.
        self.layer = nn.Linear(256, 256)             # Second layer for deeper processing.
        self.output = nn.Linear(256, 1)              # Output layer predicts the value of the state.

    def forward(self, x):
        # Use ReLU activation for non-linear transformation.
        x = F.relu(self.inputlayer(x))  # Activation for first layer.
        x = F.relu(self.layer(x))       # Activation for second layer.
        return self.output(x)           # Return the state value.

# The actor network outputs a probability distribution over actions.
class Network_Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network_Actor, self).__init__()
        # Neural network with two hidden layers and an output layer.
        self.inputlayer = nn.Linear(input_dim, 256)  # Processes the input state.
        self.layer = nn.Linear(256, 256)             # Further processes the state representation.
        self.output = nn.Linear(256, output_dim)     # Outputs probabilities for each possible action.

    def forward(self, x):
        # Activation functions followed by a softmax to create a probability distribution.
        x = F.relu(self.inputlayer(x))  # Non-linear transformation of the input.
        x = F.relu(self.layer(x))       # Additional non-linearity to capture complex patterns.
        return F.softmax(self.output(x), dim=-1)  # Softmax to get a probability distribution over actions.

# Function to evaluate the model's performance by running several episodes.
def evaluate(model_agent, environment):
    rewards = []
    episodes = 5  # Set the number of evaluation episodes.
    for episode in range(episodes):
        is_done = False
        r = 0
        s, _ = environment.reset()  # Reset environment to start a new episode.
        while not is_done:
            a = model_agent.choose_action(s, greedy=False)[0]  # Model chooses an action.
            s, reward, is_done, *extra = environment.step(a)  # Environment responds to the action.
            r += reward
            if is_done:
                rewards.append(r)  # Collect total reward for the episode.
                break
    average_reward = np.mean(rewards)  # Calculate the average reward over all episodes.
    print(f"Average Reward: {average_reward:.2f}")
    return average_reward

# Agent that uses actor-critic method for learning.
class Agent_ActorCritic:
    def __init__(self, input_dim, action_size, learning_rate, gamma, entropy):
        # Instantiate actor and critic networks.
        self.actor = Network_Actor(input_dim, action_size)
        self.critic = Network_Critic(input_dim)
        # Set up optimizers for both networks.
        self.optimizer_for_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_for_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        # Discount factor for future rewards and entropy term for exploration.
        self.gamma = gamma
        self.entropy_coefficient = entropy

    # Update function to apply the learning algorithm.
    def update(self, rewards, action_log_probs, entropies, state_values, method, n_steps=1):
        # Stack various tensors for batch processing.
        entropies = torch.stack(entropies)
        state_values = torch.stack(state_values).squeeze()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        action_log_probs = torch.stack(action_log_probs)

        # Calculate returns and advantages using different methods.
        if method == "Bootstrapping":
            # Calculate discounted returns using bootstrapping.
            returns = []
            discounted_reward = 0
            for i in range(len(rewards)):
                discounted_reward = rewards[i] + self.gamma * discounted_reward if i < len(rewards) - n_steps else rewards[i]
                returns.append(discounted_reward)
            returns = torch.tensor(returns)
            advantage_estimates = returns + state_values - state_values

        elif method == "Baseline subtraction":
            # Calculate returns using a baseline to reduce variance.
            Q_values = torch.zeros_like(rewards)
            Q_value = 0
            for i in reversed(range(len(rewards))):
                Q_value = rewards[i] + self.gamma * Q_value
                Q_values[i] = Q_value
            advantage_estimates = Q_values - state_values

        elif method == "Bootstrapping + baseline subtraction":
            # Combination of bootstrapping and baseline subtraction.
            returns = []
            discounted_reward = 0
            for i in range(len(rewards)):
                discounted_reward = rewards[i] + self.gamma * discounted_reward if i < len(rewards) - n_steps else rewards[i]
                returns.append(discounted_reward)
            returns = torch.tensor(returns)
            advantage_estimates = returns - state_values

        # Calculate the losses for actor and critic using the advantages.
        actor_of_loss = -(action_log_probs * advantage_estimates.detach()).mean() - self.entropy_coefficient * entropies.mean()
        critic_of_loss = 0.5 * (advantage_estimates.pow(2)).mean()
        
        # Zero gradients, perform backpropagation, and update weights.
        self.optimizer_for_actor.zero_grad()
        self.optimizer_for_critic.zero_grad()
        actor_of_loss.backward()
        critic_of_loss.backward()
        self.optimizer_for_actor.step()
        self.optimizer_for_critic.step()

    # Method to choose an action based on the current policy.
    def choose_action(self, state, greedy=False):
        state = torch.FloatTensor(state).unsqueeze(0)  # Prepare state input.
        probs = self.actor(state)  # Get action probabilities from the actor.
        dist = torch.distributions.Categorical(probs)  # Create a distribution to sample from.
        
        if greedy:
            a = torch.argmax(probs).item()  # Choose the action with the highest probability.
        else:
            a = dist.sample().item()  # Sample an action according to the distribution.
        
        action_log_prob = dist.log_prob(torch.tensor(a))
        entropy = dist.entropy()
        
        return a, action_log_prob, entropy  # Return the chosen action and its log probability and entropy.

# Main training and evaluation function.
def ActorCritic(num_timesteps, method, learning_rate, gamma, entropy, n_steps=5):
    environment = gym.make('LunarLander-v2')  # Create an instance of the Lunar Lander environment.
    input_dim = environment.observation_space.shape[0]
    output_dim = environment.action_space.n
    model_agent = Agent_ActorCritic(input_dim, output_dim, learning_rate, gamma, entropy)
    timesteps = 0
    eval_interval = 5000
    eval_returns = []
    eval_timesteps = []
    episode = 0
    
    while timesteps < num_timesteps:
        rewards = []
        entropies = []
        total_reward = 0
        action_log_probs = []
        state_values = []
        is_done = False
        state, _  = environment.reset()

        while not is_done:
            timesteps += 1
            action, log_prob, entropy = model_agent.choose_action(state)
            next_state, reward, is_done, *extra = environment.step(action)
            value = model_agent.critic(torch.FloatTensor(state).unsqueeze(0))

            action_log_probs.append(log_prob)
            state_values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            total_reward += reward
            state = next_state

            eval_interation = timesteps
            if eval_interation % eval_interval == 0:
                mean_return = evaluate(model_agent, environment)
                eval_returns.append(mean_return)
                eval_timesteps.append(timesteps)
                print(f"Timestep {timesteps + 1}: Reward {mean_return}")  

            if is_done:
                episode += 1
                model_agent.update(rewards, action_log_probs, entropies, state_values, method, n_steps)
                print(f"Timestep {timesteps + 1}: Total Reward = {total_reward} using method: {method}")
                break

    environment.close()
    print(eval_returns)
    return eval_returns, eval_timesteps

# Function to test different learning methods on the environment.
def test():
    methods = ["Bootstrapping + baseline subtraction", "Bootstrapping", "Baseline subtraction"]
    smoothing_window = 11
    num_timesteps = 1500000
    learning_rate = 0.001
    gamma = 0.95
    entropy = 0.1
    n_steps = 5  # Define the number of steps for bootstrapping
    plt.figure(figsize=(10, 5))
    for method in methods:
        print(f"Running experiment with {method}")
        eval_returns, eval_timesteps = ActorCritic(num_timesteps, method, learning_rate, gamma, entropy, n_steps)
        smoothed_returns = np.convolve(eval_returns, np.ones(smoothing_window)/smoothing_window, mode='valid')
        smoothed_timesteps = eval_timesteps[:len(smoothed_returns)]
        plt.plot(smoothed_timesteps, smoothed_returns, label=method)
        print(eval_returns, eval_timesteps)

    plt.title('Actor-Critic Variants on Lunar Lander')  # Set the plot title.
    plt.xlabel('Timesteps')  # Label the x-axis.
    plt.ylabel('Average Return')  # Label the y-axis.
    plt.legend()  # Add a legend to explain the plot lines.
    plt.show()

if __name__ == '__main__':
    test()
