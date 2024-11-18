# LunarLander-RL

A policy-based reinforcement learning project implementing and analyzing algorithms on the **LunarLander-v2** environment from OpenAI Gym. This project explores the performance of REINFORCE and Actor-Critic algorithms (including their variants) to optimize rocket landings, with a focus on hyperparameter tuning and variance reduction.

## Project Overview

This repository contains the implementation of the following policy-based RL algorithms:
1. **REINFORCE (Monte Carlo Policy Gradient)**  
2. **Actor-Critic Variants**:
   - Bootstrapping
   - Baseline Subtraction
   - Bootstrapping + Baseline Subtraction

Key features include:
- Exploration techniques using **entropy regularization**.
- Hyperparameter tuning for optimal performance.
- Comparison of algorithm efficiency in the LunarLander environment.

---

## LunarLander Environment

The **LunarLander-v2** environment involves landing a rocket safely on a specified landing pad, with a reward structure that promotes precise landings and penalizes crashes or fuel wastage.

- **Action Space**: Four discrete actions
  - Do nothing
  - Fire the left orientation engine
  - Fire the main engine
  - Fire the right orientation engine
- **Observation Space**: 8-dimensional state vector
- **Rewards**:
  - +100–140 for a successful landing.
  - -100 for crashing.
  - +10 for each leg touching the ground.
  - Penalties for fuel consumption.

---

## Algorithms Implemented

### 1. REINFORCE
A Monte Carlo Policy Gradient method that updates policies based on full-episode returns.  
[Code: `Reinforce.py`]

### 2. Actor-Critic
Combines policy-based and value-based methods to improve learning stability and efficiency. Variants implemented:
- **Bootstrapping**: Reduces variance by leveraging partial trajectories.
- **Baseline Subtraction**: Stabilizes learning with a value-function baseline.
- **Bootstrapping + Baseline Subtraction**: Balances variance reduction with learning efficiency.  
[Code: `ActorCritic.py`]

### Exploration Technique
All algorithms use **entropy regularization** to encourage exploration and prevent premature convergence.

---

## Results

### Key Observations:
- **REINFORCE** exhibited high variance but faster learning compared to Actor-Critic.
- Actor-Critic with **Bootstrapping + Baseline Subtraction** balanced variance reduction with stable returns.
- Entropy regularization improved exploration and stabilized learning.

### Performance Comparison:
Both algorithms achieved positive episodic returns, demonstrating their ability to learn effective policies. However, REINFORCE outperformed Actor-Critic in terms of learning speed and episodic rewards in the LunarLander environment.

---

## Repository Structure

```plaintext
.
├── ActorCritic.py       # Actor-Critic algorithm implementation
├── Reinforce.py         # REINFORCE algorithm implementation
├── Experiment.py        # Experimental setup and execution scripts
├── Helper.py            # Utility functions for training and evaluation
├── README.md            # Project documentation

```

## How to Run

### Install Dependencies:

Ensure you have **Python 3.x** installed along with the required libraries:

```bash
pip install gym pygame
```

### Run the Code:
Execute the provided Python scripts to train agents or visualize results:

```bash
python Reinforce.py
python ActorCritic.py
```

### Experiment Configuration:
Modify hyperparameters directly in Experiment.py to replicate tuning experiments.

---

## References
- OpenAI Gym: LunarLander-v2
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- Mnih, V. et al. (2016). Asynchronous methods for deep reinforcement learning.


