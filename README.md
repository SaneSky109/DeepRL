# Deep Learning Project: Asteroid Clone Agent

This repository hosts an Asteroids game clone powered by a Deep Q-Learning Network (DQN) agent, created using Python, Pygame, and TensorFlow. The project aims to demonstrate the application of deep reinforcement learning in controlling the player's spacecraft to avoid and destroy asteroids.

## Overview
The game environment is built using Pygame, where the player controls a spaceship to navigate through a field of asteroids. The primary objective is to avoid collisions while destroying asteroids using bullets. The player's actions include moving forward, rotating left, right, and shooting. The DQN agent, implemented using TensorFlow, learns to play the game by interacting with the environment, receiving rewards based on its actions.

## Features
* **Pygame Environment:** A custom game environment for the classic Asteroids game.
* **Deep Q-Learning Agent:** Utilizes a neural network to learn optimal policies for game playing.
* **Reinforcement Learning:** Implements reinforcement learning principles, including exploration vs. exploitation, through an epsilon-greedy strategy.
* **Replay Buffer:** Improves learning efficiency and stability by storing and replaying past experiences.
* **Target Network:** Enhances training stability by using a separate target network.

## How it Works

### Game Environment
The game environment (Game) simulates the Asteroids game, including player movement, asteroid generation, collision detection, and scoring. It also provides methods for resetting the game state, performing actions, and rendering the game window.

### DQN Agent
The DQN agent (DQNAgent) is responsible for deciding actions based on the game state. It utilizes a neural network to estimate Q-values for each action given the current state. The agent improves its policy over time through training episodes, where it explores the action space and learns from the outcomes of its actions.

### Training
The agent is trained over a series of episodes, with the game environment resetting at the beginning of each episode. The training process involves the agent interacting with the environment, storing experiences in the replay buffer, and periodically training the neural network using batches of sampled experiences. 

The agent's learning progress can be monitored through the reward it accumulates over each episode. A successful training process should show an increasing trend in the total reward as the agent learns to avoid asteroids and destroy them more effectively.

## Improvements for Future Work
The project's modular design allows for various enhancements and experimentation. Future work could explore different neural network architectures, hyperparameter tuning, and advanced RL algorithms like Double DQN, Dueling DQN, or Prioritized Experience Replay to further improve the agent's performance.









## Customization and Improvement
* **Network Architecture:** Modify the DQN's neural network architecture in DQNAgent._build_model to experiment with different layer configurations and regularization techniques.
**Reward System:** Adjust the reward logic in Game.get_reward to incentivize or penalize different behaviors.
**Hyperparameters:** Experiment with different settings for learning rate, epsilon decay, replay buffer size, and batch size to optimize learning.





















## Setup

To run this project, ensure you have Python 3.10 installed along with the required libraries: Pygame for the game environment and TensorFlow for the DQN agent.

1. Install dependencies

```python
pip install pygame tensorflow
```

2. Clone the repository to your local machine.

3. Open '' and change file path to where the .h5 file is saved
