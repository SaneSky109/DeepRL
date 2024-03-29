# Deep Q-Learning Project: Asteroid Clone Agent

Welcome to the GitHub repository of my Asteroids Game Clone project, where I explore the fascinating world of reinforcement learning (RL) by training a Deep Q-Network (DQN) agent to navigate and survive in a custom-built environment simulating the classic Asteroids arcade game. This project is crafted with Python, leveraging the Pygame library for game development and TensorFlow for implementing the DQN agent.



## Project Overview
The project simulates the Asteroids game, challenging the player's spacecraft to avoid and destroy asteroids. The game environment, constructed using Pygame, features dynamic asteroid generation, player movement (forward thrust and rotation), shooting mechanics, and collision detection.

The heart of our project is the DQN agent, developed with TensorFlow. This agent learns to make decisions (move forward, rotate, shoot) based on the current game state, aiming to maximize its survival time and score by avoiding asteroids and shooting them down.

## Game Environment 
The Game class encapsulates the game environment. It initializes the game window, handles game logic (e.g., spawning asteroids, updating game objects, detecting collisions), and renders the game state to the screen. This class provides a realistic and challenging setting for training our DQN agent, with features including:

* Dynamic asteroid spawning and movement
* Player spaceship control (acceleration and rotation)
* Bullet shooting with cooldown mechanics
* Collision detection between the spaceship, asteroids, and bullets
* Score tracking based on asteroid destruction

## DQN Agent
The DQNAgent class represents the reinforcement learning agent. It uses a neural network to approximate the Q-function, mapping state-action pairs to expected rewards. The agent learns an optimal policy over time through exploration (selecting random actions) and exploitation (choosing actions based on learned Q-values), balancing these strategies using an epsilon-greedy approach. Key aspects include:

* Neural network architecture with dense layers and dropout for generalization
* Experience replay mechanism for efficient learning from past actions
* Target network for stable Q-value approximation
* Epsilon decay strategy for gradual shift from exploration to exploitation

## Training and Evaluation
Training involves running the game environment in episodes, where the agent interacts with the environment, makes decisions, and learns from the outcomes. Each episode's total reward and score provide insight into the agent's performance and learning progress. The training loop includes:

* Resetting the environment at the start of each episode
* Selecting actions based on the current state using the DQN agent
* Updating the game state and receiving rewards based on the agent's actions
* Saving the trained model periodically and upon achieving high scores

## Future Enhancements
This project lays the groundwork for further exploration and experimentation in reinforcement learning. Potential improvements include:

Experimenting with different neural network architectures and hyperparameters
Implementing advanced RL algorithms (e.g., Double DQN, Dueling DQN)
Enhancing the game environment for more complex scenarios and learning challenges

Contributions, suggestions, and discussions on extending and improving the project are welcome.












## Getting Started

### Running the Agent with a Pretrained Model
To demonstrate the capabilities of our trained DQN agent without going through the entire training process, you can run the agent using a pretrained model. This section guides you through the steps to load a pretrained model and watch the agent play the game.

#### Prerequisites
Before proceeding, ensure you have a pretrained model file (.h5 format) available. This file contains the weights of the neural network that the agent uses to make decisions. If you have followed the training steps outlined in this project, you should have this file saved in the designated directory folder called 'Model_Files'.

#### Steps to Run the Agent


## Acknowledgments
This project is inspired by the classic Atari Asteroids game and the wealth of research in the field of reinforcement learning. We are grateful to the open-source community for providing the tools and libraries that made this project possible.

Happy learning and exploring!



















## Setup

To run this project, ensure you have Python 3.10 installed along with the required libraries: Pygame for the game environment and TensorFlow for the DQN agent.

1. Install dependencies

```python
pip install pygame tensorflow
```

2. Clone the repository to your local machine.

3. Open '' and change file path to where the .h5 file is saved
