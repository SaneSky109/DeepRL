# Deep Learning Project: Asteroid Clone Agent

This repository hosts an Asteroids game clone powered by a Deep Q-Learning Network (DQN) agent, created using Python, Pygame, and TensorFlow. The project aims to demonstrate the application of deep reinforcement learning in controlling the player's spacecraft to avoid and destroy asteroids.

## Overview
The game environment is built using Pygame, where the player controls a spaceship to navigate through a field of asteroids. The primary objective is to avoid collisions while destroying asteroids using bullets. The player's actions include moving forward, rotating left, right, and shooting. The DQN agent, implemented using TensorFlow, learns to play the game by interacting with the environment, receiving rewards based on its actions.

## Features
**Pygame Environment:** A custom game environment for the classic Asteroids game.
**Deep Q-Learning Agent:** Utilizes a neural network to learn optimal policies for game playing.
**Reinforcement Learning:** Implements reinforcement learning principles, including exploration vs. exploitation, through an epsilon-greedy strategy.
**Replay Buffer:** Improves learning efficiency and stability by storing and replaying past experiences.
**Target Network:** Enhances training stability by using a separate target network.
