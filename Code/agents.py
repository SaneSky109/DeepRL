from collections import deque
import json
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, regularizers
from tensorflow.keras.models import load_model
import numpy as np
import random
import pickle

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        return random.randint(0, self.action_size - 1)

class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer_size=250000, update_target_every=10, model_file_path = None, target_model_file_path = None, training_state_file_path = None, replay_buffer_file_path = None, evaluation = False):
        self.state_size = state_size
        self.action_size = action_size
        if replay_buffer_file_path:
            self.load_replay_buffer(replay_buffer_file_path)
        else:
            self.memory = deque(maxlen=replay_buffer_size)
        self.gamma = 0.95  # discount rate
        if training_state_file_path:
            self.load_training_state(training_state_file_path)
        elif evaluation:
            self.epsilon = 0.01
        else:
            self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00000001 # 0.000001 used through 4500
        self.update_target_every = update_target_every  # Update target network every # episodes
        if model_file_path:
            self.model = load_model(model_file_path)
            print('Model loaded from ', model_file_path)
        else:
            self.model = self._build_model()
        if target_model_file_path:
            self.target_model = load_model(target_model_file_path)
            print('Target model loaded from ', target_model_file_path)
        else:
            self.target_model = self._build_model()
        self.update_target_model()  # Initialize target model weights
        self.target_update_counter = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(512, input_dim=self.state_size, activation='relu'))
        
        # Hidden layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """Update weights of the target model to match the primary model."""
        self.target_model.set_weights(self.model.get_weights())

    
    def save_model(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")
        
    def save_target_model(self, file_path):
        self.target_model.save(file_path)
        print(f"Model saved to {file_path}")
        
    def save_training_state(self, file_path, epsilon):
        with open(file_path, 'w') as f:
            json.dump({'epsilon': epsilon}, f)
            print(f"Model saved to {file_path} with epsilon {epsilon}")
            
    def load_training_state(self, file_path):
        with open(file_path, 'r') as f:
            training_state = json.load(f)
            self.epsilon = training_state.get('epsilon', 1.0) # Default to 1.0 if not found
                
    def save_best_model_reward(self, file_path, reward, episode):
        with open(file_path, 'w') as f:
            json.dump({'reward': reward, 'episode': episode}, f)
        print(f"Best model reward saved to {file_path}")
            
    def load_best_model_reward(self, file_path):
        with open(file_path, 'r') as f:
            reward = json.load(f)
        print(f"Best model reward loaded with reward value: {reward.get('reward', -500)}")
        return reward.get('reward', -500) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def save_replay_buffer(self, file_path):
        """Saves the replay buffer to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f"Replay buffer saved to {file_path}")
        
    def load_replay_buffer(self, file_path):
        """Loads the replay buffer from a file."""
        try:
            with open(file_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Replay buffer loaded from {file_path}")
        except FileNotFoundError:
            print(f"No replay buffer file found at {file_path}. Starting with an empty replay buffer.")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose = 0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose = 0)[0]))
            target_f = self.model.predict(state, verbose = 0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.update_target_model()
            self.target_update_counter = 0