import pygame
import math
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, regularizers
from tensorflow.keras.models import load_model
import pandas as pd
import os
import json
import re

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Call this function at the beginning of your script
set_seed(42)


class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.font = pygame.font.SysFont('Arial', 24)  # Create a Font object
        self.screen_width, self.screen_height = 800, 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.FPS = 60
        self.last_asteroid_time = 0  # Tracks the time since the last asteroid was added
        self.asteroid_interval = 2500  # Milliseconds between asteroid spawns
        self.max_asteroids = 10  # Maximum number of asteroids allowed
        self.player = Player(self)
        self.asteroids = []
        self.bullets = []
        self.score = 0
        self.target_player_next = False
        self.asteroid_hit = False
        self.small_asteroid_hit = False
        self.closest_asteroid = None
        self.closest_asteroid_distance = float('inf')
        self.closest_asteroid_destroyed = False
        
    def run_with_random_agent(self, agent):
        self.reset()
        while self.running:
            action = agent.get_action()  # Get a random action
            self.step(action)  # Execute the chosen action in the game
            self.check_collisions()  # Check for collisions between the player and asteroids
            
            if not self.running:  # If a collision has occurred
                pygame.quit()  # Cleanly exit Pygame
                return  # Exit the function, effectively ending the game
            
            self.render()  # Render the game state to the screen
    
            for event in pygame.event.get():  # Allow closing the window
                if event.type == pygame.QUIT:
                    self.running = False
    
            self.clock.tick(self.FPS)  # Maintain the game's FPS
            
    def run_with_dqn_agent(self, agent, num_episodes = 10):
        for episode in range(num_episodes):
            state = self.reset()
            state = np.reshape(state, [1, agent.state_size])
            total_reward = 0
            
            while self.running:
                self.render()
                
                action = agent.act(state)
                next_state, reward, done, _ = self.step(action)
                next_state = np.reshape(next_state, [1, agent.state_size])
                self.check_collisions()
                
                state = next_state
                total_reward += reward
                
                
                for event in pygame.event.get():  # Allow closing the window
                    if event.type == pygame.QUIT:
                        self.running = False
                
                self.clock.tick(self.FPS)
                
                if done:
                    print(f'Episode: {episode + 1}, Score: {self.score}, Total Reward: {total_reward}')
                    
        pygame.quit()
                    
                    
        
    def reset(self):
        """Resets the game to an initial state and returns an initial observation."""
        self.asteroids = []
        self.bullets = []
        self.player = Player(self)
        self.score = 0
        self.running = True
        # Initialize other necessary components for a reset
        # Return an initial observation
        return self.get_state()

    def step(self, action):
        self.handle_events()  # Optional, might be removed for faster training
        self.handle_action(action)  # Update based on action
        self.update()  # Update game objects and check collisions

        observation = self.get_state()  # Get the new game state
        reward = self.get_reward()  # Calculate the reward
        done = not self.running  # Check if the game is over
        info = {}  # Additional info for debugging, not used here
    
        return observation, reward, done, info

    def render(self):
        """Renders the current game state to the screen."""
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw game entities
        self.draw()
        
        # Update the display
        pygame.display.flip()

    def normalize_position(self, position):
        """Normalize position to a range between 0 and 1."""
        return [position[0] / self.screen_width, position[1] / self.screen_height]

    def get_nearest_asteroid_info(self):
        """Get the position, size, and velocity of the nearest 5 asteroids, normalized."""
        info = []
        if self.asteroids:
            # Calculate distances to the player for each asteroid
            distances = [(ast, math.sqrt((ast.position[0] - self.player.position[0]) ** 2 + (ast.position[1] - self.player.position[1]) ** 2)) for ast in self.asteroids]
            # Sort asteroids by their distance to the player
            distances.sort(key=lambda x: x[1])
            # Get up to the nearest 5 asteroids
            for i in range(min(10, len(distances))):
                ast = distances[i][0]
                normalized_position = self.normalize_position(ast.position)
                normalized_size = ast.size / max(self.screen_width, self.screen_height)
                normalized_velocity = [ast.velocity[0] / 5, ast.velocity[1] / 5]  # Assuming 5 is the maximum speed for normalization
                info.extend(normalized_position + [normalized_size] + normalized_velocity)
        # If there are fewer than 20 asteroids, fill the remaining space with zeros
        while len(info) < 10 * 5:  # 20 asteroids * (x, y, size, vel_x, vel_y)
            info.extend([0, 0, 0, 0, 0])
        return info
    
    def find_closest_asteroid(self):
        closest_asteroid = None
        min_distance = float('inf')
        for asteroid in self.asteroids:
            distance = math.sqrt((asteroid.position[0] - self.player.position[0]) ** 2 + (asteroid.position[1] - self.player.position[1]) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_asteroid = asteroid
        return closest_asteroid, min_distance

    def get_state(self):
        """Return the current state of the game as perceived by the agent."""
        player_pos_normalized = self.normalize_position(self.player.position)
        player_angle_normalized = [self.player.angle / 360]
        player_velocity_normalized = [self.player.velocity[0] / self.player.max_speed, self.player.velocity[1] / self.player.max_speed]
        # Update this to include player's velocity
        state = player_pos_normalized + player_angle_normalized + player_velocity_normalized + self.get_nearest_asteroid_info()
        return np.array(state)

    
    def get_reward(self):
        reward = 0

        if self.asteroid_hit:
            reward += self.score * 10
            self.asteroid_hit = False
            
        if self.small_asteroid_hit:
            reward += self.score * 25
            
        if self.closest_asteroid_destroyed:
            reward += self.score * 20
            self.closest_asteroid_destroyed = False
        
# =============================================================================
#         if self.score % 10 == 0 and self.score > 0:
#             reward += self.score * 100
# =============================================================================
        
        # Add a small reward for moving forward
# =============================================================================
#         if self.player.moving_forward:
#             reward += 0.1
#             
#         if self.player.rotating_left:
#             reward += 0.05
#         if self.player.rotating_right:
#             reward += 0.05
# =============================================================================
        
        # Check for close encounters and apply a negative reward
        close_encounter_penalty = -15  # Penalty amount for getting too close to an asteroid
        min_safe_distance = 40  # Define what you consider a close encounter (in pixels)
    
        for asteroid in self.asteroids:
            distance = math.sqrt((asteroid.position[0] - self.player.position[0]) ** 2 + 
                                 (asteroid.position[1] - self.player.position[1]) ** 2)
            if distance < min_safe_distance:
                reward += close_encounter_penalty
                break  # Only apply the penalty once per step, regardless of the number of close encounters
        
        # Negative reward for game over
        if not self.running:
            reward -= 500
        return reward


    def run(self):
        while self.running:
            self.handle_events()
            self.handle_action()
            self.update()
            self.draw()
            self.clock.tick(self.FPS)
        pygame.quit()


    def cap_player_speed(self):
        speed = math.sqrt(self.player.velocity[0] ** 2 + self.player.velocity[1] ** 2)
        if speed > self.player.max_speed:
            scale = self.player.max_speed / speed
            self.player.velocity[0] *= scale
            self.player.velocity[1] *= scale

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
    def handle_action(self, action):
        # Handles player actions and applies game logic accordingly.
        self.player.execute_action(action)

    def update(self):
        # Update player
        self.player.update()
        
        # Cap the player's speed to ensure it doesn't exceed the maximum allowed speed
        self.cap_player_speed()
    
        # Update bullets
        self.update_bullets()
    
        # Update asteroids
        self.update_asteroids()
    
        # find closest asteroid
        self.closest_asteroid, self.closest_asteroid_distance = self.find_closest_asteroid()
        
        # Check for collisions
        self.check_collisions()



    def update_bullets(self):
        to_remove = []
        for bullet in self.bullets:
            bullet.update()
            if not bullet.is_alive():
                to_remove.append(bullet)
        for bullet in to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
    
    def update_asteroids(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_asteroid_time > self.asteroid_interval and len(self.asteroids) < self.max_asteroids:
            self.add_asteroid()
            self.last_asteroid_time = current_time
        for asteroid in self.asteroids:
            asteroid.update()
    
    def check_collisions(self):
        to_remove_bullets = []
        to_remove_asteroids = []
        for bullet in self.bullets:
            for asteroid in self.asteroids:
                if bullet.collides_with(asteroid):
                    self.asteroid_hit = True
                    if asteroid.size > 15:
                        self.score += 1
                        self.split_asteroid(asteroid)
                    if asteroid.size <= 15:
                        self.score += 2
                        self.small_asteroid_hit = True
                    if asteroid == self.closest_asteroid:
                            self.closest_asteroid_destroyed = True
                    to_remove_asteroids.append(asteroid)
                    to_remove_bullets.append(bullet)
                    # self.score += 1
                    break
        for bullet in to_remove_bullets:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
        for asteroid in to_remove_asteroids:
            if asteroid in self.asteroids:
                self.asteroids.remove(asteroid)

        for asteroid in self.asteroids:
            if asteroid.collides_with(self.player):
                self.running = False
                print("Game Over: Ship hit by an asteroid.")
                break
    

    def split_asteroid(self, asteroid):
        # Calculate the new size for the split asteroids
        new_size = asteroid.size // 2
        
        # Generate two new velocities randomly for the split asteroids
        for _ in range(2):
            new_velocity = [
                asteroid.velocity[0] + random.uniform(-2, 2),
                asteroid.velocity[1] + random.uniform(-2, 2)
            ]
            new_asteroid = Asteroid(self, asteroid.position[:], new_size, new_velocity)
            self.asteroids.append(new_asteroid)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.player.draw()
        for asteroid in self.asteroids:
            asteroid.draw()
        for bullet in self.bullets:
            bullet.draw()
            
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))  # Position the score in the top-left corner
        
        pygame.display.flip()
        
        

    def add_asteroid(self):
        if len(self.asteroids) < self.max_asteroids:
            size = random.randint(15, 30)  # Assuming these are the desired min and max sizes
            edge = random.choice(['top', 'bottom', 'left', 'right'])

            # Assign position based on the chosen edge
            if edge == 'top':
                position = [random.randint(0, self.screen_width), 0]
            elif edge == 'bottom':
                position = [random.randint(0, self.screen_width), self.screen_height]
            elif edge == 'left':
                position = [0, random.randint(0, self.screen_height)]
            else:  # 'right'
                position = [self.screen_width, random.randint(0, self.screen_height)]

            # Toggle the targeting behavior and set initial velocity accordingly
            if self.target_player_next:
                # Calculate the direction towards the player
                direction = [self.player.position[0] - position[0], self.player.position[1] - position[1]]
                # Normalize the direction vector
                norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                velocity = [direction[0] / norm * 2, direction[1] / norm * 2]  # Adjust speed as needed
            else:
                # Use random velocity for non-targeting asteroids
                velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]

            # Create and add the asteroid
            self.asteroids.append(Asteroid(self, position, size, velocity))

            # Toggle the targeting behavior for the next asteroid
            self.target_player_next = not self.target_player_next







class Player:
    def __init__(self, game):
        self.game = game
        self.position = [self.game.screen_width // 2, self.game.screen_height // 2]
        self.size = 5
        self.angle = 0
        self.velocity = [0, 0]
        self.acceleration = 0.1
        self.rotation_speed = 10
        self.max_speed = 5
        self.friction = 0.99
        self.last_bullet_time = 0  # Time the last bullet was fired
        self.bullet_cooldown_ms = 250  # 1000 ms divided by 4 bullets per second
        self.moving_forward = False
        self.rotating_left = False
        self.rotating_right = False
        self.shooting = False


    def shoot(self):
        # Get the current time to check against the last bullet shot time
        current_time = pygame.time.get_ticks()
        if current_time - self.last_bullet_time >= self.bullet_cooldown_ms:
            self.last_bullet_time = current_time  # Update the last bullet time

            # Calculate bullet's starting position based on the player's position and angle
            bullet_position = [self.position[0], self.position[1]]
            bullet_angle = self.angle

            # Create a new bullet and add it to the game's bullet list
            new_bullet = Bullet(self.game, bullet_position, bullet_angle)
            self.game.bullets.append(new_bullet)
    
    def apply_friction(self):
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction


    def wrap_position(self):
        # Wrap around screen edges
        if self.position[0] < 0:
            self.position[0] = self.game.screen_width
        elif self.position[0] > self.game.screen_width:
            self.position[0] = 0

        if self.position[1] < 0:
            self.position[1] = self.game.screen_height
        elif self.position[1] > self.game.screen_height:
            self.position[1] = 0
    
    def move_forward(self):
        # Apply acceleration in the direction the player is facing
        self.velocity[0] += self.acceleration * math.cos(math.radians(self.angle))
        self.velocity[1] += self.acceleration * math.sin(math.radians(self.angle))
        
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale
    
    
    def rotate_left(self):
        self.angle -= self.rotation_speed
        
    def rotate_right(self):
        self.angle += self.rotation_speed
    
        
    def cap_speed(self):
        # Calculate the current speed
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        
        # If the current speed exceeds the max speed, scale down the velocity
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale

        
        
    def execute_action(self, action):
        # Reset action states
        self.moving_forward = False
        self.rotating_left = False
        self.rotating_right = False
        
        # Translate action to flags
        if action == 1:
            self.moving_forward = True
        elif action == 2:
            self.moving_forward = True
            self.shooting = True
        elif action == 3:
            self.rotating_left = True
        if action == 4:
            self.rotating_left = True
            self.shooting = True
        elif action == 5:
            self.rotating_right = True
            self.shooting = True
        elif action == 6:
            self.rotating_right = True
            self.shooting = True
        elif action == 7:
            self.shooting = True
        
        # Execute actions based on flags
        if self.moving_forward and not self.shooting:
            self.move_forward()
        if self.moving_forward and self.shooting:
            self.move_forward()
            self.shoot()
        if self.rotating_left and not self.shooting:
            self.rotate_left()
        if self.rotating_left and self.shooting:
            self.rotate_left()
            self.shoot()
        if self.rotating_right and not self.shooting:
            self.rotate_right()
        if self.rotating_right and self.shooting:
            self.rotate_right()
            self.shoot()
        if self.shooting and not self.moving_forward and not self.rotating_left and not self.rotating_right:
            self.shoot()
        
        if not self.moving_forward:
            self.apply_friction()
        
        self.cap_speed()
        self.wrap_position()
    
    def update(self):
        # Apply current velocity to update position
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
    
        # Apply friction if not moving forward
        # This can be checked by seeing if the acceleration applied is zero
        # Assuming a method or a flag that checks if forward acceleration is applied
        if not self.moving_forward:  # You need to define how you set this flag based on your agent's actions
            self.apply_friction()
    
        # Cap the speed to the maximum speed
        self.cap_speed()
    
        # Screen wrapping
        self.wrap_position()
  

    def draw(self):
        radians = math.radians(self.angle)
        tip = (self.position[0] + 15 * math.cos(radians), self.position[1] + 15 * math.sin(radians))
        left_base = (self.position[0] + 15 * math.cos(radians + math.pi * 5/6), self.position[1] + 15 * math.sin(radians + math.pi * 5/6))
        right_base = (self.position[0] + 15 * math.cos(radians - math.pi * 5/6), self.position[1] + 15 * math.sin(radians - math.pi * 5/6))
        pygame.draw.polygon(self.game.screen, (255, 255, 255), [tip, left_base, right_base])
        



class Bullet:
    def __init__(self, game, position, angle):
        self.game = game
        self.position = list(position)  # Make a copy of the position list to avoid modifying the original
        self.angle = angle
        self.speed = 7.5  # Speed of the bullet
        self.velocity = [
            math.cos(math.radians(self.angle)) * self.speed,
            math.sin(math.radians(self.angle)) * self.speed
        ]
        self.lifetime = 75  # Frames until the bullet is automatically removed

    def update(self):
        # Move the bullet based on its velocity
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        # Decrease the lifetime of the bullet
        self.lifetime -= 1
        self.wrap_position()

    def draw(self):
        # Draw the bullet as a small circle or a point for simplicity
        start_pos = self.position
        end_pos = (start_pos[0] + self.speed * math.cos(math.radians(self.angle)),
                   start_pos[1] + self.speed * math.sin(math.radians(self.angle)))
        pygame.draw.line(self.game.screen, (255, 255, 255), start_pos, end_pos, width = 2)
        # pygame.draw.circle(self.game.screen, (255, 255, 255), (int(self.position[0]), int(self.position[1])), 3)

    def is_alive(self):
        # Check if the bullet is still within the screen bounds
        return self.lifetime > 0

    
    def wrap_position(self):
        """Wrap the position to appear on the opposite side of the screen when it goes off the edge."""
        if self.position[0] < 0:
            self.position[0] = self.game.screen_width
        elif self.position[0] > self.game.screen_width:
            self.position[0] = 0
        if self.position[1] < 0:
            self.position[1] = self.game.screen_height
        elif self.position[1] > self.game.screen_height:
            self.position[1] = 0
        return self.position

    def collides_with(self, asteroid):
        # Check for collision with an asteroid using simple distance-based collision detection
        dx = self.position[0] - asteroid.position[0]
        dy = self.position[1] - asteroid.position[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance < asteroid.size








class Asteroid:
    def __init__(self, game, position, size, velocity = None):
        self.game = game
        self.position = position
        self.size = size
        # Randomize velocity based on size for some variability
        if velocity is None:
            self.velocity = [
                random.uniform(-2, 2) * (30 / size),  # Smaller asteroids move faster
                random.uniform(-2, 2) * (30 / size)
                ]
        else:
            # Use specified velocity for split asteroids
            self.velocity = velocity


    def update(self):
        
           
        # Update asteroid position based on velocity
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        
        
        # Screen wrapping
        self.wrap_position()
        
        
    def wrap_position(self):
        """Wrap the position to appear on the opposite side of the screen when it goes off the edge."""
        if self.position[0] < 0:
            self.position[0] = self.game.screen_width
        elif self.position[0] > self.game.screen_width:
            self.position[0] = 0
        if self.position[1] < 0:
            self.position[1] = self.game.screen_height
        elif self.position[1] > self.game.screen_height:
            self.position[1] = 0
        return self.position

    def draw(self):
        # Draw the asteroid as a circle for simplicity
        pygame.draw.circle(self.game.screen, (255, 255, 255), self.position, self.size)

    def collides_with(self, other):
        # Simple collision detection: check if distance between centers is less than sum of radii
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance < self.size + 5














class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        return random.randint(0, self.action_size - 1)



class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer_size=100000, update_target_every=5, model_file_path = None, training_state_file_path = None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_buffer_size)  # Increase replay buffer size
        self.gamma = 0.95  # discount rate
        if training_state_file_path:
            with open(training_state_file_path, 'r') as f:
                training_state = json.load(f)
                self.epsilon = training_state['epsilon']
                print('Training state load from', training_state_file_path)
        else:
            self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_every = update_target_every  # Update target network every # episodes
        if model_file_path:
            self.model = load_model(model_file_path)
            print('Model loaded from', model_file_path)
        else:
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Initialize target model weights
        self.target_update_counter = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        
        # Hidden layers
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        
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
        
    def save_training_state(self, file_path, epsilon):
        with open(file_path, 'w') as f:
            json.dump({'epsilon': epsilon}, f)
        
    def load_from_file(cls, file_path, state_size, action_size):
        model = load_model(file_path)
        agent = cls(state_size, action_size)
        agent.model = model
        print(f"Model loaded from {file_path}")
        return agent

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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



# =============================================================================
# if __name__ == "__main__":
#     game = Game()
#     random_agent = RandomAgent(action_size=8)  # 8 possible actions
#     game.run_with_random_agent(random_agent)
# 
# =============================================================================

# reward_per_episode = []
batch_size = 32
best_score = -500

env = Game()  # Your game environment

# Assuming the player state consists of player_pos_normalized (2), player_angle_normalized (1), 
# player_velocity_normalized (2), and the asteroid info (20 asteroids * 5 properties each)
state_size = 2 + 1 + 2 + (10 * 5)
# state_size = len(env.get_state())
action_size = 8  # Based on the game's action space
# agent = DQNAgent(state_size, action_size)


EPISODES = 900
SAVE_INTERVAL = 25  # Save the model every 50 episodes
RENDER_EVERY_N_EPISODES = 25  # Adjust this number based on your preference
render = False

# save_dir = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning"  # Directory to save models

base_dir = "D:/AnacondaProjects"  # Change this to your specific path
folder_name = "DQN_Models"
folder_path = os.path.join(base_dir, folder_name)

# Create the directory if it does not exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


model_path = f'{folder_path}/model_episode_625_ten_asteroids_movement_incentive.h5'
training_state_path = f'{folder_path}/model_episode_625_ten_asteroids_movement_incentive_training_state.json'

agent = DQNAgent(state_size, action_size, model_file_path = model_path, training_state_file_path = training_state_path)

episode_num = re.findall(r'_(\d+)_', model_path)
episode_num = int(episode_num[0])

for episode in range(episode_num, EPISODES + 1):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    
    for time in range(1, 7501):
        if render:
            if episode % RENDER_EVERY_N_EPISODES == 0 or episode == 1:
                env.render()  # Render the environment
        # Your existing logic for taking actions and updating the environment
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        if time == 7500:  # This is your maximum step limit
            done = True  # Manually set done to True to indicate end of episode
            # Optionally adjust the reward here for surviving the full episode
            reward += 1500.1  # survival_bonus
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            # reward_per_episode.append(total_reward)  # Append the score at the end of the episode
            print(f"Episode: {episode}/{EPISODES}, score: {env.score} Total Reward: {round(total_reward, 2)}, e: {agent.epsilon:.2}")
            break
    
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Save the model every SAVE_INTERVAL episodes
    if episode % SAVE_INTERVAL == 0 or episode == 1:
        agent.save_model(f'{folder_path}/model_episode_{episode}_ten_asteroids_movement_incentive.h5')
        agent.save_training_state(f'{folder_path}/model_episode_{episode}_ten_asteroids_movement_incentive_training_state.json', agent.epsilon)
    if total_reward > best_score:
        best_score = total_reward
        # Save the current model as the new best model
        agent.save_model(f'{folder_path}/model_best_movement_incentive.h5')
        print(f"New best model at {episode} saved with score: {best_score}")
 
# =============================================================================
# df_dict = {'Episode': list(range(1, len(reward_per_episode) + 1)), 'Reward': reward_per_episode}
# df = pd.DataFrame(df_dict)
# df.to_csv(f'{folder_path}/DQN_Reward_Per_Episode.csv')
# =============================================================================

# print(df_dict)