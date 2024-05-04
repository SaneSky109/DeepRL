import pygame
import math
import random
from collections import deque
import numpy as np
from player import Player
from asteroid import Asteroid
import os

class Game:
    
    """
    Represents the main game environment for the Asteroids clone game.

    This class handles game initialization, game loop execution, rendering of game elements,
    handling user inputs, and game state updates. It manages the player, asteroids, bullets,
    and tracks the game score and other statistics.

    Attributes:
        font (pygame.font.Font): Font object for rendering text.
        screen_width (int): Width of the game screen.
        screen_height (int): Height of the game screen.
        screen (pygame.Surface): The main screen surface where game elements are drawn.
        clock (pygame.time.Clock): Clock object used to control game framerate.
        running (bool): Flag indicating if the game is currently running.
        FPS (int): Frames per second the game tries to maintain.
        last_asteroid_time (int): Timestamp of when the last asteroid was spawned.
        asteroid_interval (int): Interval in milliseconds between spawning new asteroids.
        max_asteroids (int): Maximum number of asteroids allowed on the screen at once.
        player (Player): The player's spaceship.
        asteroids (list of Asteroid): List of currently active asteroids.
        bullets (list of Bullet): List of currently active bullets shot by the player.
        score (int): Current score of the player.
        target_player_next (bool): Determines if the next spawned asteroid will target the player.
        asteroid_hit (bool): Flag indicating if an asteroid has been hit.
        small_asteroid_hit (bool): Flag indicating if a small asteroid has been hit.
        closest_asteroid (Asteroid): The closest asteroid to the player.
        closest_asteroid_distance (float): Distance of the closest asteroid to the player.
        closest_asteroid_destroyed (bool): Flag indicating if the closest asteroid was destroyed.

    Methods:
        run_with_random_agent(agent): Runs the game with an agent making random decisions.
        run_with_dqn_agent(agent, num_episodes): Runs the game with a DQN agent for a given number of episodes.
        reset(): Resets the game to an initial state.
        step(action): Executes a given action and updates the game state.
        render(): Renders the current game state to the screen.
        normalize_position(position): Normalizes a position to the screen size.
        get_nearest_asteroid_info(): Retrieves information about the nearest 10 asteroids.
        find_closest_asteroid(): Finds the closest asteroid to the player.
        get_state(): Returns the current state of the game as perceived by an agent.
        get_reward(): Calculates the reward based on the current game state.
        run(): Main game loop that handles events and updates the game state.
        cap_player_speed(): Caps the player's speed to the maximum allowed speed.
        handle_events(): Handles game events like quitting the game.
        handle_action(action): Processes player actions.
        update(): Updates the game state, including player, bullets, and asteroids.
        update_bullets(): Updates the state of bullets, removing expired ones.
        update_asteroids(): Updates the state of asteroids, spawning new ones as needed.
        check_collisions(): Checks for and handles collisions between game elements.
        split_asteroid(asteroid): Splits an asteroid into two smaller ones if applicable.
        draw(): Draws all game elements on the screen.
        add_asteroid(): Adds a new asteroid to the game, potentially targeting the player.
    """
    
    def __init__(self, history_length = 10):
        
        
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.font = pygame.font.SysFont('Arial', 24)  # Create a Font object
        self.screen_width, self.screen_height = 800, 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.FPS = 10
        self.last_asteroid_time = 0  
        self.asteroid_interval = 2500  # Milliseconds between asteroid spawns
        self.max_asteroids = 10  
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
        self.stationary_time_counter = 0
        self.state_history = deque(maxlen = history_length)
        
    def run_with_random_agent(self, agent):
        """
        Runs the game with a given agent making decisions randomly.
    
        Parameters:
            agent: An agent instance capable of deciding actions based on the game state.
        """
        
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
            
    def run_with_dqn_agent(self, agent, num_episodes = 10, screenshot = False):
        """
        Runs the game with a Deep Q-Network (DQN) agent for a specified number of episodes.
    
        Parameters:
            agent: A DQN agent instance for making decisions.
            num_episodes (int): The number of episodes to run the game for.
        """
        
             
        if screenshot:
            for episode in range(num_episodes):
                state = self.reset()
                state = np.reshape(state, [1, agent.state_size])
                total_reward = 0
                frame_count = 0
                capture_every_n_frames = 5
                
                
                while self.running:
                    self.render()
                    
                    if frame_count % capture_every_n_frames == 0:
                        screenshot_path = os.path.join('C:/Users/ericl/OneDrive/Documents/ReinforcementLearning/Screenshots', f"screenshot_{episode}_{frame_count}.png")
                        pygame.image.save(self.screen, screenshot_path)
                    
                    action = agent.act(state)
                    next_state, reward, done, _ = self.step(action)
                    next_state = np.reshape(next_state, [1, agent.state_size])
                    self.check_collisions()
                    
                    state = next_state
                    total_reward += reward
                    frame_count += 1
                    
                    for event in pygame.event.get():  # Allow closing the window
                        if event.type == pygame.QUIT:
                            self.running = False
                    
                    self.clock.tick(self.FPS)
                    
                    if done:
                        print(f'Episode: {episode + 1}, Score: {self.score}, Total Reward: {total_reward}')
        else:
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
        """
        Resets the game to its initial state, clearing the screen of asteroids and bullets, and reinitializing the player.
        
        Returns:
            The initial game state after resetting.
        """
        self.asteroids = []
        self.bullets = []
        self.player = Player(self)
        self.score = 0
        self.running = True

        return self.get_state()

    def step(self, action):
        """
        Executes a given action in the game environment, updating the game state accordingly.
    
        Parameters:
            action: The action to be executed.
    
        Returns:
            A tuple containing the new state, reward, done flag, and additional info.
        """
        self.handle_events()  # Optional, might be removed for faster training
        self.handle_action(action)  # Update based on action
        self.update()  # Update game objects and check collisions

        observation = self.get_state()  # Get the new game state
        reward = self.get_reward()  # Calculate the reward
        done = not self.running  # Check if the game is over
        info = {}  # Additional info for debugging
    
        return observation, reward, done, info

    def render(self):
        """
        Renders the current state of the game to the screen.
        """
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw game entities
        self.draw()
        
        # Update the display
        pygame.display.flip()
        
        # Cap frame rate
        self.clock.tick(self.FPS)

    def normalize_position(self, position):
        """
        Normalizes a position to the scale of the screen dimensions.
    
        Parameters:
            position (list): A list containing x and y coordinates.
    
        Returns:
            A list containing normalized x and y coordinates.
        """
        return [position[0] / self.screen_width, position[1] / self.screen_height]

    def get_nearest_asteroid_info(self):
        """
        Retrieves information about the nearest 10 asteroids to the player.
    
        Returns:
            A list containing the normalized position, size, and velocity of the 10 nearest asteroids.
        """
        
        info = []
        if self.asteroids:
            # Calculate distances to the player for each asteroid
            distances = [(ast, math.sqrt((ast.position[0] - self.player.position[0]) ** 2 + (ast.position[1] - self.player.position[1]) ** 2)) for ast in self.asteroids]
            # Sort asteroids by their distance to the player
            distances.sort(key=lambda x: x[1])
            # Get up to the nearest 10 asteroids
            for i in range(min(10, len(distances))):
                ast = distances[i][0]
                normalized_position = self.normalize_position(ast.position)
                
                # Proximity to edges
                proximity_to_left = normalized_position[0]
                proximity_to_right = 1 - normalized_position[0]
                proximity_to_top = normalized_position[1]
                proximity_to_bottom = 1 - normalized_position[1]
                    
                normalized_size = ast.size / 30 # max size of asteroid
                normalized_velocity = [ast.velocity[0] / 5, ast.velocity[1] / 5]  # 5 is the maximum speed of asteroid for normalization
                info.extend(normalized_position + [normalized_size] + normalized_velocity + [proximity_to_left, proximity_to_right, proximity_to_top, proximity_to_bottom])
        # If there are fewer than 10 asteroids, fill the remaining space with zeros
        while len(info) < 10 * 9:  # 10 asteroids * (pos_x, pos_y, size, vel_x, vel_y)
            info.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        return info
    
    def find_closest_asteroid(self):
        """
        Identifies the closest asteroid to the player.
    
        Returns:
            The closest asteroid and its distance from the player.
        """
        
        closest_asteroid = None
        min_distance = float('inf')
        for asteroid in self.asteroids:
            distance = math.sqrt((asteroid.position[0] - self.player.position[0]) ** 2 + (asteroid.position[1] - self.player.position[1]) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_asteroid = asteroid
        return closest_asteroid, min_distance

    def get_current_state(self):
        """
        Returns the current state of the game as perceived by an agent.
    
        Returns:
            An array representing the current game state.
        """
        
        # Player and Asteroid
        player_pos_normalized = self.normalize_position(self.player.position)
        player_angle_normalized = [self.player.angle / 360]
        player_velocity_normalized = [self.player.velocity[0] / self.player.max_speed, self.player.velocity[1] / self.player.max_speed]
        
        proximity_to_left = player_pos_normalized[0]  # Normalized distance from the left edge
        proximity_to_right = 1 - player_pos_normalized[0]  # Normalized distance from the right edge
        proximity_to_top = player_pos_normalized[1]  # Normalized distance from the top
        proximity_to_bottom = 1 - player_pos_normalized[1]  # Normalized distance from the bottom
        
        state = player_pos_normalized + player_angle_normalized + player_velocity_normalized + [proximity_to_left, proximity_to_right, proximity_to_top, proximity_to_bottom] + self.get_nearest_asteroid_info()
        
        # Stationary counter
        max_stationary_frames = 5
        stationary_time_normalized = [self.stationary_time_counter / max_stationary_frames]
        
        
        # Bullets
        bullet_info = []
        max_bullets = 10
        for bullet in self.bullets[:max_bullets]:
            initial_pos_normalized = self.normalize_position(bullet.initial_position)
            bullet_pos_normalized = self.normalize_position(bullet.position)
            
            # Proximity to edges
            proximity_to_left = bullet_pos_normalized[0]
            proximity_to_right = 1 - bullet_pos_normalized[0]
            proximity_to_top = bullet_pos_normalized[1]
            proximity_to_bottom = 1 - bullet_pos_normalized[1]
            
            bullet_velocity_normalized = [bullet.velocity[0] / 7.5, bullet.velocity[1] / 7.5]  # Bullet max speed is 7.5 for normalization
            bullet_angle_normalized = [bullet.angle / 360]
            bullet_lifetime_normalized = [bullet.lifetime / 75]
            bullet_info.extend(initial_pos_normalized + bullet_pos_normalized + bullet_velocity_normalized + bullet_angle_normalized + bullet_lifetime_normalized + [proximity_to_left, proximity_to_right, proximity_to_top, proximity_to_bottom])
            
        # If there are fewer bullets than max_bullets, pad the remaining bullet info with zeros
        while len(bullet_info) < max_bullets * 12:  # Each bullet has 6 values (pos_x, pos_y, vel_x, vel_y, angle, lifetime)
            bullet_info.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        # Cooldown timer
        frames_since_last_shot = self.player.frame_counter - self.player.last_bullet_frame
        cooldown_remaining = max(0, self.player.bullet_cooldown_frames - frames_since_last_shot)
        cooldown_normalized = [cooldown_remaining / self.player.bullet_cooldown_frames]
        
        
        
        extended_state = np.array(state + stationary_time_normalized + bullet_info + cooldown_normalized)
        
        return extended_state

    def update_state_history(self):
        """
        Updates the state history buffer with the current state.
        """
        current_state = self.get_current_state()
        if len(self.state_history) < self.state_history.maxlen:
            # Pad with the initial state if history is not full
            while len(self.state_history) < self.state_history.maxlen:
                self.state_history.appendleft(current_state)
        else:
            self.state_history.append(current_state)

    
    def get_state(self):
        """
        Returns a flattened array of the current state and historical states.
        """
        self.update_state_history()
        # Flatten the state history into a single array
        state_array = np.concatenate(self.state_history, axis=0)  
        return state_array


    def get_reward(self):
        """
        Calculates the reward based on the current game state and recent actions.
    
        Returns:
            The calculated reward.
        """
        
        reward = 0

        if self.asteroid_hit:
            reward += 10
            self.asteroid_hit = False
            
        if self.small_asteroid_hit:
            reward += 15
            self.small_asteroid_hit = False
            
        if self.closest_asteroid_destroyed:
            reward += 15
            self.closest_asteroid_destroyed = False
        
        if np.linalg.norm(self.player.velocity) > 0:
            reward += 5  # Small reward for moving
        
        # Check for close encounters and apply a negative reward
        close_encounter_penalty = -50  # Penalty amount for getting too close to an asteroid
        min_safe_distance = 50  # Close encounter (in pixels)
        # safe = True
        
        for asteroid in self.asteroids:
            distance = math.sqrt((asteroid.position[0] - self.player.position[0]) ** 2 + 
                                 (asteroid.position[1] - self.player.position[1]) ** 2)

            if distance < min_safe_distance:
                reward += close_encounter_penalty
                # safe = False
                break  # Only apply the penalty once per step, regardless of the number of close encounters
        
        # if safe:
        #    reward += 1
            
        if self.stationary_time_counter >= 5:
            reward -= 100
        
        # Negative reward for game over
        if not self.running:
            reward -= 100
        return reward


    def run(self):
        """
        Main game loop responsible for event handling and updating the game state.
        """
        
        while self.running:
            self.handle_events()
            self.handle_action()
            self.update()
            self.draw()
            self.render()
            self.clock.tick(self.FPS)
        pygame.quit()


    def cap_player_speed(self):
        """
        Caps the player's speed to the maximum allowed, preventing it from exceeding set limits.
        """
        
        speed = math.sqrt(self.player.velocity[0] ** 2 + self.player.velocity[1] ** 2)
        if speed > self.player.max_speed:
            scale = self.player.max_speed / speed
            self.player.velocity[0] *= scale
            self.player.velocity[1] *= scale

    def handle_events(self):
        """
        Handles system events, such as closing the game window.
        """
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
    def handle_action(self, action):
        """
        Processes a player action, triggering the appropriate game logic.
    
        Parameters:
            action: The action code to be processed.
        """
        
        # Handles player actions and applies game logic accordingly.
        self.player.execute_action(action)
        

    def update(self):
        """
        Updates the game state, including the player, asteroids, and bullets.
        """
        
        # Update player
        self.player.update()
        
        if self.player.velocity == [0, 0]:
            self.stationary_time_counter += 1  # Increment counter if player is stationary
        else:
            self.stationary_time_counter = 0  # Reset counter if player moves
        
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
        """
        Updates the state of bullets, removing those that have exceeded their lifetime.
        """
        to_remove = []
        for bullet in self.bullets:
            bullet.update()
            if not bullet.is_alive():
                to_remove.append(bullet)
        for bullet in to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
    
    def update_asteroids(self):
        """
        Updates the state of asteroids, adding new ones as needed according to spawn intervals.
        """
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_asteroid_time > self.asteroid_interval and len(self.asteroids) < self.max_asteroids:
            self.add_asteroid()
            self.last_asteroid_time = current_time
        for asteroid in self.asteroids:
            asteroid.update()
    
    def check_collisions(self):
        """
        Checks for collisions between game entities and handles them appropriately.
        """
        
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
        """
        Splits an asteroid into two smaller asteroids upon collision with a bullet.
    
        Parameters:
            asteroid: The asteroid object to be split.
        """
        
        # Calculate the new size for the split asteroids
        new_size = asteroid.size // 2
        
        # Generate two new velocities randomly for the split asteroids
        for _ in range(2):
            new_velocity = [
                asteroid.velocity[0] + random.uniform(-1, 1),
                asteroid.velocity[1] + random.uniform(-1, 1)
            ]
            new_asteroid = Asteroid(self, asteroid.position[:], new_size, new_velocity)
            self.asteroids.append(new_asteroid)

    def draw(self):
        """
        Draws all game entities to the screen, including the player, asteroids, and bullets.
        """
    
        self.screen.fill((0, 0, 0))
        self.player.draw()
        for asteroid in self.asteroids:
            asteroid.draw()
        for bullet in self.bullets:
            bullet.draw()
            
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        # fps = str(int(self.clock.get_fps()))
        # fps_text = self.font.render(f'FPS: {fps}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))  # Position the score in the top-left corner
        # self.screen.blit(fps_text, (700, 10))  # Position the score in the top-right corner
        
        pygame.display.flip()
        
        

    def add_asteroid(self):
        """
        Adds a new asteroid to the game, with the possibility of targeting the player directly.
        """
        
        if len(self.asteroids) < self.max_asteroids:
            size = random.randint(15, 30)  
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
                velocity = [direction[0] / norm * 4.5, direction[1] / norm * 4.5]  # Adjust speed as needed
            else:
                # Use random velocity for non-targeting asteroids
                velocity = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5)]

            # Create and add the asteroid
            self.asteroids.append(Asteroid(self, position, size, velocity))

            # Toggle the targeting behavior for the next asteroid
            self.target_player_next = not self.target_player_next


