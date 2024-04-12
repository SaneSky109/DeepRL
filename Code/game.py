import pygame
import math
import random
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
    
    def __init__(self):
        
        
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.font = pygame.font.SysFont('Arial', 24)  # Create a Font object
        self.screen_width, self.screen_height = 800, 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.FPS = 60
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
                normalized_size = ast.size / 30 # max size of asteroid
                normalized_velocity = [ast.velocity[0] / 5, ast.velocity[1] / 5]  # 5 is the maximum speed of asteroid for normalization
                info.extend(normalized_position + [normalized_size] + normalized_velocity)
        # If there are fewer than 10 asteroids, fill the remaining space with zeros
        while len(info) < 10 * 5:  # 10 asteroids * (pos_x, pos_y, size, vel_x, vel_y)
            info.extend([0, 0, 0, 0, 0])
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

    def get_state(self):
        """
        Returns the current state of the game as perceived by an agent.
    
        Returns:
            An array representing the current game state.
        """
        
        player_pos_normalized = self.normalize_position(self.player.position)
        player_angle_normalized = [self.player.angle / 360]
        player_velocity_normalized = [self.player.velocity[0] / self.player.max_speed, self.player.velocity[1] / self.player.max_speed]

        state = player_pos_normalized + player_angle_normalized + player_velocity_normalized + self.get_nearest_asteroid_info()
        return np.array(state)

    
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
            reward += 25
            self.small_asteroid_hit = False
            
        if self.closest_asteroid_destroyed:
            reward += 20
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
        close_encounter_penalty = -10  # Penalty amount for getting too close to an asteroid
        min_safe_distance = 75  # Close encounter (in pixels)
    
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
        """
        Main game loop responsible for event handling and updating the game state.
        """
        
        while self.running:
            
            self.handle_action()
            self.update()
            self.draw()
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
                asteroid.velocity[0] + random.uniform(-2, 2),
                asteroid.velocity[1] + random.uniform(-2, 2)
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
        self.screen.blit(score_text, (10, 10))  # Position the score in the top-left corner
        
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
                velocity = [direction[0] / norm * 2, direction[1] / norm * 2]  # Adjust speed as needed
            else:
                # Use random velocity for non-targeting asteroids
                velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]

            # Create and add the asteroid
            self.asteroids.append(Asteroid(self, position, size, velocity))

            # Toggle the targeting behavior for the next asteroid
            self.target_player_next = not self.target_player_next


