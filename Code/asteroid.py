import pygame
import math
import random

class Asteroid:
    
    """
    Represents an asteroid in the game with a specific position, size, and velocity.

    The asteroid can move across the screen, wrapping around edges to reappear on the opposite side. It can collide with other objects, like the player's spaceship or bullets.

    Attributes:
        game (Game): The game object that holds global configurations and controls the game environment.
        position (list of float): The x and y coordinates of the asteroid on the screen.
        size (int): The diameter of the asteroid, affecting its speed and interactions.
        velocity (list of float): The x and y components of the asteroid's velocity, dictating its movement direction and speed.
        
    Methods:
        update(): Moves the asteroid based on its velocity and applies screen wrapping.
        wrap_position(): Wraps the asteroid's position to the opposite side of the screen if it goes off the edges.
        draw(): Renders the asteroid on the screen as a circle.
        collides_with(other): Checks if the asteroid collides with another object, such as the player or a bullet.
    """
    
    def __init__(self, game, position, size, velocity = None):
        """
        Initializes a new asteroid instance.

        Args:
            game: Reference to the game object for accessing global configurations and methods.
            position (list): The starting x and y coordinates of the asteroid.
            size (int): The diameter of the asteroid. Impacts the asteroid's speed and determines if the asteroid will split.
            velocity (list, optional): The initial velocity of the asteroid. If not provided, velocity is randomized based on the asteroid's size.

        Smaller asteroids are set to move faster to increase difficulty and variability in the game.
        """
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
        """
        Updates asteroid position
        """
           
        # Update asteroid position based on velocity
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        
        
        # Screen wrapping
        self.wrap_position()
        
        
    def wrap_position(self):
        """
        Wrap the position to appear on the opposite side of the screen when it goes off the edge.
        """
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
        """Draw the asteroid as a circle"""
        pygame.draw.circle(self.game.screen, (255, 255, 255), self.position, self.size)

    def collides_with(self, other):
        """
        Collision detection: check if distance between centers is less than sum of radii
        
        Args:
            other: Another game object with a defined position and size attribute.

        Returns:
            bool: True if the asteroid collides with the other object, False otherwise.
        """
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance < self.size + 5