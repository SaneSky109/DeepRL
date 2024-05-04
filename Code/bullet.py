import pygame
import math

class Bullet:
    """
    Represents a bullet fired by the player's spaceship in the game.

    Attributes:
        game (Game): Reference to the game instance this bullet belongs to.
        position (list of float): The starting x and y coordinates of the bullet.
        angle (float): The direction in which the bullet is fired, in degrees.
        speed (float): The speed at which the bullet moves across the screen.
        velocity (list of float): The velocity of the bullet along the x and y axes, calculated based on its speed and angle.
        lifetime (int): The number of frames the bullet will remain active before being automatically removed.

    Methods:
        update(): Updates the bullet's position based on its velocity and decreases its lifetime.
        draw(): Renders the bullet on the screen as a small line or point.
        is_alive(): Returns True if the bullet's lifetime is greater than 0, indicating it is still active.
        wrap_position(): Adjusts the bullet's position to wrap around the screen edges, appearing on the opposite side if moved off-screen.
        collides_with(asteroid): Checks for collision with an asteroid using distance-based collision detection.
    """
    
    def __init__(self, game, position, angle):
        """
        Initializes a new bullet instance with a given position and angle.

        Args:
            game (Game): The game instance this bullet belongs to.
            position (list of float): The starting x and y coordinates of the bullet.
            angle (float): The direction in which the bullet is fired, in degrees.
        """
        self.game = game
        self.initial_position = list(position)
        self.position = list(position)  # Make a copy of the position list to avoid modifying the original
        self.angle = angle
        self.speed = 7.5  # Speed of the bullet
        self.velocity = [
            math.cos(math.radians(self.angle)) * self.speed,
            math.sin(math.radians(self.angle)) * self.speed
        ]
        self.lifetime = 75  # Frames until the bullet is automatically removed

    def update(self):
        """
        Updates the bullet's position based on its velocity and decreases its lifetime.
        """
        # Move the bullet based on its velocity
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        # Decrease the lifetime of the bullet
        self.lifetime -= 1
        self.wrap_position()

    def draw(self):
        """
        Renders the bullet on the screen as a small line.
        """
        # Draw the bullet as a small circle or a point for simplicity
        start_pos = self.position
        end_pos = (start_pos[0] + self.speed * math.cos(math.radians(self.angle)),
                   start_pos[1] + self.speed * math.sin(math.radians(self.angle)))
        pygame.draw.line(self.game.screen, (255, 255, 255), start_pos, end_pos, width = 2)
        # pygame.draw.circle(self.game.screen, (255, 255, 255), (int(self.position[0]), int(self.position[1])), 3)

    def is_alive(self):
        """
       Determines whether the bullet is still active based on its remaining lifetime.

       Returns:
           bool: True if the bullet's lifetime is greater than 0, otherwise False.
       """
        return self.lifetime > 0

    
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

    def collides_with(self, asteroid):
        """
        Checks for collision with an asteroid using simple distance-based collision detection.

        Args:
            asteroid (Asteroid): The asteroid to check for collision with.

        Returns:
            bool: True if the bullet collides with the asteroid, False otherwise.
        """
        # Check for collision with an asteroid using simple distance-based collision detection
        dx = self.position[0] - asteroid.position[0]
        dy = self.position[1] - asteroid.position[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance < asteroid.size