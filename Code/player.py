import pygame
import math
from bullet import Bullet

class Player:
    """
    Represents the player's spaceship in the game.

    Attributes:
        game (Game): The game instance this player belongs to.
        position (list of float): The x and y coordinates of the player on the screen.
        size (int): The size of the player's ship.
        angle (float): The current angle of the player's ship.
        velocity (list of float): The velocity of the player along the x and y axes.
        acceleration (float): The acceleration rate of the player's ship.
        rotation_speed (float): The rotation speed of the player's ship.
        max_speed (float): The maximum speed of the player's ship.
        friction (float): The friction coefficient affecting the player's movement.
        last_bullet_time (int): The timestamp of the last bullet shot.
        bullet_cooldown_ms (int): The cooldown time between shots in milliseconds.
        moving_forward (bool): A flag indicating if the player is moving forward.
        rotating_left (bool): A flag indicating if the player is rotating left.
        rotating_right (bool): A flag indicating if the player is rotating right.
        shooting (bool): A flag indicating if the player is shooting.
        
    Methods:
        shoot(): Fires a bullet from the player's current position and orientation.
        apply_friction(): Applies friction to the player's velocity, slowing them down.
        wrap_position(): Ensures the player wraps around the screen when moving out of bounds.
        move_forward(): Accelerates the player forward in the direction they are facing.
        rotate_left(): Rotates the player's ship to the left.
        rotate_right(): Rotates the player's ship to the right.
        cap_speed(): Caps the player's speed to the maximum allowed speed.
        execute_action(action): Executes the given action (e.g., move, shoot).
        update(): Updates the player's state, including position and actions.
        draw(): Draws the player's spaceship on the game screen.
    """
    
    def __init__(self, game):
        """
        Initializes a new player instance.

        Args:
            game (Game): The game instance this player belongs to.
        """
        
        self.game = game
        self.position = [self.game.screen_width // 2, self.game.screen_height // 2]
        self.size = 5
        self.angle = 0
        self.velocity = [0, 0]
        self.acceleration = 0.1
        self.rotation_speed = 10
        self.max_speed = 5
        self.friction = 0.99
        self.last_bullet_time = 0  
        self.bullet_cooldown_ms = 250
        self.moving_forward = False
        self.rotating_left = False
        self.rotating_right = False
        self.shooting = False


    def shoot(self):
        """
        Shoots a bullet from the player's current position and direction.
        """
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
        """
        Applies friction to the player's velocity, slowing them down.
        """
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction


    def wrap_position(self):
        """
        Ensures the player wraps around the screen when moving out of bounds.
        """
        if self.position[0] < 0:
            self.position[0] = self.game.screen_width
        elif self.position[0] > self.game.screen_width:
            self.position[0] = 0

        if self.position[1] < 0:
            self.position[1] = self.game.screen_height
        elif self.position[1] > self.game.screen_height:
            self.position[1] = 0
    
    def move_forward(self):
        """
        Moves the player forward in the direction they are facing.
        """
        
        self.velocity[0] += self.acceleration * math.cos(math.radians(self.angle))
        self.velocity[1] += self.acceleration * math.sin(math.radians(self.angle))
        
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale
    
    
    def rotate_left(self):
        """
        Rotates the player to the left.
        """
        self.angle -= self.rotation_speed
        
    def rotate_right(self):
        """
        Rotates the player to the right.
        """
        self.angle += self.rotation_speed
    
        
    def cap_speed(self):
        """
        Caps the player's speed to the maximum speed.
        """
        # Calculate the current speed
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        
        # If the current speed exceeds the max speed, scale down the velocity
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale

        
        
    def execute_action(self, action):
        """
        Executes an action based on the given action code.

        Args:
            action (int): The action code to execute.
        """
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

    
    def update(self):
        """
        Updates the player's state.
        """
        # Apply current velocity to update position
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
    
        # Apply friction if not moving forward
        if not self.moving_forward:
            self.apply_friction()
    
        # Cap the speed to the maximum speed
        self.cap_speed()
    
        # Screen wrapping
        self.wrap_position()
  

    def draw(self):
        """
        Draws the player's spaceship on the game screen.
        """
        radians = math.radians(self.angle)
        tip = (self.position[0] + 15 * math.cos(radians), self.position[1] + 15 * math.sin(radians))
        left_base = (self.position[0] + 15 * math.cos(radians + math.pi * 5/6), self.position[1] + 15 * math.sin(radians + math.pi * 5/6))
        right_base = (self.position[0] + 15 * math.cos(radians - math.pi * 5/6), self.position[1] + 15 * math.sin(radians - math.pi * 5/6))
        pygame.draw.polygon(self.game.screen, (255, 255, 255), [tip, left_base, right_base])
        