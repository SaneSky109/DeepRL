state_size = 2210

# =============================================================================
# Player Information:
# 
# Normalized position (x, y): 2 elements
# Normalized angle: 1 element
# Normalized velocity (vx, vy): 2 elements
# Proximity to edges (left, right, top, bottom): 4 elements
# 
# Asteroid Information:
# 
# For each of the nearest 10 asteroids, you have:
# Normalized position (x, y): 2 elements
# Normalized size: 1 element
# Normalized velocity (vx, vy): 2 elements
# Proximity to edges (left, right, top, bottom): 4 elements
# Each asteroid contributes 9 elements, and since there are up to 10 asteroids: 10 * 9 = 90 elements
# Stationary Counter:
# 
# Normalized stationary time: 1 element
# Bullet Information:
# 
# For each of the 2 bullets, you include:
# Initial normalized position (x, y): 2 elements
# Normalized position (x, y): 2 elements
# Normalized velocity (vx, vy): 2 elements
# Normalized angle: 1 element
# Normalized lifetime: 1 element
# Proximity to edges (left, right, top, bottom): 4 elements
# Each bullet contributes 12 elements, and since there are up to 10 bullets: 10 * 12 = 120 elements
# Cooldown Timer:
# 
# Normalized cooldown remaining: 1 element
# 
# 
# 
# Player: 9 elements
# Asteroids: 90 elements
# Stationary Counter: 1 element
# Bullets: 120 elements
# Cooldown Timer: 1 element
#
# Total = 221 elements
# Multiply by temporal_buffer: 10
# Final = 2210
#
# =============================================================================
