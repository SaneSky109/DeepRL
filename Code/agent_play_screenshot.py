from agents import DQNAgent
from game import Game
import os


base_dir = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning"
folder_name = "Screenshots"
folder_path = os.path.join(base_dir, folder_name)

# Create the directory if it does not exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


model = 'C:/Users/ericl/OneDrive/Documents/ReinforcementLearning/DQN_Models/model_episode_1000.h5'

env = Game()


state_size = 2 + 1 + 2 + (10 * 5)
action_size = 8  # Based on the game's action space
agent = DQNAgent(state_size = state_size, action_size = action_size, model_file_path = model)


env.run_with_dqn_agent(agent, screenshot = True)