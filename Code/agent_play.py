from agents import DQNAgent
from game import Game
import os
from utils import set_seed
from state_size import state_size

set_seed(1575)

base_dir = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning"
folder_name = "DQN_Models"
folder_path = os.path.join(base_dir, folder_name)

# Create the directory if it does not exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


model = f'{folder_path}/model_best_reward.h5'
# model = f'{folder_path}/model_episode_5000.h5'


env = Game()



action_size = 8  # Based on the game's action space
agent = DQNAgent(state_size = state_size, action_size = action_size, model_file_path = model, evaluation=True)


env.run_with_dqn_agent(agent, num_episodes=10, screenshot=True)