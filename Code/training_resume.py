from game import Game
from agents import DQNAgent
from utils import set_seed
from state_size import state_size
import numpy as np
import os

set_seed(2468) # Seed 15 used through first 700. Changed to 16 (700 - 1200), changed to 125 (1200 - 1700), change to 4 (1700-2000), 12345 (2000+), 123 (3200+), 2468 (3900+) 

# learning rate 0.00001 for first 1500 episodes, 1500-1800 lr 0.000001, 1800 + lr 0.0000001, 2400 lr 0.00001, 3500 lr 0.000001
# Batch size 64 for 1-2400, 256 2400 - 3100, 3500+ 128, bvack to 256 3700+
# added saving functionality for target network at 2700. Primed it with the model_path
# Saved reward 15432 for best model in base folder
batch_size = 256
env = Game()  # Initialize game environment

action_size = 8 

STARTING_EPISODE = 4901
ENDING_EPISODE = 5000

SAVE_INTERVAL = 50  
RENDER_EVERY_N_EPISODES = 25  
render = False


base_dir = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning"  
folder_name = "DQN_Models"
folder_path = os.path.join(base_dir, folder_name)

# Create the directory if it does not exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


model_path = f'{folder_path}/model_episode_{STARTING_EPISODE - 1}.h5'
target_model_path = f'{folder_path}/target_model_episode_{STARTING_EPISODE - 1}.h5'
training_state_path = f'{folder_path}/model_episode_{STARTING_EPISODE - 1}_training_state.json'
replay_buffer_path = f'{folder_path}/model_episode_{STARTING_EPISODE - 1}_replay_buffer.pkl'

agent = DQNAgent(state_size,
                 action_size,
                 model_file_path = model_path,
                 target_model_file_path = target_model_path, 
                 training_state_file_path = training_state_path,
                 replay_buffer_file_path = replay_buffer_path)

best_reward = agent.load_best_model_reward(f'{folder_path}/model_best_reward.json')

for episode in range(STARTING_EPISODE, ENDING_EPISODE + 1):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    
    for time in range(1, 5001):
        if render:
            if episode % RENDER_EVERY_N_EPISODES == 0 or episode == 1:
                env.render()  # Render the environment
        # Your existing logic for taking actions and updating the environment
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        if time == 5000:  
            done = True  # Manually set done to True to indicate end of episode
            reward += 100.1  # survival_bonus
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {episode}/{ENDING_EPISODE}, score: {env.score} Total Reward: {round(total_reward, 2)}, e: {agent.epsilon:.2}")
            break
    
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Save the model every SAVE_INTERVAL episodes or first episode
    if episode % SAVE_INTERVAL == 0 or episode == 1:
        agent.save_model(f'{folder_path}/model_episode_{episode}.h5')
        agent.save_target_model(f'{folder_path}/target_model_episode_{episode}.h5')
        agent.save_training_state(f'{folder_path}/model_episode_{episode}_training_state.json', agent.epsilon)
        agent.save_replay_buffer(f'{folder_path}/model_episode_{episode}_replay_buffer.pkl')
    if total_reward > best_reward:
        best_reward = total_reward
        # Save the current model as the new best model
        agent.save_model(f'{folder_path}/model_best_reward.h5')
        agent.save_best_model_reward(f'{folder_path}/model_best_reward.json', best_reward, episode)
        print(f"New best model at {episode} saved with reward: {best_reward}")

agent.save_model(f'{folder_path}/model_episode_{ENDING_EPISODE}.h5')
agent.save_target_model(f'{folder_path}/target_model_episode_{episode}.h5')
agent.save_training_state(f'{folder_path}/model_episode_{ENDING_EPISODE}_training_state.json', agent.epsilon)
agent.save_replay_buffer(f'{folder_path}/model_episode_{ENDING_EPISODE}_replay_buffer.pkl')