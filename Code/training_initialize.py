from game import Game
from agents import DQNAgent
from utils import set_seed
from state_size import state_size
import numpy as np
import os

set_seed(15)


batch_size = 64
best_reward = -10000

env = Game()  # Initialize game environment

action_size = 8  # Based on the game's action space
agent = DQNAgent(state_size, action_size)

STARTING_EPISODE = 1
ENDING_EPISODE = 100

SAVE_INTERVAL = 50
RENDER_EVERY_N_EPISODES = 25
render = True

base_dir = "C:/Users/ericl/OneDrive/Documents/ReinforcementLearning"
folder_name = "DQN_Models"
folder_path = os.path.join(base_dir, folder_name)

# Create the directory if it does not exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


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
agent.save_training_state(f'{folder_path}/model_episode_{ENDING_EPISODE}_training_state.json', agent.epsilon)
agent.save_replay_buffer(f'{folder_path}/model_episode_{ENDING_EPISODE}_replay_buffer.pkl')