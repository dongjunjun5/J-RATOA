import pandas as pd
import numpy as np
import os

# numpy array 생성
res_dir = os.path.join('C:/Users/dongj/RL/MADDPG/results/MEC/617')
arr = np.load(os.path.join(res_dir, 'rewards.npy'))
# arr = np.load('C:/Users/dongj/RL/MADDPG/results/MEC/401/rewards.npy')
df = pd.DataFrame(arr)
# df.to_csv('C:/Users/dongj/RL/MADDPG/results/MEC/401/rewards.csv', index=False)
df.to_csv(os.path.join(res_dir, 'rewards.csv'), index=False)

for a in range(20):
    arr = np.load(os.path.join(res_dir, f'episode_step_rewards_agent{a}.npy'))
    df = pd.DataFrame(arr)
    df.to_csv(os.path.join(res_dir, f'episode_step_rewards_agent{a}.csv'), index=False)
