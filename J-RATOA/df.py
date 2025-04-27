import pandas as pd
import numpy as np
import os

def find_max_row(csv_file):
    df = pd.read_csv(csv_file)
    row_sums = df.sum(axis=1)
    print(row_sums)
    max_row_index = row_sums.idxmax()
    return max_row_index


def find_mean_row(csv_file):
    df = pd.read_csv(csv_file)
    row_sums = df.sum(axis=1)
    print(row_sums)
    print(df.values.mean())
    diff_from_mean = np.abs(row_sums - df.values.mean())
    print(diff_from_mean)
    mean_row_index = diff_from_mean.idxmin()
    return mean_row_index

env_dir = os.path.join('results', 'MEC_uniform')
res_dir = os.path.join(env_dir, '30')
reward_file = os.path.join(res_dir, 'rewards.csv')
mean_row_index = find_mean_row(reward_file)
print("중간값을 갖는 행의 인덱스:", mean_row_index + 2)