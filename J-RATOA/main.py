import argparse
import datetime
import os
from time import time

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

import pandas as pd
import csv

from MADDPG import MADDPG

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MEC_uniform', help='name of the environment')
    parser.add_argument('--episode-length', type=int, default=10, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=20000, help='total number of episode')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer-capacity', default=int(1e7))
    parser.add_argument('--batch-size', default=100)
    parser.add_argument('--update-delay-step-interval', default=1)
    parser.add_argument('--actor-lr', type=float, default=5e-6, help='learning rate of actor')
    parser.add_argument('--critic-lr', type=float, default=5e-6, help='learning rate of critic')
    parser.add_argument('--steps-for-random', default=0)
    parser.add_argument('--steps-before-learn', type=int, default=20000,
                        help='steps to be executed before agents start to learn')
    parser.add_argument('--learn-interval', type=int, default=1,
                        help='maddpg will only learn every this many steps')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='save model once every time this many episodes are completed')
    parser.add_argument('--tau', type=float, default=0.01, help='soft update parameter')
    args = parser.parse_args()
    start = time()

    # create folder to save result
    env_dir = os.path.join('results', args.env)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    res_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(res_dir)
    model_dir = os.path.join(res_dir, 'model')
    os.makedirs(model_dir)

    # create env
    scenario = scenarios.load(f'{args.env}.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)

    # get dimension info about observation and action
    obs_dim_list = []
    for obs_space in env.observation_space:  # continuous observation
        obs_dim_list.append(obs_space.shape[0])  # Box
    act_dim_list = []
    for act_space in env.action_space:  # continuous action
        act_dim_list.append(act_space.shape[0])  # Box

    maddpg = MADDPG(obs_dim_list, act_dim_list, args.buffer_capacity, args.actor_lr, args.critic_lr, res_dir)

    with open(os.path.join(res_dir, "parameter.txt"), "w") as file:
        file.write('lr: ' + str(args.actor_lr) + '\nkappa: ' + str(env.world.energy_coefficient) + '\nradius: ' + str(env.world.radius) + '\nMECradius: ' + str(env.world.MEC_radius) + '\nN: ' + str(env.n) + '\nepisode length: ' + str(args.episode_length) + '\nupdate delay interval: ' + str(args.update_delay_step_interval) + '\nlearn interval: ' + str(args.learn_interval))

    total_step = 0
    total_reward = np.zeros((args.episode_num, env.n))  # reward of each agent in each episode
    training_loss = np.zeros((args.episode_num, env.n))
    actoring_loss = np.zeros((args.episode_num, env.n))
    final_total_reward = np.zeros(args.episode_num)
    episode_step_reward = np.zeros((env.n, args.episode_num, args.episode_length))
    episode_step_latency = np.zeros((env.n, args.episode_num, args.episode_length))

    for episode in range(args.episode_num):
        obs = env.reset()
        # record reward of each agent in this episode
        episode_reward = np.zeros((args.episode_length, env.n))
        episode_latency = np.zeros((args.episode_length, env.n))
        for step in range(args.episode_length):  # interact with the env for an episode
            # 랜덤 액션 사용
            if total_step < args.steps_for_random:
                actions = maddpg.random_action()
            else:
                actions = maddpg.select_action(obs, total_step)

            # # 일반 액션 사용
            # actions = maddpg.select_action(obs, total_step)

            next_obs, rewards, latencys, dones, infos = env.step(actions)
            episode_reward[step] = rewards
            episode_latency[step] = latencys
            total_step += 1
            # print('total_step', total_step)
            maddpg.add(obs, actions, rewards, next_obs, dones)
            # only start to learn when there are enough experiences to sample
            if total_step >= args.steps_before_learn:
                if total_step % args.learn_interval == 0:
                    train_loss, actor_loss = maddpg.learn(args.batch_size, args.gamma, total_step, args.update_delay_step_interval)
                    train_loss = [tensor_item.tolist() for tensor_item in train_loss]
                    actor_loss = [tensor_item.tolist() for tensor_item in actor_loss]
                    training_loss[episode] = train_loss
                    actoring_loss[episode] = actor_loss
                    if total_step % args.update_delay_step_interval == 0:
                        maddpg.update_target(args.tau)
                if episode % args.save_interval == 0:
                    torch.save([agent.actor.state_dict() for agent in maddpg.agents],
                               os.path.join(model_dir, f'model_{episode}.pt'))
            obs = next_obs
            # if sum(dones) != 0:
            #     break
        # save episode and step reward
        for a in range(env.n):
            episode_step_reward[a][episode] = episode_reward[:, a].reshape((1, -1))
            episode_step_latency[a][episode] = episode_latency[:, a].reshape((1, -1))

        # episode finishes
        # calculate cumulative reward of each agent in this episode
        cumulative_reward = episode_reward.sum(axis=0)
        total_reward[episode] = cumulative_reward
        final_total_reward[episode] = sum(cumulative_reward)
        print(f'episode {episode + 1}: cumulative reward: {cumulative_reward}, '
              f'sum reward: {sum(cumulative_reward)}')

    for a in range(env.n):
        np.save(os.path.join(res_dir, f'episode_step_rewards_agent{a}.npy'), episode_step_reward[a])
        np.save(os.path.join(res_dir, f'episode_step_latencys_agent{a}.npy'), episode_step_latency[a])

    # all episodes performed, training finishes
    # save agent parameters
    torch.save([agent.actor.state_dict() for agent in maddpg.agents], os.path.join(res_dir, 'model.pt'))
    # save training reward
    np.save(os.path.join(res_dir, 'rewards.npy'), total_reward)


    def get_running_reward(reward_array: np.ndarray, window=10):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(reward_array)
        for i in range(window - 1):
            running_reward[i] = np.mean(reward_array[:i + 1])
        for i in range(window - 1, len(reward_array)):
            running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
        return running_reward


    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, total_reward[:, agent], label=agent)
        # ax.plot(x, get_running_reward(total_reward[:, agent]))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve 1 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    print(f'training finishes, time spent: {datetime.timedelta(seconds=int(time() - start))}')

    # all episodes performed, training finishes
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    ax.plot(x, final_total_reward)
    # ax.plot(x, get_running_reward(final_total_reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve 2 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, training_loss[:, agent], label=agent)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('training_loss')
    title = f'training result of maddpg solve 3 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    ax.plot(x, np.sum(training_loss, axis=1, keepdims=True))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('training_loss')
    title = f'training result of maddpg solve 4 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, actoring_loss[:, agent], label=agent)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('actor_loss')
    title = f'training result of maddpg solve 5 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    ax.plot(x, np.sum(actoring_loss, axis=1, keepdims=True))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('actor_loss')
    title = f'training result of maddpg solve 6 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))


    def find_max_row(csv_file):
        df = pd.read_csv(csv_file)
        row_sums = df.sum(axis=1)
        max_row_index = row_sums.idxmax()
        return max_row_index


    def get_reward_value(csv_file, row_num, col_num):
        df = pd.read_csv(csv_file)
        value = df.iloc[row_num, col_num] * (10 ** (-5))  # reward normalization
        return value


    def get_latency_value(csv_file, row_num, col_num):
        df = pd.read_csv(csv_file)
        value = df.iloc[row_num, col_num]
        return value


    arr = np.load(os.path.join(res_dir, 'rewards.npy'))
    df = pd.DataFrame(arr)
    df.to_csv(os.path.join(res_dir, 'rewards.csv'), index=False)

    agent_numbers = env.n

    for a in range(agent_numbers):
        arr = np.load(os.path.join(res_dir, f'episode_step_rewards_agent{a}.npy'))
        df = pd.DataFrame(arr)
        df.to_csv(os.path.join(res_dir, f'episode_step_rewards_agent{a}.csv'), index=False)

    for a in range(agent_numbers):
        arr = np.load(os.path.join(res_dir, f'episode_step_latencys_agent{a}.npy'))
        df = pd.DataFrame(arr)
        df.to_csv(os.path.join(res_dir, f'episode_step_latencys_agent{a}.csv'), index=False)

    energy_consumption_for_each_agent = []
    latency_for_each_agent = []

    reward_file = os.path.join(res_dir, 'rewards.csv')
    max_row_index = find_max_row(reward_file)
    print("가장 큰 합을 갖는 행의 인덱스:", max_row_index + 2)
    print("episode:", max_row_index + 1)

    target_step = args.episode_length
    row_num = max_row_index
    col_num = target_step - 1

    save_term = 10 #####
    interval = args.episode_num // save_term ####

    energy_consumption_for_each_agent_list = []
    for agent_num in range(agent_numbers):
        agent_reward_file = os.path.join(res_dir, f'episode_step_rewards_agent{agent_num}.csv')
        value = get_reward_value(agent_reward_file, row_num, col_num)
        energy_consumption_for_each_agent_list.append(value)

    energy_consumption_for_each_agent.append(energy_consumption_for_each_agent_list)

    for i in range(args.episode_num):
        if i % interval == 0:
            energy_consumption_for_each_agent_list = []
            for agent_num in range(agent_numbers):
                agent_reward_file = os.path.join(res_dir, f'episode_step_rewards_agent{agent_num}.csv')
                value_save = get_reward_value(agent_reward_file, i, col_num)
                energy_consumption_for_each_agent_list.append(value_save)
            energy_consumption_for_each_agent.append(energy_consumption_for_each_agent_list)

    print('energy_consumption_for_each_agent', energy_consumption_for_each_agent)

    latency_for_each_agent_list = []
    for agent_num in range(agent_numbers):
        agent_latency_file = os.path.join(res_dir, f'episode_step_latencys_agent{agent_num}.csv')
        value = get_latency_value(agent_latency_file, row_num, col_num)
        latency_for_each_agent_list.append(value)

    latency_for_each_agent.append(latency_for_each_agent_list)

    for i in range(args.episode_num):
        if i % interval == 0:
            latency_for_each_agent_list = []
            for agent_num in range(agent_numbers):
                agent_latency_file = os.path.join(res_dir, f'episode_step_latencys_agent{agent_num}.csv')
                value_save = get_latency_value(agent_latency_file, i, col_num)
                latency_for_each_agent_list.append(value_save)
            latency_for_each_agent.append(latency_for_each_agent_list)

    print('latency_for_each_agent', latency_for_each_agent)


    def list_to_csv(data_list, csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_list)

    # 예시 데이터 리스트
    data_list1 = energy_consumption_for_each_agent

    # CSV 파일로 저장
    csv_file1 = os.path.join(res_dir, 'energy_consumption_for_each_agent.csv')
    list_to_csv(data_list1, csv_file1)

    # 예시 데이터 리스트
    data_list2 = latency_for_each_agent

    # CSV 파일로 저장
    csv_file2 = os.path.join(res_dir, 'latency_for_each_agent.csv')
    list_to_csv(data_list2, csv_file2)

    # 예시 데이터 리스트
    data_list3 = training_loss
    data_list4 = actoring_loss

    # CSV 파일로 저장
    csv_file3 = os.path.join(res_dir, 'critic_loss.csv')
    csv_file4 = os.path.join(res_dir, 'actor_loss.csv')
    list_to_csv(data_list3, csv_file3)
    list_to_csv(data_list4, csv_file4)

    with open(os.path.join(res_dir, "max_episode.txt"), "w") as file:
        file.write('episode: ' + str(max_row_index + 1))