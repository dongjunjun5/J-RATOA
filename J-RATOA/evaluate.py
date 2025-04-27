import argparse
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

from MADDPG import MADDPG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('env', type=str, help='name of the environment',
    #                     choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
    #                              'simple_speaker_listener', 'simple_spread', 'simple_tag',
    #                              'simple_world_comm'])
    parser.add_argument('--env', type=str, default='MEC', help='name of the environment')
    parser.add_argument('--folder', type=str, default='323', help='name of the folder where model is saved')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=5, help='total number of episode')
    args = parser.parse_args()

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

    maddpg = MADDPG(obs_dim_list, act_dim_list, 0, 0, 0)
    model_dir = os.path.join('results', args.env, args.folder)
    assert os.path.exists(model_dir)
    # data = torch.load(os.path.join(model_dir, 'model.pt'))
    data = torch.load(os.path.join(model_dir, 'model_1500.pt'))
    for agent, actor_parameter in zip(maddpg.agents, data):
        agent.actor.load_state_dict(actor_parameter)
    print(f'MADDPG load model.pt from {model_dir}')

    total_reward = np.zeros((args.episode_num, env.n))  # reward of each episode
    final_total_reward = np.zeros(args.episode_num)  # reward of each agent in each episode
    for episode in range(args.episode_num):
        obs = env.reset()
        # record reward of each agent in this episode
        episode_reward = np.zeros((args.episode_length, env.n))
        for step in range(args.episode_length):  # interact with the env for an episode
            actions = maddpg.select_action(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            episode_reward[step] = rewards
            # env.render()
            time.sleep(0.02)
            obs = next_obs

        # episode finishes
        # calculate cumulative reward of each agent in this episode
        cumulative_reward = episode_reward.sum(axis=0)
        total_reward[episode] = cumulative_reward
        final_total_reward[episode] = sum(cumulative_reward)
        print(f'episode {episode + 1}: cumulative reward: {cumulative_reward}, '
              f'sum reward: {sum(cumulative_reward)}')

    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, total_reward[:, agent], label=agent)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'evaluating result of maddpg solve {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

    # all episodes performed, evaluating finishes
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    ax.plot(x, final_total_reward)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'evaluating result of maddpg solve 2 {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))