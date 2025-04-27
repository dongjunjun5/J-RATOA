import gym
import math
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        self.landmarks = self.world.landmarks
        # set required vectorized gym env property
        self.n = len(world.policy_agents)

        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.association_count_list = []
        # # environment parameters
        # self.discrete_action_space = True
        # # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        # self.discrete_action_input = False
        # # if true, even the action is continuous, action will be performed discretely
        # self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # # if true, every agent has the same reward
        # self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False

        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # association action space
            # association_action_space = spaces.Discrete(len(world.landmarks))
            association_action_space = spaces.Box(low=0, high=len(world.landmarks), shape=(1,), dtype=np.float32)
            total_action_space.append(association_action_space)
            # CPU allocation action space
            cpu_allocation_action_space = spaces.Box(low=world.min_CPU, high=world.max_CPU, shape=(1,), dtype=np.float32)
            total_action_space.append(cpu_allocation_action_space)
            # offloading ratio action space
            offloading_ratio_action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) # alpha, beta
            total_action_space.append(offloading_ratio_action_space)
            # transmission power allocation action space for LTE
            trans_power_allocation_lte_action_space = spaces.Box(low=world.min_p_trans_LTE, high=world.max_p_trans_LTE, shape=(1,), dtype=np.float32)
            total_action_space.append(trans_power_allocation_lte_action_space)
            # transmission power allocation action space for 5G
            trans_power_allocation_5g_action_space = spaces.Box(low=world.min_p_trans_5G, high=world.max_p_trans_5G, shape=(1,), dtype=np.float32)
            total_action_space.append(trans_power_allocation_5g_action_space)
            # RB allocation action space for LTE
            # rb_allocation_lte_action_space = spaces.Discrete(world.max_RB_number_LTE)
            rb_allocation_lte_action_space = spaces.Box(low=0, high=world.max_RB_number_LTE, shape=(1,), dtype=np.float32)
            total_action_space.append(rb_allocation_lte_action_space)
            # RB allocation action space for 5G
            # rb_allocation_5g_action_space = spaces.Discrete(world.max_RB_number_5G)
            rb_allocation_5g_action_space = spaces.Box(low=0, high=world.max_RB_number_5G, shape=(1,), dtype=np.float32)
            total_action_space.append(rb_allocation_5g_action_space)

            # total action space
            if len(total_action_space) > 1:
                act_dim = 0
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                for i in range(len(total_action_space)):
                    act_dim += total_action_space[i].shape[0]
                self.action_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(act_dim,), dtype=np.float32))
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        # # rendering
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        latency_n = []
        done_n = []
        info_n = {'n': []}
        association_list = []
        arrival_rate_mec_list = []
        rb_number_lte_list = []
        rb_number_5g_list = []
        self.agents = self.world.policy_agents
        self.landmarks = self.world.landmarks
        workload_mec = 0
        local_order = 0
        processing_time_local_for_sum_list = []
        # set action for each agent
        for i, agent in enumerate(self.agents):
            agent.local_order = 0
            agent.waiting_time = 0
            self._set_action(action_n[i], agent)
            association_list.append(agent.action.association) # 각 agent들의 연결 MEC server index를 나타내는 list
            arrival_rate_mec = (1-agent.action.offloading_ratio[0]) * agent.state.arrival_rate * agent.state.task_size * agent.state.required_CPU
            arrival_rate_mec_list.append(arrival_rate_mec)
            required_cpu_for_task_for_sum = agent.state.task_size * agent.state.required_CPU
            processing_time_local_for_sum = math.ceil(agent.action.offloading_ratio[0] * required_cpu_for_task_for_sum / agent.action.CPU_allocation / self.world.time_slot)
            if agent.action.offloading_ratio[0] != 0: # local 처리 순서 설정 (조금이라도 local 처리하면)
                local_order += 1
                agent.local_order = local_order
                processing_time_local_for_sum_list.append(processing_time_local_for_sum)
                if agent.local_order > 1:
                    agent.waiting_time = sum(processing_time_local_for_sum_list[0:local_order-1])
            # print('agent.waiting_time', agent.waiting_time)

        for i, agent in enumerate(self.agents):
            self._set_association_count_list(association_list, agent) # 각 MEC server들에 연결된 agent 개수를 나타내는 list

        for i, agent in enumerate(self.agents):
            self._set_action2(action_n[i], agent)
            rb_number_lte_list.append(agent.action.RB_allocation_LTE)
            rb_number_5g_list.append(agent.action.RB_allocation_5G)

        for j, landmark in enumerate(self.landmarks):
            rb_number_lte_mec = 0
            rb_number_5g_mec = 0
            for i, agent in enumerate(self.agents):
                if j == agent.action.association:
                    workload_mec += arrival_rate_mec_list[i]
                    rb_number_lte_mec += rb_number_lte_list[i]
                    rb_number_5g_mec += rb_number_5g_list[i]
            landmark.workload = workload_mec / self.world.CPU_MEC
            landmark.RB_number_LTE = rb_number_lte_mec
            landmark.RB_number_5G = rb_number_5g_mec
        #     print('landmark.RB_number_LTE', landmark.RB_number_LTE)
        # print('end', self.landmarks[0].RB_number_LTE)

        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            a = self._get_reward(agent)
            reward_n.append(a[0])
            latency_n.append(a[1])
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        return obs_n, reward_n, latency_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def _set_association_count_list(self, association_list, agent):
        association_count_list = []
        for i in range(len(self.world.landmarks)):
            association_count_list.append(association_list.count(i))
        agent.association_count_list = association_count_list

    # def _set_workload_mec(self, landmark):
    #     arrival_rate_mec = 0
    #     if agent.action.association == :
    #
    #
    #     for i, agent in enumerate(self.agents):
    #         agent.action.association
    #         arrival_rate_mec += (1-agent.action.offloading_ratio[0])*agent.state.arrival_rate*agent.state.task_size)
    #
    #     self.world.CPU_MEC
    #     landmark.workload = workload_mec

    # set env action for a particular agent
    def _set_action(self, action, agent):
        #####################이게 맞나!!!!!!!!!####################
        # print('action', action)
        agent.action.association = np.zeros(1) # float
        agent.action.CPU_allocation = np.zeros(1) # float
        agent.action.offloading_ratio = np.zeros(3) # [alpha, (1-alpha)*beta, (1-alpha)*(1-beta)] = [local, LTE, 5G] (list)
        agent.action.trans_power_allocation_LTE = np.zeros(1) # float
        agent.action.trans_power_allocation_5G = np.zeros(1) # float

        ################### softmax 썼을때 (0~1, 합이 1) #########################
        # # association 연결된 MEC server index 설정
        # for i in range(len(self.world.landmarks)):
        #     if i/len(self.world.landmarks) <= action[0] < (i + 1)/len(self.world.landmarks):
        #         agent.action.association = i
        # if action[0] == 1:
        #     agent.action.association = len(self.world.landmarks)-1
        #
        # # CPU_allocation
        # agent.action.CPU_allocation = action[1] * (self.world.max_CPU - self.world.min_CPU) + self.world.min_CPU
        #
        # # offloading_ratio
        # if agent.name[0] == 'U':
        #     agent.action.offloading_ratio = [action[2], (1 - action[2]) * action[3], (1 - action[2]) * (1 - action[3])]
        #
        #     max_val = max(agent.action.offloading_ratio)
        #     for i in range(len(agent.action.offloading_ratio)):
        #         if agent.action.offloading_ratio[i] == max_val:
        #             agent.action.offloading_ratio[i] = 1
        #         else:
        #             agent.action.offloading_ratio[i] = 0
        #
        # else:
        #     agent.action.offloading_ratio = [action[2], (1 - action[2]) * action[3], (1 - action[2]) * (1 - action[3])]
        # # print(agent.action.offloading_ratio)
        #
        # # trans_power_allocation
        # agent.action.trans_power_allocation_LTE = action[4] * (self.world.max_p_trans_LTE - self.world.min_p_trans_LTE) + self.world.min_p_trans_LTE
        # agent.action.trans_power_allocation_5G = action[5] * (self.world.max_p_trans_5G - self.world.min_p_trans_5G) + self.world.min_p_trans_5G
        #
        # # RB_allocation
        # agent.action.RB_allocation_LTE = math.floor(action[6] * self.world.max_RB_number_LTE) + 1
        # agent.action.RB_allocation_5G = math.floor(action[7] * self.world.max_RB_number_5G) + 1
        ##########################

        ########## tanh 썼을때 (각각 -1~1) ##########
        # association 연결된 MEC server index 설정
        for i in range(len(self.world.landmarks)):
            if -1 + 2*i/len(self.world.landmarks) <= action[0] < -1 + 2*(i + 1)/len(self.world.landmarks):
                agent.action.association = i
        if action[0] == 1:
            agent.action.association = len(self.world.landmarks)-1
        # print('association', agent.action.association)

        # CPU_allocation
        agent.action.CPU_allocation = (action[1]+1)/2 * (self.world.max_CPU - self.world.min_CPU) + self.world.min_CPU

        # offloading_ratio
        if agent.name[0] == 'U':
            agent.action.offloading_ratio = [(action[2]+1)/2, (1 - (action[2]+1)/2) * (action[3]+1)/2, (1 - (action[2]+1)/2) * (1 - (action[3]+1)/2)]

            max_val = max(agent.action.offloading_ratio)
            for i in range(len(agent.action.offloading_ratio)):
                if agent.action.offloading_ratio[i] == max_val:
                    agent.action.offloading_ratio[i] = 1
                else:
                    agent.action.offloading_ratio[i] = 0

        else:
            agent.action.offloading_ratio = [(action[2]+1)/2, (1 - (action[2]+1)/2) * (action[3]+1)/2, (1 - (action[2]+1)/2) * (1 - (action[3]+1)/2)]
        # print('offloading_ratio', agent.action.offloading_ratio)

        # trans_power_allocation
        agent.action.trans_power_allocation_LTE = (action[4]+1)/2 * (self.world.max_p_trans_LTE - self.world.min_p_trans_LTE) + self.world.min_p_trans_LTE
        agent.action.trans_power_allocation_5G = (action[5]+1)/2 * (self.world.max_p_trans_5G - self.world.min_p_trans_5G) + self.world.min_p_trans_5G

    def _set_action2(self, action, agent):
        agent.action.RB_allocation_LTE = np.zeros(1)  # float
        agent.action.RB_allocation_5G = np.zeros(1)  # float

        # RB_allocation
        agent.action.RB_allocation_LTE = math.floor((action[6] + 1) / 2 * self.world.max_RB_number_LTE / agent.association_count_list[agent.action.association])
        if agent.action.RB_allocation_LTE == 0:
            agent.action.RB_allocation_LTE = 1
        agent.action.RB_allocation_5G = math.floor((action[7] + 1) / 2 * self.world.max_RB_number_5G / agent.association_count_list[agent.action.association])
        if agent.action.RB_allocation_5G == 0:
            agent.action.RB_allocation_5G = 1
        # agent.action.RB_allocation_LTE = math.floor((action[6] + 1) / 2 * self.world.max_RB_number_LTE ) + 1
        # agent.action.RB_allocation_5G = math.floor((action[7] + 1) / 2 * self.world.max_RB_number_5G ) + 1