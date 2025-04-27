import numpy as np
np.random.seed(0)

# state of agents
class AgentState(object):
    def __init__(self):
        # Task information
        self.arrival_rate = None
        self.task_size = None
        self.required_CPU = None
        self.latency_QoS = None
        self.reliability_QoS = None
        # CSI matrix
        self.large_scale_channel_gain = None
        self.small_scale_channel_gain = None


# action of the agent
class Action(object):
    def __init__(self):
        # Association
        self.association = None
        # CPU resource allocation
        self.CPU_allocation = None
        # Multi-RAT offloading ratio decision
        self.offloading_ratio = None
        # Transmission power allocation
        self.trans_power_allocation_LTE = None
        self.trans_power_allocation_5G = None
        # RB allocation
        self.RB_allocation_LTE = None
        self.RB_allocation_5G = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name
        self.name = ''
        # position
        self.x_pos = None
        self.y_pos = None


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.workload = None
        self.RB_number_LTE = None
        self.RB_number_5G = None


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.association_count_list = []
        self.local_order = None
        self.waiting_time = None


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # time duration for one time slot
        self.time_slot = 0.125 * (10**(-3)) # 0.125 ms
        # self.time_slot = 1 * (10 ** (-3)) # 1 ms
        # min & max CPU of UE
        self.min_CPU = 0.1 * (10**9)
        self.max_CPU = 2 * (10**9)
        # min & max Transmission power of UE for LTE (dBm)
        self.min_p_trans_LTE = 1
        self.max_p_trans_LTE = 23
        # min & max Transmission power of UE for 5G (dBm)
        self.min_p_trans_5G = 1
        self.max_p_trans_5G = 23
        # CPU resource of MEC server
        self.CPU_MEC = 5 * (10**9)
        # The size and maximum number of Resource block for LTE in frequency domain
        self.RB_size_LTE = 180 * (10**3)
        self.max_RB_number_LTE = 50 # 50
        # The size and maximum number of Resource block for 5G in frequency domain
        self.RB_size_5G = 180 * (10**3)
        self.max_RB_number_5G = 200 # 200
        # Environment radius
        self.radius = 38 # 380
        self.MEC_radius = 40 # 400
        # Noise spectral density (dBm/Hz)
        self.noise_density = -174
        # self.energy_coefficient = 10 ** (-15)
        self.energy_coefficient = 10**(-23) #23
        # self.energy_coefficient = 10 ** (-27)

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # update state of the world
    def step(self):
        for agent in self.agents:
            self.update_agent_state(agent)

    def update_agent_state(self, agent):
        # URLLC
        if agent.name[0] == 'U':
            # Task information
            agent.state.arrival_rate = 500
            agent.state.task_size = 32*8
            # agent.state.required_CPU = 330/8
            agent.state.required_CPU = np.random.randint(300, 400) / 8
            agent.state.latency_QoS = 10**(-3)
            agent.state.reliability_QoS = 10 ** (-5)
            # CSI matrix
            path_loss = 35.3 + 37.6 * np.log10(self.distance_from_associated_landmark(agent))
            agent.state.large_scale_channel_gain = 10 ** (-path_loss / 10)
            if path_loss < 0:
                agent.state.large_scale_channel_gain = 1
            # agent.state.large_scale_channel_gain = 10 ** (-(35.3 + 37.6 * np.log10(world.distance_from_associated_landmark(agent))) / 10)
            # agent.state.small_scale_channel_gain = np.sqrt(np.abs(np.random.normal(scale=np.sqrt(0.5)) + 1j * np.random.normal(scale=np.sqrt(0.5))) ** 2)
            agent.state.small_scale_channel_gain = abs(complex(np.random.normal(0, 1), np.random.normal(0, 1)))
        # eMBB
        else:
            # Task information 필요
            agent.state.arrival_rate = 5
            agent.state.task_size = np.random.randint(50*(2**10)*8, 100*(2**10)*8)
            # agent.state.required_CPU = 330/8
            agent.state.required_CPU = np.random.randint(300, 400) / 8
            agent.state.latency_QoS = 1
            agent.state.reliability_QoS = 1
            # CSI matrix 필요
            path_loss = 35.3 + 37.6 * np.log10(self.distance_from_associated_landmark(agent))
            agent.state.large_scale_channel_gain = 10 ** (-path_loss / 10)
            if path_loss < 0:
                agent.state.large_scale_channel_gain = 1
            # agent.state.small_scale_channel_gain = np.sqrt(np.abs(np.random.normal(scale=np.sqrt(0.5)) + 1j * np.random.normal(scale=np.sqrt(0.5))) ** 2)
            agent.state.small_scale_channel_gain = abs(complex(np.random.normal(0, 1), np.random.normal(0, 1)))

    def distance_from_associated_landmark(self, agent):
        return np.sqrt((agent.x_pos-self.landmarks[agent.action.association].x_pos)**2+(agent.y_pos-self.landmarks[agent.action.association].y_pos)**2)
