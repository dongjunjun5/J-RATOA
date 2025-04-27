import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import randompoint
import uniformpoint
import math
import decimal
decimal.getcontext().prec = 5
from scipy.special import erfcinv, erfinv, erf, erfc
import matplotlib.pyplot as plt
np.random.seed(0)

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        num_agents = 20 # 100
        num_landmarks = 4
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        points_urllc = []
        points_embb = []
        for i, agent in enumerate(world.agents):
            if i < len(world.agents)/2:
                agent.name = 'URLLC UE %d' % (i+1)
                agent.x_pos = uniformpoint.evenly_distributed_points_4vs6()[i][0]
                agent.y_pos = uniformpoint.evenly_distributed_points_4vs6()[i][1]
                points_urllc.append((agent.x_pos, agent.y_pos))
            else:
                agent.name = 'eMBB UE %d' % (i-len(world.agents)/2+1)
                agent.x_pos = uniformpoint.evenly_distributed_points_4vs6()[int(i - len(world.agents) / 2)][0] + 10
                agent.y_pos = uniformpoint.evenly_distributed_points_4vs6()[int(i - len(world.agents) / 2)][1] + 10
                points_embb.append((agent.x_pos, agent.y_pos))
            print(agent.name, agent.x_pos, agent.y_pos)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        points_landmark = []
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'MEC server %d' % (i+1)
            if i==0:
                landmark.x_pos = 0
                landmark.y_pos = 0
            else:
                landmark.x_pos = uniformpoint.evenly_distributed_points_in_circle(num_landmarks-1, world.MEC_radius)[i-1][0]
                landmark.y_pos = uniformpoint.evenly_distributed_points_in_circle(num_landmarks-1, world.MEC_radius)[i-1][1]
            points_landmark.append((landmark.x_pos, landmark.y_pos))
            # print(landmark.name, landmark.x_pos, landmark.y_pos)


        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # URLLC UE state
            if i < len(world.agents)/2:
                # Task information
                agent.state.arrival_rate = 500
                agent.state.task_size = 32*8
                # agent.state.required_CPU = 330/8
                agent.state.required_CPU = np.random.randint(300, 400) / 8
                agent.state.latency_QoS = 10**(-3)
                agent.state.reliability_QoS = 10 ** (-5)
                # # CSI matrix  랜덤하게 설정
                # path_loss = 35.3 + 37.6 * np.log10(np.random.uniform(1, 200))
                # CSI matrix  랜덤이아니라 기존 제일 가까운 AP 기준으로
                reset_dist_list = []
                for j, landmark in enumerate(world.landmarks):
                    reset_dist_list.append(np.sqrt((agent.x_pos - landmark.x_pos) ** 2 + (agent.y_pos - landmark.y_pos) ** 2))
                reset_dist = min(reset_dist_list)
                path_loss = 35.3 + 37.6 * np.log10(reset_dist)

                agent.state.large_scale_channel_gain = 10 ** (-path_loss / 10)
                if path_loss < 0:
                    agent.state.large_scale_channel_gain = 1
                # agent.state.large_scale_channel_gain = 10 ** (-(35.3 + 37.6 * np.log10(world.distance_from_associated_landmark(agent))) / 10)
                agent.state.small_scale_channel_gain = abs(complex(np.random.normal(0,1), np.random.normal(0,1)))
            # eMBB UE state
            else:
                # Task information 필요
                agent.state.arrival_rate = 5
                agent.state.task_size = np.random.randint(50*(2**10)*8, 100*(2**10)*8)
                # agent.state.required_CPU = 330/8
                agent.state.required_CPU = np.random.randint(300, 400) / 8
                agent.state.latency_QoS = 1
                agent.state.reliability_QoS = 1
                # # CSI matrix  랜덤하게 설정
                # path_loss = 35.3 + 37.6 * np.log10(np.random.uniform(1, 200))
                # CSI matrix  랜덤이아니라 기존 제일 가까운 AP 기준으로
                reset_dist_list = []
                for j, landmark in enumerate(world.landmarks):
                    reset_dist_list.append(np.sqrt((agent.x_pos - landmark.x_pos) ** 2 + (agent.y_pos - landmark.y_pos) ** 2))
                reset_dist = min(reset_dist_list)
                path_loss = 35.3 + 37.6 * np.log10(reset_dist)

                agent.state.large_scale_channel_gain = 10 ** (-path_loss / 10)
                if path_loss < 0:
                    agent.state.large_scale_channel_gain = 1
                # agent.state.small_scale_channel_gain = np.sqrt(np.abs(np.random.normal(scale=np.sqrt(0.5)) + 1j * np.random.normal(scale=np.sqrt(0.5))) ** 2)
                agent.state.small_scale_channel_gain = abs(complex(np.random.normal(0, 1), np.random.normal(0, 1)))

    def reward(self, agent, world):
        # Agents are rewarded based on energy consumption
        # URLLC
        if agent.name[0] == 'U':
            if agent.action.offloading_ratio[0] == 1:  ## local
                # latency (local)
                required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
                processing_time_local = math.ceil(required_cpu_for_task / agent.action.CPU_allocation / world.time_slot)
                # queueing_delay = agent.waiting_time
                queueing_delay = 0 # arrival rate << service rate
                total_latency = processing_time_local + queueing_delay

                # reliability (local)
                if agent.action.offloading_ratio[0] == 1: # local_queueing_delay_violation_prob: probability(queueing_delay>(agent.state.latency_QoS-processing_time_local))
                    l = agent.state.latency_QoS / world.time_slot - processing_time_local
                    q = agent.state.arrival_rate * world.time_slot
                    rho = q * processing_time_local
                    local_queueing_delay_violation_prob = 1 - ((1 - q) ** (-l - 1)) * (1 - rho)

                # energy (local)
                energy_consumption = world.energy_coefficient * (agent.action.CPU_allocation ** 2) * required_cpu_for_task
                print('URLLC local', energy_consumption / agent.state.task_size)

            elif agent.action.offloading_ratio[1] == 1: ## LTE
                # communication model (LTE)
                noise_density_mw = 10 ** (world.noise_density / 10)
                channel_bw = agent.action.RB_allocation_LTE * world.RB_size_LTE
                trans_power_allocation_mw = 10 ** (agent.action.trans_power_allocation_LTE / 10)
                snr = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / channel_bw / noise_density_mw
                snr_error = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / world.RB_size_LTE / noise_density_mw
                channel_dispersion = 1 - 1/(1 + snr)**2
                channel_dispersion_error = 1 - 1/(1 + snr_error)**2
                # decoding_error_rate = (1/2)*erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion) * (math.log(1 + snr) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw))/np.sqrt(2))
                # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion_error) * (math.log(1 + snr_error) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
                # decoding_error_rate = 0.5 * (10 ** (-5))
                decoding_error_rate = 0
                # achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr) - math.sqrt(channel_dispersion / world.time_slot / channel_bw) * (np.sqrt(2)*erfinv(1-2*decoding_error_rate)))
                achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr_error) - math.sqrt(channel_dispersion_error / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
                data_rate = channel_bw * math.log2(1 + snr)

                # latency (offloading)
                required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
                processing_time_mec = math.ceil(required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot)
                if data_rate > achievable_data_rate:
                    transmission_delay = 1
                else:
                    transmission_delay = math.inf
                # transmission_delay = math.ceil(agent.state.task_size / data_rate / world.time_slot)

                total_latency = transmission_delay + processing_time_mec

                # reliability (offloading)
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        mec_queueing_delay_violation_prob = landmark.workload ** (world.CPU_MEC*world.time_slot * (agent.state.latency_QoS/world.time_slot - transmission_delay) / required_cpu_for_task - 1)
                        if mec_queueing_delay_violation_prob > 1:
                            mec_queueing_delay_violation_prob = 1

                packet_loss_probability = 1-(1-decoding_error_rate)*(1-mec_queueing_delay_violation_prob)

                # energy (offloading)
                energy_consumption = agent.action.trans_power_allocation_LTE * transmission_delay * world.time_slot
                print('URLLC off_LTE', energy_consumption / agent.state.task_size)
                # print('decoding_error_rate',decoding_error_rate)
            elif agent.action.offloading_ratio[2] == 1: ## 5G
                # communication model (5G)
                noise_density_mw = 10 ** (world.noise_density / 10)
                channel_bw = agent.action.RB_allocation_5G * world.RB_size_5G
                trans_power_allocation_mw = 10 ** (agent.action.trans_power_allocation_5G / 10)
                snr = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / channel_bw / noise_density_mw
                snr_error = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / world.RB_size_5G / noise_density_mw
                channel_dispersion = 1 - 1/(1 + snr)**2
                channel_dispersion_error = 1 - 1 / (1 + snr_error) ** 2
                # decoding_error_rate = (1/2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion) * (math.log(1 + snr) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
                # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion_error) * (math.log(1 + snr_error) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
                # decoding_error_rate = 0.5 * (10 ** (-5))
                decoding_error_rate = 0
                # achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr) - math.sqrt(channel_dispersion / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
                achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr_error) - math.sqrt(channel_dispersion_error / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
                data_rate = channel_bw * math.log2(1 + snr)

                # latency (offloading)
                required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
                processing_time_mec = math.ceil(required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot)
                if data_rate > achievable_data_rate:
                    transmission_delay = 1
                else:
                    transmission_delay = math.inf

                total_latency = transmission_delay + processing_time_mec

                # reliability (offloading)
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        mec_queueing_delay_violation_prob = landmark.workload ** (world.CPU_MEC * world.time_slot * (
                                    agent.state.latency_QoS / world.time_slot - transmission_delay) / required_cpu_for_task - 1)
                        if mec_queueing_delay_violation_prob > 1:
                            mec_queueing_delay_violation_prob = 1

                packet_loss_probability = 1 - (1 - decoding_error_rate) * (1 - mec_queueing_delay_violation_prob)

                # energy (offloading)
                energy_consumption = agent.action.trans_power_allocation_5G * transmission_delay * world.time_slot
                print('URLLC off_5G', energy_consumption / agent.state.task_size)
                # print('decoding_error_rate',decoding_error_rate)
            # normalized total energy consumption
            total_energy_consumption = energy_consumption / agent.state.task_size

            # if agent.action.offloading_ratio[0] != 1:
            #     print('decoding_error_rate', decoding_error_rate)
            #     print('achievable_data_rate', achievable_data_rate)
            #     print('agent.action.offloading_ratio',agent.action.offloading_ratio)
            #     print('data_rate', data_rate)
            #     print('transmission_delay', transmission_delay)
            #     print('packet_loss_probability', packet_loss_probability)

        # eMBB
        else:
            # communication model (LTE)
            noise_density_mw = 10 ** (world.noise_density / 10)
            channel_bw_lte = agent.action.RB_allocation_LTE * world.RB_size_LTE
            trans_power_allocation_lte_mw = 10 ** (agent.action.trans_power_allocation_LTE / 10)
            snr_lte = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_lte_mw / channel_bw_lte / noise_density_mw

            data_rate_lte = channel_bw_lte * math.log2(1+snr_lte)

            # communication model (5G)
            noise_density_mw = 10 ** (world.noise_density / 10)
            channel_bw_5g = agent.action.RB_allocation_5G * world.RB_size_5G
            trans_power_allocation_5g_mw = 10 ** (agent.action.trans_power_allocation_5G / 10)
            snr_5g = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_5g_mw / channel_bw_5g / noise_density_mw

            data_rate_5g = channel_bw_5g * math.log2(1+snr_5g)

            # latency (local)
            required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
            processing_time_local = math.ceil(agent.action.offloading_ratio[0] * required_cpu_for_task / agent.action.CPU_allocation / world.time_slot)
            # queueing_delay = agent.waiting_time
            queueing_delay = 0  # arrival rate << service rate
            total_latency_local = processing_time_local + queueing_delay

            # latency (offloading)
            required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
            processing_time_mec = sum([math.ceil(agent.action.offloading_ratio[1] * required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association] ) / world.time_slot), math.ceil(agent.action.offloading_ratio[2] * required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association] ) / world.time_slot)])
            transmission_delay_lte = math.ceil(agent.action.offloading_ratio[1] * agent.state.task_size / data_rate_lte / world.time_slot)
            if data_rate_lte <= 0:
                transmission_delay_lte = math.inf
            transmission_delay_5g = math.ceil(agent.action.offloading_ratio[2] * agent.state.task_size / data_rate_5g / world.time_slot)
            if data_rate_5g <= 0:
                transmission_delay_5g = math.inf
            transmission_delay = max(transmission_delay_lte, transmission_delay_5g)
            total_latency_offloading = transmission_delay + processing_time_mec

            # energy (local)
            energy_consumption_local = agent.action.offloading_ratio[0] * world.energy_coefficient * (agent.action.CPU_allocation**2) * required_cpu_for_task

            # energy (offloading)
            energy_consumption_offloading = sum([agent.action.offloading_ratio[1]*agent.action.trans_power_allocation_LTE*transmission_delay_lte*world.time_slot, agent.action.offloading_ratio[2]*agent.action.trans_power_allocation_5G*transmission_delay_5g*world.time_slot])

            # total latency
            total_latency = max(total_latency_local, total_latency_offloading)

            # total energy consumption
            total_energy_consumption = energy_consumption_local + energy_consumption_offloading
            print('eMBB local', energy_consumption_local / agent.state.task_size, 'eMBB off', energy_consumption_offloading / agent.state.task_size)
            # normalized total energy consumption
            total_energy_consumption = total_energy_consumption / agent.state.task_size
        # print('agent.action.offloading_ratio', agent.action.offloading_ratio)
        # rew = - total_energy_consumption
        rew = - total_energy_consumption * 10 ** 5
        # rew = - total_energy_consumption * 10 ** (-6)

        for i, landmark in enumerate(world.landmarks):
            if i == agent.action.association:
                total_rb_number_lte = landmark.RB_number_LTE
                total_rb_number_5g = landmark.RB_number_5G

        # if #constraint들 벗어나는 경우
        if total_rb_number_lte > world.max_RB_number_LTE or total_rb_number_5g > world.max_RB_number_5G:
            # rew = - 100
            # rew = - 10
            rew = rew * 10
            print('1', 'total_rb_number_lte', total_rb_number_lte, 'total_rb_number_5g', total_rb_number_5g)
        if agent.name[0] == 'U':
            if agent.action.offloading_ratio[0] == 1:
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        workload_mec_for_constraint = landmark.workload
                if workload_mec_for_constraint > 1:
                    # rew = - 101
                    # rew = - 10
                    rew = rew * 10
                    print('2')
            else:
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        workload_mec_for_constraint = landmark.workload
                workload_threshold = (0.5 * agent.state.reliability_QoS)**(required_cpu_for_task/(world.CPU_MEC*(agent.state.latency_QoS/world.time_slot - transmission_delay)*world.time_slot-required_cpu_for_task))
                if workload_mec_for_constraint > workload_threshold:
                    # rew = - 102
                    # rew = - 10
                    rew = rew * 10
                    # print('3')
        if agent.name[0] == 'U':
            if agent.action.offloading_ratio[0] == 1:
                if local_queueing_delay_violation_prob > agent.state.reliability_QoS:
                    # rew = - 103
                    # rew = - 10
                    rew = rew * 10
                    print('4')
            else:
                if packet_loss_probability > agent.state.reliability_QoS:
                    # rew = - 104
                    # rew = - 10
                    rew = rew * 10
                    print('5')
                if decoding_error_rate > (0.5 * agent.state.reliability_QoS):
                    # rew = - 105
                    # rew = - 10
                    rew = rew * 10
                    print('6')
                if mec_queueing_delay_violation_prob > (0.5 * agent.state.reliability_QoS):
                    # rew = - 106
                    # rew = - 10
                    rew = rew * 10
                    print('7')
        if total_latency > agent.state.latency_QoS/world.time_slot:
            # rew = - 107
            # rew = - 10
            rew = rew * 10
            print('8', total_latency)
        if agent.name[0] == 'e':
            if agent.action.CPU_allocation < (agent.action.offloading_ratio[0] * agent.state.arrival_rate * (75*(2**10)*8)):
                # rew = - 108
                # rew = - 10
                rew = rew * 10
                print('9')
        # print(rew)
        return [rew, total_latency*world.time_slot]

    def done(self, agent, world):
        done = 0

        # URLLC
        # if agent.name[0] == 'U':
        #     if agent.action.offloading_ratio[0] == 1:  ## local
        #         # latency (local)
        #         required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
        #         processing_time_local = math.ceil(required_cpu_for_task / agent.action.CPU_allocation / world.time_slot)
        #         # queueing_delay = agent.waiting_time
        #         queueing_delay = 0  # arrival rate << service rate
        #         total_latency = processing_time_local + queueing_delay
        #
        #         # reliability (local)
        #         if agent.action.offloading_ratio[
        #             0] == 1:  # local_queueing_delay_violation_prob: probability(queueing_delay>(agent.state.latency_QoS-processing_time_local))
        #             l = agent.state.latency_QoS / world.time_slot - processing_time_local
        #             q = agent.state.arrival_rate * world.time_slot
        #             rho = q * processing_time_local
        #             local_queueing_delay_violation_prob = 1 - ((1 - q) ** (-l - 1)) * (1 - rho)
        #
        #         # energy (local)
        #         energy_consumption = world.energy_coefficient * (
        #                     agent.action.CPU_allocation ** 2) * required_cpu_for_task
        #
        #     elif agent.action.offloading_ratio[1] == 1:  ## LTE
        #         # communication model (LTE)
        #         noise_density_mw = 10 ** (world.noise_density / 10)
        #         channel_bw = agent.action.RB_allocation_LTE * world.RB_size_LTE
        #         trans_power_allocation_mw = 10 ** (agent.action.trans_power_allocation_LTE / 10)
        #         snr = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / channel_bw / noise_density_mw
        #         snr_error = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / world.RB_size_LTE / noise_density_mw
        #         channel_dispersion = 1 - 1 / (1 + snr) ** 2
        #         channel_dispersion_error = 1 - 1 / (1 + snr_error) ** 2
        #         # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion) * (math.log(1 + snr) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
        #         # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion_error) * (math.log(1 + snr_error) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
        #         # decoding_error_rate = 0.5 * (10 ** (-5))
        #         decoding_error_rate = 0
        #         # achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr) - math.sqrt(channel_dispersion / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
        #         achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr_error) - math.sqrt(channel_dispersion_error / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
        #         data_rate = channel_bw * math.log2(1 + snr)
        #
        #         # latency (offloading)
        #         required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
        #         processing_time_mec = math.ceil(required_cpu_for_task / (
        #                     world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot)
        #         if data_rate > achievable_data_rate:
        #             transmission_delay = 1
        #         else:
        #             transmission_delay = math.inf
        #         # transmission_delay = math.ceil(agent.state.task_size / data_rate / world.time_slot)
        #
        #         total_latency = transmission_delay + processing_time_mec
        #
        #         # reliability (offloading)
        #         for i, landmark in enumerate(world.landmarks):
        #             if i == agent.action.association:
        #                 mec_queueing_delay_violation_prob = landmark.workload ** (world.CPU_MEC * world.time_slot * (
        #                             agent.state.latency_QoS / world.time_slot - transmission_delay) / required_cpu_for_task - 1)
        #                 if mec_queueing_delay_violation_prob > 1:
        #                     mec_queueing_delay_violation_prob = 1
        #
        #         packet_loss_probability = 1 - (1 - decoding_error_rate) * (1 - mec_queueing_delay_violation_prob)
        #
        #         # energy (offloading)
        #         energy_consumption = agent.action.trans_power_allocation_LTE * transmission_delay
        #
        #     elif agent.action.offloading_ratio[2] == 1:  ## 5G
        #         # communication model (5G)
        #         noise_density_mw = 10 ** (world.noise_density / 10)
        #         channel_bw = agent.action.RB_allocation_5G * world.RB_size_5G
        #         trans_power_allocation_mw = 10 ** (agent.action.trans_power_allocation_5G / 10)
        #         snr = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / channel_bw / noise_density_mw
        #         snr_error = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / world.RB_size_LTE / noise_density_mw
        #         channel_dispersion = 1 - 1 / (1 + snr) ** 2
        #         channel_dispersion_error = 1 - 1 / (1 + snr_error) ** 2
        #         # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion) * (math.log(1 + snr) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
        #         # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion_error) * (math.log(1 + snr_error) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
        #         # decoding_error_rate = 0.5 * (10 ** (-5))
        #         decoding_error_rate = 0
        #         # achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr) - math.sqrt(channel_dispersion / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
        #         achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr_error) - math.sqrt(channel_dispersion_error / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
        #         data_rate = channel_bw * math.log2(1 + snr)
        #
        #         # latency (offloading)
        #         required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
        #         processing_time_mec = math.ceil(required_cpu_for_task / (
        #                     world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot)
        #         if data_rate > achievable_data_rate:
        #             transmission_delay = 1
        #         else:
        #             transmission_delay = math.inf
        #         # transmission_delay = math.ceil(agent.state.task_size / data_rate / world.time_slot)
        #
        #         total_latency = transmission_delay + processing_time_mec
        #
        #         # reliability (offloading)
        #         for i, landmark in enumerate(world.landmarks):
        #             if i == agent.action.association:
        #                 mec_queueing_delay_violation_prob = landmark.workload ** (world.CPU_MEC * world.time_slot * (
        #                         agent.state.latency_QoS / world.time_slot - transmission_delay) / required_cpu_for_task - 1)
        #                 if mec_queueing_delay_violation_prob > 1:
        #                     mec_queueing_delay_violation_prob = 1
        #
        #         packet_loss_probability = 1 - (1 - decoding_error_rate) * (1 - mec_queueing_delay_violation_prob)
        #
        #         # energy (offloading)
        #         energy_consumption = agent.action.trans_power_allocation_5G * transmission_delay
        #
        #     # normalized total energy consumption
        #     total_energy_consumption = energy_consumption / agent.state.task_size
        #
        # # eMBB
        # else:
        #     # communication model (LTE)
        #     noise_density_mw = 10 ** (world.noise_density / 10)
        #     channel_bw_lte = agent.action.RB_allocation_LTE * world.RB_size_LTE
        #     trans_power_allocation_lte_mw = 10 ** (agent.action.trans_power_allocation_LTE / 10)
        #     snr_lte = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_lte_mw / channel_bw_lte / noise_density_mw
        #
        #     data_rate_lte = channel_bw_lte * math.log2(1 + snr_lte)
        #
        #     # communication model (5G)
        #     noise_density_mw = 10 ** (world.noise_density / 10)
        #     channel_bw_5g = agent.action.RB_allocation_5G * world.RB_size_5G
        #     trans_power_allocation_5g_mw = 10 ** (agent.action.trans_power_allocation_5G / 10)
        #     snr_5g = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_5g_mw / channel_bw_5g / noise_density_mw
        #
        #     data_rate_5g = channel_bw_5g * math.log2(1 + snr_5g)
        #
        #     # latency (local)
        #     required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
        #     processing_time_local = math.ceil(agent.action.offloading_ratio[
        #                                           0] * required_cpu_for_task / agent.action.CPU_allocation / world.time_slot)
        #     # queueing_delay = agent.waiting_time
        #     queueing_delay = 0  # arrival rate << service rate
        #     total_latency_local = processing_time_local + queueing_delay
        #
        #     # latency (offloading)
        #     required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
        #     processing_time_mec = sum([math.ceil(agent.action.offloading_ratio[1] * required_cpu_for_task / (
        #                 world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot),
        #                                math.ceil(agent.action.offloading_ratio[2] * required_cpu_for_task / (
        #                                            world.CPU_MEC / agent.association_count_list[
        #                                        agent.action.association]) / world.time_slot)])
        #     transmission_delay_lte = math.ceil(
        #         agent.action.offloading_ratio[1] * agent.state.task_size / data_rate_lte / world.time_slot)
        #     if data_rate_lte <= 0:
        #         transmission_delay_lte = math.inf
        #     transmission_delay_5g = math.ceil(
        #         agent.action.offloading_ratio[2] * agent.state.task_size / data_rate_5g / world.time_slot)
        #     if data_rate_5g <= 0:
        #         transmission_delay_5g = math.inf
        #     transmission_delay = max(transmission_delay_lte, transmission_delay_5g)
        #     total_latency_offloading = transmission_delay + processing_time_mec
        #
        #     # energy (local)
        #     energy_consumption_local = agent.action.offloading_ratio[0] * world.energy_coefficient * (
        #                 agent.action.CPU_allocation ** 2) * required_cpu_for_task
        #
        #     # energy (offloading)
        #     energy_consumption_offloading = sum(
        #         [agent.action.offloading_ratio[1] * agent.action.trans_power_allocation_LTE * transmission_delay_lte,
        #          agent.action.offloading_ratio[2] * agent.action.trans_power_allocation_5G * transmission_delay_5g])
        #
        #     # total latency
        #     total_latency = max(total_latency_local, total_latency_offloading)
        #
        # for i, landmark in enumerate(world.landmarks):
        #     if i == agent.action.association:
        #         total_rb_number_lte = landmark.RB_number_LTE
        #         total_rb_number_5g = landmark.RB_number_5G

        if agent.name[0] == 'U':
            if agent.action.offloading_ratio[0] == 1:  ## local
                # latency (local)
                required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
                processing_time_local = math.ceil(required_cpu_for_task / agent.action.CPU_allocation / world.time_slot)
                # queueing_delay = agent.waiting_time
                queueing_delay = 0 # arrival rate << service rate
                total_latency = processing_time_local + queueing_delay

                # reliability (local)
                if agent.action.offloading_ratio[0] == 1: # local_queueing_delay_violation_prob: probability(queueing_delay>(agent.state.latency_QoS-processing_time_local))
                    l = agent.state.latency_QoS / world.time_slot - processing_time_local
                    q = agent.state.arrival_rate * world.time_slot
                    rho = q * processing_time_local
                    local_queueing_delay_violation_prob = 1 - ((1 - q) ** (-l - 1)) * (1 - rho)

                # energy (local)
                energy_consumption = world.energy_coefficient * (agent.action.CPU_allocation ** 2) * required_cpu_for_task
                # print('URLLC local', energy_consumption / agent.state.task_size)

            elif agent.action.offloading_ratio[1] == 1: ## LTE
                # communication model (LTE)
                noise_density_mw = 10 ** (world.noise_density / 10)
                channel_bw = agent.action.RB_allocation_LTE * world.RB_size_LTE
                trans_power_allocation_mw = 10 ** (agent.action.trans_power_allocation_LTE / 10)
                snr = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / channel_bw / noise_density_mw
                snr_error = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / world.RB_size_LTE / noise_density_mw
                channel_dispersion = 1 - 1/(1 + snr)**2
                channel_dispersion_error = 1 - 1/(1 + snr_error)**2
                # decoding_error_rate = (1/2)*erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion) * (math.log(1 + snr) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw))/np.sqrt(2))
                # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion_error) * (math.log(1 + snr_error) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
                # decoding_error_rate = 0.5 * (10 ** (-5))
                decoding_error_rate = 0
                # achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr) - math.sqrt(channel_dispersion / world.time_slot / channel_bw) * (np.sqrt(2)*erfinv(1-2*decoding_error_rate)))
                achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr_error) - math.sqrt(channel_dispersion_error / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
                data_rate = channel_bw * math.log2(1 + snr)

                # latency (offloading)
                required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
                processing_time_mec = math.ceil(required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot)
                if data_rate > achievable_data_rate:
                    transmission_delay = 1
                else:
                    transmission_delay = math.inf
                # transmission_delay = math.ceil(agent.state.task_size / data_rate / world.time_slot)

                total_latency = transmission_delay + processing_time_mec

                # reliability (offloading)
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        mec_queueing_delay_violation_prob = landmark.workload ** (world.CPU_MEC*world.time_slot * (agent.state.latency_QoS/world.time_slot - transmission_delay) / required_cpu_for_task - 1)
                        if mec_queueing_delay_violation_prob > 1:
                            mec_queueing_delay_violation_prob = 1

                packet_loss_probability = 1-(1-decoding_error_rate)*(1-mec_queueing_delay_violation_prob)

                # energy (offloading)
                energy_consumption = agent.action.trans_power_allocation_LTE * transmission_delay * world.time_slot
                # print('URLLC off_LTE', energy_consumption / agent.state.task_size)
                # print('decoding_error_rate',decoding_error_rate)
            elif agent.action.offloading_ratio[2] == 1: ## 5G
                # communication model (5G)
                noise_density_mw = 10 ** (world.noise_density / 10)
                channel_bw = agent.action.RB_allocation_5G * world.RB_size_5G
                trans_power_allocation_mw = 10 ** (agent.action.trans_power_allocation_5G / 10)
                snr = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / channel_bw / noise_density_mw
                snr_error = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_mw / world.RB_size_5G / noise_density_mw
                channel_dispersion = 1 - 1/(1 + snr)**2
                channel_dispersion_error = 1 - 1 / (1 + snr_error) ** 2
                # decoding_error_rate = (1/2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion) * (math.log(1 + snr) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
                # decoding_error_rate = (1 / 2) * erfc((np.sqrt(world.time_slot * channel_bw / channel_dispersion_error) * (math.log(1 + snr_error) - agent.state.task_size * math.log(2) / world.time_slot / channel_bw)) / np.sqrt(2))
                # decoding_error_rate = 0.5 * (10 ** (-5))
                decoding_error_rate = 0
                # achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr) - math.sqrt(channel_dispersion / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
                achievable_data_rate = channel_bw / math.log(2) * (math.log(1 + snr_error) - math.sqrt(channel_dispersion_error / world.time_slot / channel_bw) * (np.sqrt(2) * erfinv(1 - 2 * decoding_error_rate)))
                data_rate = channel_bw * math.log2(1 + snr)

                # latency (offloading)
                required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
                processing_time_mec = math.ceil(required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association]) / world.time_slot)
                if data_rate > achievable_data_rate:
                    transmission_delay = 1
                else:
                    transmission_delay = math.inf

                total_latency = transmission_delay + processing_time_mec

                # reliability (offloading)
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        mec_queueing_delay_violation_prob = landmark.workload ** (world.CPU_MEC * world.time_slot * (
                                    agent.state.latency_QoS / world.time_slot - transmission_delay) / required_cpu_for_task - 1)
                        if mec_queueing_delay_violation_prob > 1:
                            mec_queueing_delay_violation_prob = 1

                packet_loss_probability = 1 - (1 - decoding_error_rate) * (1 - mec_queueing_delay_violation_prob)

                # energy (offloading)
                energy_consumption = agent.action.trans_power_allocation_5G * transmission_delay * world.time_slot
                # print('URLLC off_5G', energy_consumption / agent.state.task_size)
                # print('decoding_error_rate',decoding_error_rate)
            # normalized total energy consumption
            total_energy_consumption = energy_consumption / agent.state.task_size

            # if agent.action.offloading_ratio[0] != 1:
            #     print('decoding_error_rate', decoding_error_rate)
            #     print('achievable_data_rate', achievable_data_rate)
            #     print('agent.action.offloading_ratio',agent.action.offloading_ratio)
            #     print('data_rate', data_rate)
            #     print('transmission_delay', transmission_delay)
            #     print('packet_loss_probability', packet_loss_probability)

        # eMBB
        else:
            # communication model (LTE)
            noise_density_mw = 10 ** (world.noise_density / 10)
            channel_bw_lte = agent.action.RB_allocation_LTE * world.RB_size_LTE
            trans_power_allocation_lte_mw = 10 ** (agent.action.trans_power_allocation_LTE / 10)
            snr_lte = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_lte_mw / channel_bw_lte / noise_density_mw

            data_rate_lte = channel_bw_lte * math.log2(1+snr_lte)

            # communication model (5G)
            noise_density_mw = 10 ** (world.noise_density / 10)
            channel_bw_5g = agent.action.RB_allocation_5G * world.RB_size_5G
            trans_power_allocation_5g_mw = 10 ** (agent.action.trans_power_allocation_5G / 10)
            snr_5g = agent.state.large_scale_channel_gain * agent.state.small_scale_channel_gain * trans_power_allocation_5g_mw / channel_bw_5g / noise_density_mw

            data_rate_5g = channel_bw_5g * math.log2(1+snr_5g)

            # latency (local)
            required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
            processing_time_local = math.ceil(agent.action.offloading_ratio[0] * required_cpu_for_task / agent.action.CPU_allocation / world.time_slot)
            # queueing_delay = agent.waiting_time
            queueing_delay = 0  # arrival rate << service rate
            total_latency_local = processing_time_local + queueing_delay

            # latency (offloading)
            required_cpu_for_task = agent.state.task_size * agent.state.required_CPU
            processing_time_mec = sum([math.ceil(agent.action.offloading_ratio[1] * required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association] ) / world.time_slot), math.ceil(agent.action.offloading_ratio[2] * required_cpu_for_task / (world.CPU_MEC / agent.association_count_list[agent.action.association] ) / world.time_slot)])
            transmission_delay_lte = math.ceil(agent.action.offloading_ratio[1] * agent.state.task_size / data_rate_lte / world.time_slot)
            if data_rate_lte <= 0:
                transmission_delay_lte = math.inf
            transmission_delay_5g = math.ceil(agent.action.offloading_ratio[2] * agent.state.task_size / data_rate_5g / world.time_slot)
            if data_rate_5g <= 0:
                transmission_delay_5g = math.inf
            transmission_delay = max(transmission_delay_lte, transmission_delay_5g)
            total_latency_offloading = transmission_delay + processing_time_mec

            # energy (local)
            energy_consumption_local = agent.action.offloading_ratio[0] * world.energy_coefficient * (agent.action.CPU_allocation**2) * required_cpu_for_task

            # energy (offloading)
            energy_consumption_offloading = sum([agent.action.offloading_ratio[1]*agent.action.trans_power_allocation_LTE*transmission_delay_lte*world.time_slot, agent.action.offloading_ratio[2]*agent.action.trans_power_allocation_5G*transmission_delay_5g*world.time_slot])

            # total latency
            total_latency = max(total_latency_local, total_latency_offloading)

            # total energy consumption
            total_energy_consumption = energy_consumption_local + energy_consumption_offloading
            # print('eMBB local', energy_consumption_local / agent.state.task_size, 'eMBB off', energy_consumption_offloading / agent.state.task_size)
            # normalized total energy consumption
            total_energy_consumption = total_energy_consumption / agent.state.task_size

        for i, landmark in enumerate(world.landmarks):
            if i == agent.action.association:
                total_rb_number_lte = landmark.RB_number_LTE
                total_rb_number_5g = landmark.RB_number_5G

        # if #constraint들 벗어나는 경우
        if total_rb_number_lte > world.max_RB_number_LTE or total_rb_number_5g > world.max_RB_number_5G:
            done = 1
        if agent.name[0] == 'U':
            if agent.action.offloading_ratio[0] == 1:
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        workload_mec_for_constraint = landmark.workload
                if workload_mec_for_constraint > 1:
                    done = 1
            else:
                for i, landmark in enumerate(world.landmarks):
                    if i == agent.action.association:
                        workload_mec_for_constraint = landmark.workload
                workload_threshold = (0.5 * agent.state.reliability_QoS) ** (required_cpu_for_task / (world.CPU_MEC * (agent.state.latency_QoS / world.time_slot - transmission_delay) * world.time_slot- required_cpu_for_task))
                if workload_mec_for_constraint > workload_threshold:
                    done = 1
        if agent.name[0] == 'U':
            if agent.action.offloading_ratio[0] == 1:
                if local_queueing_delay_violation_prob > agent.state.reliability_QoS:
                    done = 1
            else:
                if packet_loss_probability > agent.state.reliability_QoS:
                    done = 1
                    # print('done 5')
                if decoding_error_rate > (0.5 * agent.state.reliability_QoS):
                    done = 1
                    # print('done 6')
                if mec_queueing_delay_violation_prob > (0.5 * agent.state.reliability_QoS):
                    done = 1
                    # print('done 7')
        if total_latency > agent.state.latency_QoS / world.time_slot:
            done = 1
            # print('done 8')
        if agent.name[0] == 'e':
            if agent.action.CPU_allocation < (agent.action.offloading_ratio[0] * agent.state.arrival_rate * (75 * (2 ** 10) * 8)):
                done = 1
                # print('done 9')

        return done

    def observation(self, agent, world):
        if agent.name[0] == 'U':
            norm_arrival_rate = 1
            norm_task_size = 1
            norm_required_CPU = agent.state.required_CPU / (350/8)
            norm_latency_QoS = 1
            norm_reliability_QoS = 1
            # norm_large_scale_channel_gain = agent.state.large_scale_channel_gain / (10**(-9.8)) + 1
            norm_large_scale_channel_gain = agent.state.large_scale_channel_gain + 0.5
            norm_small_scale_channel_gain = agent.state.small_scale_channel_gain + 1

        else:
            norm_arrival_rate = 1
            norm_task_size = agent.state.task_size / (75 * (2 ** 10) * 8)
            norm_required_CPU = agent.state.required_CPU / (350/8)
            norm_latency_QoS = 1
            norm_reliability_QoS = 1
            # norm_large_scale_channel_gain = agent.state.large_scale_channel_gain / (10**(-9.8)) + 1
            norm_large_scale_channel_gain = agent.state.large_scale_channel_gain + 0.5
            norm_small_scale_channel_gain = agent.state.small_scale_channel_gain + 1

        return np.concatenate([[norm_arrival_rate]] + [[norm_task_size]] + [[norm_required_CPU]] + [[norm_latency_QoS]] + [[norm_reliability_QoS]] + [[norm_large_scale_channel_gain]] + [[norm_small_scale_channel_gain]])
        # return np.concatenate([[agent.state.arrival_rate]]+[[agent.state.task_size]]+[[agent.state.required_CPU]]+[[agent.state.latency_QoS]]+[[agent.state.reliability_QoS]]+[[agent.state.large_scale_channel_gain]]+[[agent.state.small_scale_channel_gain]])