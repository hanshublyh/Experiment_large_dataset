from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [],
                                   'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (
                        num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (
                        num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def show(env, stop_list, trf_list, dids_list, mac_kind, sev_kind, mac_sev_list, cpus, mems, return_list, rank, reward_kind, a_num):
    total_trf = []
    stop_num = []
    for i in stop_list:
        stop_num.append(sum(i))
    for i in trf_list:
        total_trf.append(sum(i))

    # 最大关停的第一个
    max_stop = max(stop_num)
    max_stop_index = []
    index_s = stop_num.index(max_stop)
    max_stop_list = stop_list[index_s]
    # 最大关停的最优流量
    stop_trf = []
    for i in range(len(stop_num)):
        if stop_num[i] == max_stop:
            max_stop_index.append(
                (i, stop_list[i], total_trf[i], dids_list[i]))
            stop_trf.append(total_trf[i])

    # index写的youdian
    a_b = stop_trf.index(max(stop_trf))
    index_s_max_t_max = max_stop_index[a_b][0]
    # index_s_max_t_max = total_trf.index(max(stop_trf))

    # a1 a3
    # 最大max_trf的第一个
    max_trf = max(total_trf)
    index = total_trf.index(max_trf)
    max_trffic_array = trf_list[index]

    trf_stop_list = []
    trf_stop_all_list = []
    for i in range(len(total_trf)):
        if total_trf[i] == max_trf:
            trf_stop_all_list.append(
                (i, stop_list[i], total_trf[i], dids_list[i]))
            trf_stop_list.append(stop_num[i])

    a_c = trf_stop_list.index(max(trf_stop_list))
    index_t_m_s_m = trf_stop_all_list[a_c][0]
    # index_t_m_s_m = stop_num.index(max(trf_stop_list))

    ini_trf = sum(env.ini_trf)

    print('1 : 初始流量分布为: 服务个数')
    print(env.ini_trf)
    print(sum(sum(env.mac_sev)))
    print("\n")

    print('2 : 调整完后的最好流量分布第一个: 序号+流量分布+关停分布+步数')
    print(index)
    print("\n")
    print(max_trffic_array)
    print("\n")
    print(stop_list[index])
    print("\n")
    print(dids_list[index])
    print('\n')

    print("3 : 调整后最好流量 / 初始流量 = 增大倍数:")
    print("%f / %f = %f" % (max_trf, ini_trf, max_trf / ini_trf))
    print("\n")

    print("4 : 调整完后的最好关停分布第一个: 序号+流量分布+关停分布+did")
    ####
    print(index_s)
    print("\n")
    print(trf_list[index_s])
    print("\n")
    print(max_stop_list)
    print("\n")
    print(dids_list[index_s])
    print('\n')

    print("5 : 最好关停里的最好流量 序号 流量分布 流量 流量增长率 关停分布 关机个数 步数 对没对 : 关机个数 流量分布 cpu mem")
    print(index_s_max_t_max)
    print("\n")
    print(trf_list[index_s_max_t_max])
    print('\n')
    print(total_trf[index_s_max_t_max] )
    print("\n")
    print(total_trf[index_s_max_t_max] / sum(env.ini_trf))
    print('\n')
    print(stop_list[index_s_max_t_max])
    print("\n")
    print(stop_num[index_s_max_t_max])
    print('\n')
    print(dids_list[index_s_max_t_max])
    print("\n")
    print(sum(stop_list[index_s_max_t_max]) == stop_num[index_s_max_t_max])
    print(sum(trf_list[index_s_max_t_max]) == total_trf[index_s_max_t_max] )
    print(sum(cpus[index_s_max_t_max]) == sum(env.init_cpus))
    print(sum(mems[index_s_max_t_max]) == sum(env.init_mems))
    print('\n')

    print("6 : 最好流量里的最好关停 序号 流量分布 流量 流量增长率 关停分布 关机个数 步数 对没对 : 关机个数 流量分布 cpu mem")
    print(index_t_m_s_m)
    print("\n")
    print(trf_list[index_t_m_s_m])
    print("\n")
    print(total_trf[index_t_m_s_m])
    print('\n')
    print(total_trf[index_t_m_s_m] / sum(env.ini_trf))
    print('\n')
    print(stop_list[index_t_m_s_m])
    print("\n")
    print(stop_num[index_t_m_s_m])
    print('\n')
    print(dids_list[index_t_m_s_m])
    print("\n")
    print(sum(stop_list[index_t_m_s_m]) == stop_num[index_t_m_s_m])
    print(sum(trf_list[index_t_m_s_m]) == total_trf[index_t_m_s_m])
    print(sum(cpus[index_t_m_s_m]) == sum(env.init_cpus))
    print(sum(mems[index_t_m_s_m]) == sum(env.init_mems))
    print("\n")

    ####

    # BT_under_BS_a1_5_50_1_r1+r2/ r1+r2
    # 5 best S
    #
    np.savetxt('BT_under_BS_a' + str(a_num) + '_' + str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank) + '_' + reward_kind + '.csv',
               mac_sev_list[index_s_max_t_max], delimiter=',')
    np.savetxt('BS_under_BT_a' + str(a_num) + '_' + str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank) + '_' + reward_kind + '.csv',
               mac_sev_list[index_t_m_s_m], delimiter=',')

    print("#### : 以下为流量+奖励变化序列：")
    print(total_trf)
    print('\n')
    print(list(np.array(return_list)))
    print('\n')

    # 计算一下平均利用率
    # cpus env.cpus
    pl = []
    pindex = []
    for i, cpu in enumerate(cpus):
        cpu = cpu * (1 - stop_list[i])  # [13, 12, 0, 0, 0]
        # 1 2 3 4 0
        pcpu_list = []
        for j, ci in enumerate(cpu):
            if (ci != 0):
                pcpu_list.append(ci / env.hcpus[j])
        pl.append(sum(pcpu_list) / len(pcpu_list))
        pindex.append(j)
        print("%d + %d = %d" %
              (len(pcpu_list), sum(stop_list[i]), env.mac_sev.shape[0]))
        if (len(pcpu_list) + sum(stop_list[i]) == env.mac_sev.shape[0]):
            print('True')
        else:
            print("False")
    print('CPU 平均利用率序列')
    print(pl)
    print('\n')
    print('CPU MAX')
    print(max(pl))
    print('\n')
    p_index = pl.index(max(pl))
    print("初始CPU")
    print(np.mean(env.init_cpus / env.hcpus))
    print('******************')
    ml = []
    mindex = []
    for i, mem in enumerate(mems):
        mem = mem * (1 - stop_list[i])
        # 1 2 3 4 0
        pmem_list = []
        for j, mi in enumerate(mem):
            if (mi != 0):
                pmem_list.append(mi / env.hmems[j])
        ml.append(sum(pmem_list) / len(pmem_list))
        mindex.append(i)
    print('MEM 平均利用率序列')
    print(ml)
    print('\n')
    print('MEM MAX')
    print(max(ml))
    print('\n')
    m_index = ml.index(max(ml))
    print('初始MEM')
    print(np.mean(env.init_mems / env.hmems))
    print('\n')

    print('CPU MEM T_under_S S_under_T ')
    print((pl[index_s_max_t_max], ml[index_s_max_t_max]))
    print('\n')
    print((pl[index_t_m_s_m], ml[index_t_m_s_m]))
    print('\n')

    print('最大CPU**************************************')
    print(p_index)
    print('\n')
    print(cpus[p_index])
    print('对不对')
    print(sum(cpus[p_index]) == sum(env.init_cpus))
    print('\n')
    print(mems[p_index])
    print('对不对')
    print(sum(mems[p_index]) == sum(env.init_mems))
    print('\n')
    print('最大MEM**************************************')
    print(m_index)
    print('\n')
    print(cpus[m_index])
    print('对不对')
    print(sum(cpus[m_index]) == sum(env.init_cpus))
    print('\n')
    print(mems[m_index])
    print('对不对')
    print(sum(mems[m_index]) == sum(env.init_mems))
    print('\n')

    print("最好流量 / 最好关停里的最优流量")
    print("%f / %f = %f" %
          (max_trf, total_trf[index_s_max_t_max], max_trf / total_trf[index_s_max_t_max]))
    print("the cpus")
    for i, c in enumerate(cpus):
        print(i)
        print(c)
        print("\n")
