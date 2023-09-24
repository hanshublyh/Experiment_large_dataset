import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections
import copy
import matplotlib.pyplot as plt
import os
import time
from src import Envn
from modules import ActorCritic, CNNandDense, rl_utils
from tqdm import tqdm
# 5_50/1/5_50_No/r1/reward,a1

ll_path = "../../"
rank_name = ll_path + 'rank.txt'

rank_array = np.loadtxt(rank_name, dtype=np.float32, delimiter=',')
rank = int(rank_array[2])

reward_kind_name = 'reward.txt'
with open(reward_kind_name, 'r') as f:
    reward_kind = f.read()

filename = ll_path + 'data' + str(rank) + '/'
dot = '.npy'
mac_kind = int(rank_array[0])
sev_kind = int(rank_array[1])

mac_sev_name = 'mac_sev' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
mac_trf_name = 'mac_trf' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
sev_trf_name = 'sev_trf' + '_' + str(sev_kind) + '_' + str(rank)  # 后面可以补充字符串说明
sev_trf_out_name = 'sev_trf_out' + '_' + \
    str(sev_kind) + '_' + str(rank)  # 后面可以补充字符串说明
sev_trf_in_name = 'sev_trf_in' + '_' + \
    str(sev_kind) + '_' + str(rank)  # 后面可以补充字符串说明
sev_cpu_name = 'sev_cpu' + '_' + str(sev_kind) + '_' + str(rank)
sev_mem_name = 'sev_mem' + '_' + str(sev_kind) + '_' + str(rank)
mac_cpu_name = 'mac_cpu' + '_' + str(mac_kind) + '_' + str(rank)
mac_mem_name = 'mac_mem' + '_' + str(mac_kind) + '_' + str(rank)
total_cpu_name = 'total_cpu' + '_' + str(mac_kind) + '_' + str(rank)
total_mem_name = 'total_mem' + '_' + str(mac_kind) + '_' + str(rank)
mac_sev_cpu_name = 'mac_sev_cpu' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
mac_sev_mem_name = 'mac_sev_mem' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
now_all_name = 'now_all' + '_' + str(mac_kind) + '_' + str(rank)

trf_cpu_mem = np.load(filename + now_all_name + dot)
mac_sev_cpu = np.load(filename + mac_sev_cpu_name + dot)
mac_sev_mem = np.load(filename + mac_sev_mem_name + dot)
mac_sev = np.load(filename + mac_sev_name + dot)
sev_trf = np.load(filename + sev_trf_name + dot)
sev_trf_out = np.load(filename + sev_trf_out_name + dot)
sev_trf_in = np.load(filename + sev_trf_in_name + dot)

s_cpu = np.load(filename + sev_cpu_name + dot)
s_mem = np.load(filename + sev_mem_name + dot)
stop = np.zeros(mac_sev.shape[0]).reshape((mac_sev.shape[0], 1))
trf_cpu_mem = np.concatenate((trf_cpu_mem, stop), axis=1)

topk = 0  # np.loadtxt('line2.csv', delimiter=",")


print(trf_cpu_mem.shape)
print(mac_sev_cpu.shape)
print(mac_sev_mem.shape)
print(mac_sev.shape)
print(sev_trf.shape)
print(sev_trf_out.shape)
print(sev_trf_in.shape)
print(s_cpu.shape)
print(s_mem.shape)

# 参数列表 这里也应该读取文件
para_list = ['actor_lr : ', 'critic_lr : ', 'num_episodes : ', 'gamma : ', 'lmbda : ',
             'epochs : ', 'eps : ', 'deep : ', 'lenth : ', 'stop_r : ', 'sa : ', 'sb : ',
             'epoch_len : ', 'pround : ', 'step_r : ', 'traffic_r : ', 'bs : ']
para_path = '../../../'
with open(para_path + 'para1.txt', 'r', encoding='utf-8') as f:
    para = f.readlines()
    # print(para)
for i, p in enumerate(para):
    para[i] = eval(p.strip(para_list[i] + ',\n'))
    print((para_list[i], para[i]))

actor_lr = para[0]
critic_lr = para[1]
num_episodes = int(para[2])
gamma = para[3]
lmbda = para[4]
epochs = int(para[5])
eps = para[6]
deep = int(para[7])
lenth = int(para[8])
stop_r = para[9]
# reshape
s_l = int(para[10])
s_h = int(para[11])

step_r = para[14]
traffic_r = para[15]
bs = para[16]

with open(para_path + '../'+ 'device.txt', 'r', encoding='utf-8') as f:
    device = torch.device(f.readlines()[0])
env_name = "VM——Reschedule"
env = Envn.Env(
    mac_sev, mac_sev_cpu, mac_sev_mem, trf_cpu_mem, sev_trf, sev_trf_out,
    sev_trf_in, reward_kind, s_cpu, s_mem, deep, lenth, topk, s_l, s_h, traffic_r)
# 随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

model = 'Actor-Critic'
act_dim = mac_sev.shape[0]

agent = ActorCritic.ActorCritic(act_dim, actor_lr, critic_lr,
                                gamma, lmbda, epochs, eps, device, model, num_episodes, deep, lenth, bs)


all_done = False
return_list = []
dids_list = []
trf_list = []
cpus = []
mems = []
stop_list = []
mac_sev_list = []

epoch_len = int(para[12])
per_len = int(num_episodes / epoch_len)
round = int(sum(sum(mac_sev)) / para[13])

print('num_episodes, actor_lr, critic_lr, gamma, lmbda, epochs, deep, lenth, round, reward, epoch_len, stop_r, eps, sl, sh step_r, traffic_r bs')
print(num_episodes, actor_lr, critic_lr,
      gamma, lmbda, epochs, deep, lenth, (para[13], round), reward_kind, epoch_len, stop_r, eps, s_l, s_h, step_r, traffic_r, bs)
cnt = 0
for i in range(epoch_len):  # 分几个进度条显示
    #with tqdm(total=per_len, desc='Iteration %d' % i) as pbar:
        for i_episode in range(per_len):
            
            start = time.perf_counter()
            # episode_return = 0
            transition_dict = {
                'states': [],
                'act_mac2s': [],
                'act_mac1s': [],
                'next_states': [],
                'rewards': [],
                'dids': [],
                'dones': [],
                'mac2': []
            }
            state = env.reset()
            # print(state.shape[1])
            done = False
            # print(env.ini_trf.reshape(10))
            while not done:
                env.the_now_mac_sev_pos_1agent()
                # print(state.shape[0])
                mask = env.mac_2_mask()
                action = agent.take_action(
                    state, mask, True, env)

                next_state, reward, did, done = env.step_1a(action)

                transition_dict['states'].append(np.array(state))
                transition_dict['act_mac2s'].append(action)
                transition_dict['act_mac1s'].append(env.m)
                transition_dict['dids'].append(did)
                transition_dict['next_states'].append(np.array(next_state))
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['mac2'].append(mask)
                state = next_state
                # episode_return += reward
            # print(transition_dict['dids'])

            #########################################
            # print(mac_sev.sum())
            # 测试本次调度对不对 主要是用mac_sev的值和s_cpu s_mem
            print('步数对不对 dids states next')
            print(len(transition_dict['dids']) == sum(sum(mac_sev)))
            print(len(transition_dict['states']) == sum(sum(mac_sev)))
            print(len(transition_dict['next_states']) == sum(sum(mac_sev)))
            print('符不符合实际 mac_sev没有为负数的 且服务种类还是一样')
            print(np.allclose(env.mac_sev.sum(axis=0), mac_sev.sum(axis=0)))
            its_a_test = 0
            for a in range(env.mac_sev.shape[0]):
                for b in range(env.mac_sev.shape[1]):
                    if env.mac_sev[a][b] < 0:
                        its_a_test = 1
                        print('有错误')
            if (its_a_test == 0):
                print('没错误')

            print('TEST: CPU MEM TRF')
            # cpu对不对
            mac1 = np.zeros((mac_sev.shape[0]))
            for a in range(mac_sev.shape[0]):
                for b in range(mac_sev.shape[1]):
                    mac1[a] += env.mac_sev[a][b] * s_cpu[b]
            print(np.allclose(mac1, env.cpus))
            # mem对不对
            mac2 = np.zeros((mac_sev.shape[0]))
            for a in range(mac_sev.shape[0]):
                for b in range(mac_sev.shape[1]):
                    mac2[a] += env.mac_sev[a][b] * s_mem[b]
            print(np.allclose(mac2, env.mems))
            # trf对不对
            mact = np.zeros((mac_sev.shape[0]))
            for a in range(env.mac_sev.shape[0]):
                for b in range(env.mac_sev.shape[1] - 1):
                    for c in range(b + 1, env.mac_sev.shape[1]):
                        mact[a] += min(env.sev_trf_out[b][c] * env.mac_sev[a]
                                       [b], env.sev_trf_in[b][c] * env.mac_sev[a][c])
            print(np.allclose(mact, env.now_trf))
            ############################################

            # print(episode_return)  # 当前奖励
            stop_list.append(env.stop)
            cpus.append(env.cpus)
            mems.append(env.mems)
            # return_list.append(sum(transition_dict['rewards']))
            dids_list.append(sum(transition_dict['dids']))
            trf_list.append(env.now_trf)
            the_computed_reward = agent.update(transition_dict, 'where', sum(
                stop_list[cnt]), stop_r, reward_kind, step_r, env.stop)
            mac_sev_list.append(env.mac_sev)
            print(sum(env.now_trf) / sum(env.ini_trf))
            # for p in agent.act_opt.param_groups:
            #    print(p['lr'])
            # 实际的reward改了 要返回一下

            #############
            return_list.append(sum(the_computed_reward))
            cnt += 1
            '''
            if (i_episode + 1) % per_len == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (per_len * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-per_len:])
                })
            pbar.update(1)
            # 那样可以多次移动
            '''
            end = time.perf_counter()
            # 计算运行时间
            runTime = end - start
            f = open('runtime1.txt', 'a')
            f.writelines('\n' + str((i * per_len + i_episode + 1, runTime)) + '\n')
            f.close()

print(dids_list)
rl_utils.show(env, stop_list, trf_list, dids_list, mac_kind, sev_kind,
              mac_sev_list, cpus, mems, return_list, rank, reward_kind, 1)
