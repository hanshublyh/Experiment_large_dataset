import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import copy
import collections
from modules import CNNandDense, rl_utils
import gc


class ActorCritic:
    def __init__(self, act_dim, actor_lr, critic_lr, gamma, lmbda, epochs, eps, device, model, epo, deep, lent, bs):
        self.act_dim = act_dim
        self.deep = deep
        self.actor = CNNandDense.ActCNN(act_dim, deep, lent).to(device)
        self.critic = CNNandDense.ValueCNN(1, deep, lent).to(device)
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.cirloss = nn.MSELoss()
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        # self.total_opt = optim.Adam(self.ac.parameters(), lr = self.actor_lr)
        self.act_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.cri_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.device = device
        self.model = model
        self.now_epo = 0
        self.total_epo = epo
        # self.topk = topk
        self.bs = bs
    def update_lr(self):
        self.now_epo += 1

        for param_group in self.act_opt.param_groups:
            # print(param_group['lr'])
            # if self.now_epo <= 9:
            param_group['lr'] = self.actor_lr * \
                (1 - self.now_epo / self.total_epo)
            # else:
            #    param_group['lr'] = 1e-3 * (1 - 9 / 10) / 1000
        for param_group in self.cri_opt.param_groups:
            # if self.now_epo <= 9:
            param_group['lr'] = self.critic_lr * \
                (1 - self.now_epo / self.total_epo)
            # else:
            #    param_group['lr'] = 1e-3 * (1 - 9 / 10) / 1000

    def take_action(self, state, mask, has_mask, env):
        gc.collect()
        torch.cuda.empty_cache()
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = torch.unsqueeze(state, 0).to(self.device)
        # 给出概率 按概率抽样 返回下标

        # 动作掩码
        # print(mask)
        '''
        test = True
        for i in self.actor.parameters():
            test *= i.isnan().all()
        print('这是当前的actor的参数 有没有nan\n')
        print(test)
        print('关停\n')
        print(env.stop)
        print('mask\n')
        print(env.mac_mask)
        '''
        # [0, 0 , 0, 0, 0]
        # (- 无穷)
        if has_mask == True:
            # 选mac1 sev 不可能出现全mask
            prob = self.actor(state) * torch.tensor(mask).to(self.device)
            if sum(mask) == 0:  # 就是else 按理说这个就没法迁移
                prob = self.actor(state)
        else:
            # 选mac2 sev-->mac2 防止出现prob为0的情况 因为全给mask了 这个就先不mask
            # 有没有全给mask的情况呢 有 选出来了 不迁移
            prob = self.actor(state)
        # 5 * 1
        # 到底要不要加mask

        # 最笨的方法 解决就是概率全是0 但mask不是0
        if (sum(sum(prob)) == 0):
            not_zero_index = [i for i in range(mask.shape[0]) if mask[i] != 0]
            not_zero_num = len(not_zero_index)
            for i in not_zero_index:
                prob[0][i] = 1.0 / not_zero_num

        '''
        print("纯这个state概率\n")
        print(self.actor(state))
        print(mask)
        print(env.stop)
        print("Prob")
        print(prob.shape)
        print(prob)
        print('(((((((((((((((((((())))))))))))))))))))')
        '''

        # print(prob)
        # prob, ntr = self.ac(state)
        # look back 推荐算法 规则发掘 专家知识

        # look ahead 蒙特卡洛树搜索
        distribution = torch.distributions.Categorical(prob)
        act = distribution.sample().item()
        # print(act)
        return act

    def update(self, transitions, agent_name, stop, per_reward, kinds, step_r, stop_list):
        if (agent_name == 'which'):
            actions_name = 'act_sevs'  # sev
        elif (agent_name == 'mac'):
            actions_name = 'act_mac1s'  # mac1
        elif (agent_name == 'where'):
            actions_name = 'act_mac2s'  # mac2

        # 做了动作的给奖励 迁移奖励
        # have action did != 0所有的都给奖励 关停奖励
        # 到底sigmoid 0不0 流量奖励
        # 注意奖励的给
        states = np.array(transitions['states'])
        actions = transitions[actions_name]
        dids = transitions['dids']
        next_states = np.array(transitions['next_states'])
        dones = transitions['dones']
        rewards = transitions['rewards']
        the_actions1 = transitions['act_mac1s']
        the_actions3 = transitions['act_mac2s']

        reward_st = stop * per_reward
        if kinds == 'r3':  # r3
            index_list = []
            for i in range(stop_list.shape[0]):
                if stop_list[i] == 1:
                    index_list.append(i)

            for i in range(len(rewards)):
                if dids[i] == 1 and (the_actions1[i] in index_list) and (the_actions3[i] not in index_list):
                    rewards[i] += reward_st

            for i in range(len(rewards)):
                if dids[i] == 1:
                    rewards[i] -= step_r

        elif kinds == 'r1 + r2':  # r1 + r2
            index_list = []
            for i in range(stop_list.shape[0]):
                if stop_list[i] == 1:
                    index_list.append(i)

            for i in range(len(rewards)):
                if dids[i] == 1 and (the_actions1[i] in index_list) and (the_actions3[i] not in index_list):
                    rewards[i] += reward_st

        
# 更新的时候到底用什么？是用所有有用步数？还是肺部？
        if self.model == 'Actor-Critic':

            BATH_SIZE = self.bs
            lenth = states.shape[0]
            if lenth % BATH_SIZE == 0:
                rnds = int(lenth / BATH_SIZE)
            else:
                rnds = int(lenth / BATH_SIZE) + 1

            for rnd in range(rnds):
                gc.collect()
                torch.cuda.empty_cache()
                rnd_states = torch.tensor(
                    states[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).to(self.device)
                rnd_next_states = torch.tensor(next_states[rnd * BATH_SIZE: min(
                    (rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).to(self.device)
                rnd_act = torch.tensor(
                    actions[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)
                # rnd_mask = torch.tesnsor(masks[rnd * BATH_SIZE : min( (rnd + 1) * BATH_SIZE, lenth)]).to(self.device)
                rnd_rewards = torch.tensor(rewards[rnd * BATH_SIZE: min(
                    (rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).view(-1, 1).to(self.device)
                # rnd_dids = torch.tensor(dids[rnd * BATH_SIZE : min( (rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)
                rnd_dones = torch.tensor(
                    dones[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)

                q_values = self.critic(rnd_states)
                q_target = rnd_rewards + self.gamma * \
                    self.critic(rnd_next_states) * ~rnd_dones
                delta = q_target - q_values

                # probs = (self.actor(rnd_states) * masks).gather(1, rnd_act)
                probs = (self.actor(rnd_states)).gather(1, rnd_act)
                log_probs = torch.log(probs + 1e-10)

                actor_loss = torch.mean(-log_probs * delta.detach())
                # 均方误差损失函数
                critic_loss = torch.mean(
                    F.mse_loss(q_values, q_target.detach()))
                '''
                print('这是看loss有无nan 已经加了梯度裁剪')
                print(actor_loss)
                print(critic_loss)
                '''

                self.act_opt.zero_grad()
                self.cri_opt.zero_grad()

                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 5, norm_type=2)  # 计算策略网络的梯度
                critic_loss.backward()  # 计算价值网络的梯度
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 5, norm_type=2)
                self.act_opt.step()  # 更新策略网络的参数
                self.cri_opt.step()  # 更新价值网络的参数

            self.update_lr()

        elif self.model == 'PPO':
            BATH_SIZE = self.bs
            lenth = states.shape[0]
            rnds = int(lenth / BATH_SIZE) + 1

            for rnd in range(rnds):
                gc.collect()
                torch.cuda.empty_cache()
                rnd_states = torch.tensor(
                    states[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).to(self.device)
                rnd_next_states = torch.tensor(next_states[rnd * BATH_SIZE: min(
                    (rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).to(self.device)
                rnd_act = torch.tensor(
                    actions[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)
                rnd_rewards = torch.tensor(rewards[rnd * BATH_SIZE: min(
                    (rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).view(-1, 1).to(self.device)
                # rnd_dids = torch.tensor(dids[rnd * BATH_SIZE : min( (rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)
                rnd_dones = torch.tensor(
                    dones[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)

                q_values = self.critic(rnd_states)
                q_target = rnd_rewards + self.gamma * \
                    self.critic(rnd_next_states) * ~rnd_dones
                delta = q_target - q_values

                advantage = rl_utils.compute_advantage(
                    self.gamma, self.lmbda, delta.cpu()).to(self.device)
                old_log_probs = torch.log(self.actor(
                    rnd_states).gather(1, rnd_act)).detach()

                for _ in range(self.epochs):
                    log_probs = torch.log(self.actor(
                        rnd_states).gather(1, rnd_act))
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps,
                                        1 + self.eps) * advantage  # 截断

                    # PPO损失函数
                    actor_loss = torch.mean(-torch.min(surr1, surr2))
                    critic_loss = torch.mean(F.mse_loss(
                        self.critic(rnd_states), q_target.detach()))

                    self.act_opt.zero_grad()
                    self.cri_opt.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    self.act_opt.step()
                    self.cri_opt.step()
            self.update_lr()
        return rewards
