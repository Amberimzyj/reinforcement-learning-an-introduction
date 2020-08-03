# Project: <<selftry_ten_armed>>
# File Created: Thursday, 30th July 2020 5:04:14 pm
# Author: <<Yajing Zhang>> (<<amberimzyj@qq.com>>)
# -----
# Last Modified: Thursday, 30th July 2020 10:12:46 pm
# Modified By: <<Yajing Zhang>> (<<amberimzyj@qq.com>>>)
# -----
# Copyright 2020 - <<2020>> <<Yajing Zhang>>, <<IWIN>>


import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

class zyj_Bandit:
    
    def __init__(self, k_arm = 10, 
                epsilon = 0, 
                step_size = 0.1, 
                true_reward = 0., 
                initial = 0., 
                sample_averages = False,
                gradient = False,
                gradient_baseline = False,
                UCB_param = None):
        self.k = k_arm
        self.epsilon = epsilon  #贪心试探度
        self.step_size = step_size  #固定步长
        self.true_reward = true_reward #Q真值
        self.initial = initial 
        self.time = 0
        self.indices = np.arange(self.k)
        self.UCB_param = UCB_param
        self.average_reward = 0
        self.sample_averages = sample_averages  #采样平均
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        


    
    #设置初始值
    def reset(self):
        #设置每个动作的真值相等（=0）
        self.q_true = np.random.randn(self.k) + self.true_reward

        #设置每个动作的估计值
        self.q_estimation = np.zeros(self.k) + self.initial

        #记录每个动作的选择次数
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0
        
    
    # get an action for this bandit
    def act(self):
        # self.q_true += np.random.normal(loc = 0, scale = 0.01, size = self.k) #非平稳过程

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices) 

        # if self.UCB_param is not None:
        #     UCB_estimation = self.q_estimation + \
        #         self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
        #     q_best = np.max(UCB_estimation)
        #     return np.random.choice(np.where(UCB_estimation == q_best)[0])

        # if self.gradient:
        #     exp_est = np.exp(self.q_estimation)
        #     self.action_prob = exp_est / np.sum(exp_est)
        #     return np.random.choice(self.indices, p=self.action_prob)


        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    



    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        self.q_true += np.random.normal(loc = 0, scale = 0.01, size = self.k) #非平稳过程
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time #更新平均reward(采样平均)

        if self.sample_averages:
            # update estimation using sample averages（增量法更新该动作的reward）
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        # elif self.gradient:
        #     one_hot = np.zeros(self.k)
        #     one_hot[action] = 1
        #     if self.gradient_baseline:
        #         baseline = self.average_reward
        #     else:
        #         baseline = 0
        #     self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        #固定步长（加权平均）
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

def exercise_2_5(runs = 2000, time = 1000):
    bandits = [zyj_Bandit(epsilon=0.1, sample_averages=True)]
    _ , rewards = simulate(runs, time, bandits)
    plt.figure(figsize=(10, 20))
    plt.subplot(2,1,1)
    plt.plot(rewards[0],label = 'sample average')
    bandits = [zyj_Bandit(epsilon=0.1, sample_averages=False)]
    _ , rewards = simulate(runs, time, bandits)
    plt.plot(rewards[0],label = 'weighted average')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()


    bandits = [zyj_Bandit(epsilon=0.1, sample_averages=True)]
    best_action_counts, _ = simulate(runs, time, bandits)
    plt.subplot(2,1,2)
    plt.plot(best_action_counts[0],label = 'sample average')
    bandits = [zyj_Bandit(epsilon=0.1, sample_averages=False)]
    best_action_counts, _ = simulate(runs, time, bandits)
    plt.plot(best_action_counts[0],label = 'weighted average')
    plt.xlabel('steps')
    plt.ylabel('optimal action')
    plt.legend()

    plt.show()



if __name__ == "__main__":
    exercise_2_5()
