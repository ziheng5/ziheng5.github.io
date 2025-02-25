---
title: 基于强化学习的多无人机协同围捕算法——MADDPG Multi-UAV Roundup
date: 2025-02-04 14:18:08
tags:
    - 强化学习
categories: 
    - 强化学习
description: |
    基于强化学习的多无人及协同围捕算法——MADDPG 的理论与仿真代码解读。
---

# 1. 仿真环境创建（基于 gymnasium）
```Python
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as transforms
import matplotlib.image as mping
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=4):
        self.length = length    # 边界长度
        self.num_obstacle = num_obstacle    # 障碍物数量
        self.num_agents = num_agents    # 智能体数量
        self.time_step = 0.5    # 每隔 0.5 步对参数进行一次更新
        self.v_max = 0.1    # agents 最大速度
        self.v_max_e = 0.12 # target 最大速度
        self.a_max = 0.04
        self.a_max_e = 0.05
        self.L_sensor = 0.2
        self.num_lasers = 16    # 激光数量
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0', 'agent_1', 'agent_2', 'target']
        self.info = np.random.get_state()   # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]

        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }

        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,)),
        }

    def reset(self):
        SEED = random.randint(1, 1000)
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1:
                # for agents
                self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            else:
                # for targets
                self.multi_current_pos.append(np.array([0.5, 1.75]))
            self.multi_current_vel.append(np.zeros(2))  # 初始化速度

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        
        return multi_obs

    def step(self, actions):
        last_d2target = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                # for agents
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))

            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 第 i 个 agent 坐标更新
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # 更新障碍物坐标
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # 检查是否碰撞到边界，并调整速度
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones = self.cal_rewards_dones(Collided, last_d2target)
        multi_next_obs = self.get_multi_obs()

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = []  # dim 3 only for target

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]   # dim 4
            S_team = [] # dim 3 for 2 agents 1 target
            S_target = []   # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:
                    # other agents
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length, pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    # target
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)    # 到 target 的距离
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        # for current agent
                        S_evade_d.append(d/np.linalg.norm(2*self.length))
            
            S_obser = self.multi_current_lasers[i]  # dim 16

            if i != self.num_agents - 1:
                # for agents
                single_obs = [S_uavi, S_team, S_obser, S_target]
            else:
                # for target
                single_obs = [S_uavi, S_obser, S_evade_d]

            _single_obs = list(itertools.chain(*))
            total_obs.append(_single_obs)
        
        return total_obs

    def cal_rewards_dones(self, IsCollied, last_d):
        dones = [False] * self.num_agents   # dim 4
        rewards = np.zeros(self.num_agents) # dim 4
        mu1 = 0.7   # r_near
        mu2 = 0.4   # r_safe
        mu3 = 0.01  # r_multi_stage
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target - self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec)

            cos_v_d = np.dot(vel, dire_vec) / (v_i * d + 1e-3)
            r_near = abs(2 * v_i / self.v_max) * cos_v_d

            rewards[i] += mu1 * r_near  # TODO: if not get nearer then receive negative reward

        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0, p1, pe)
        S1 = cal_triangle_S(p1, p2, pe)
        S1 = cal_triangle_S(p2, p0, pe)
        S1 = cal_triangle_S(p0, p1, p2)
        d1 = np.linalg.norm(p0 - pe)
        d2 = np.linalg.norm(p1 - pe)
        d3 = np.linalg.norm(p2 - pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d), -2, 2)

        # 3.2 stage-1 track
        if Sum_s > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d / max([d1, d2, d3])
            rewards[0:2] += mu3 * r_track
        # 3.3 stage-2 track
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3 * r_encircle
        # 3.4 stage-3 track
        elif Sum_s == S4 and any(Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3 * self.v_max))
            rewards[0:2] += mu3 * r_capture

        ## 4. finish rewards
        if Sum_S == S4 and all (d <= d_capture for d in [d1, d2, d3]):
            rewards[0:2] += mu4 *10
            dones = [True] * self.num_agents
        
        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in raneg(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones
    
    def render(self):
        plt.clf()

        # load UAV icon
        uav_icon = mping.imread('UAV.png')

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):

