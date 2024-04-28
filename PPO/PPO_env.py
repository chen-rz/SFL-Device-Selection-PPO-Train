import numpy as np
import random
import math
from os import path


class DeviceSelectionEnv:
    def __init__(self, device_num: int):
        # 环境属性
        self.device_param = get_device_param(device_type)
        self.dnn_param = get_dnn_param(dnn_type)
        self.edge_capacity = edge_capacity
        # self.edge_resources = edge_resources
        self.num_device = len(device_type)
        self.num_dnn_layers = len(self.dnn_param[0]) - 1
        self.bandwidth = bandwidth  #/self.num_device
        self.delay = delay
        self.worst_state = self._get_worst_state()
        # 动作空间&状态空间
        self.action_range = [1, self.num_dnn_layers]
        self.action_space = np.zeros((2, self.num_device))
        self.observation_space = np.zeros((6, self.num_device))
        # 随机化种子
        self.seed()

    # upper bound of average latency
    def _get_worst_state(self):
        # slice_point = 0
        binary_offload = []
        a1 = [[0.33, 0.33, 0.33], [0, 0, 0]]
        a2 = [[0.0, 0.0, 0.0], [self.num_dnn_layers, self.num_dnn_layers, self.num_dnn_layers]]
        u1 = np.vstack((a1[0], a1[1]))
        u2 = np.vstack((a2[0], a2[1]))
        for u in [u1, u2]:
            avg_latency = sum(self._get_cost(u))/self.num_device
            binary_offload.append(avg_latency)
        worst_latency = min(binary_offload)
        return worst_latency
    
    def _get_cost(self, action, is_step=True):
        latency = []
        latency_norm = []
        local_latency = []
        trans_latency = []
        edge_latency = []
        # print(action)
        for col in range(action.shape[1]):
            t_local = 0
            t_edge = 0
            rs_alloc = action[0, col]
            dnn_slc = round(action[1, col])
            dnn_slc = dnn_slc.astype(np.int32)
            exit_slt_bin = self._get_optimal_exits(dnn_slc)
            exit_prob, exit_slt_bin = self._get_exit_prob(exit_slt_bin)
            miu_val = [1 - p for p in exit_prob]
            miu_val.append(0)
            # local computing latency
            for k in range(dnn_slc+1):
                t_local += miu_val[k] * (self.dnn_param[0][k] + self.dnn_param[1][k] * exit_slt_bin[k])
            t_local /= self.device_param[col]
            # transmission latency
            t_trans = miu_val[dnn_slc+1] * (self.dnn_param[3][dnn_slc] / self.bandwidth + self.delay)
            # edge computing latency
            for k in range(dnn_slc+1, self.num_dnn_layers+1):
                t_edge += miu_val[k] * (self.dnn_param[0][k] + self.dnn_param[1][k] * exit_slt_bin[k])
            if rs_alloc <= 0.01:
                rs_alloc = 0.01
            t_edge /= (rs_alloc * self.edge_capacity)
            # print('device_trans_edge: {:4f}ms, {:4f}ms, {:4f}ms'.format(t_local, t_trans, t_edge))
            T = (t_local + t_trans + t_edge)*1e3
            latency.append(T)  # 毫秒 ms
            latency_norm.append(T/31.148)
            local_latency.append(t_local*1e3/T)
            trans_latency.append(t_trans*1e3/T)
            edge_latency.append(t_edge*1e3/T)
        if is_step:
            return latency
        else:
            return latency_norm, local_latency, trans_latency, edge_latency

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # u is action, size=(3, num_device)
    def step(self, u, is_train=False):
        # previous latency
        pre_u = np.vstack((self.state[0], self.state[1]))
        pre_latency = sum(self._get_cost(pre_u))/self.num_device
        # ----------v1~v5非线性变换--------
        # u = np.abs(u)
        # ----------v6线性变换-----------
        u = (u + 1) / 2
        rs_alloc = u[0]
        dnn_slc = u[1] * self.action_range[1]
        dnn_slc = np.clip(dnn_slc, 0, self.action_range[1])
        dnn_slc = dnn_slc.astype(np.int32)
        for k in range(3):
            if dnn_slc[k] == self.action_range[1]:
                rs_alloc[k] = 0
        if np.sum(rs_alloc) > 0:
            rs_alloc = np.clip(rs_alloc/np.sum(rs_alloc), 0, 1)
        # rs_alloc = np.clip(rs_alloc, 0, 1)
        # dnn_slc = np.clip(dnn_slc * self.action_range[1], 0, self.num_dnn_layers)
        self.last_u = np.vstack((rs_alloc, dnn_slc))
        # the latency here is a list of each device
        latency = self._get_cost(self.last_u)
        goal = sum(latency)/self.num_device
        done = False
        # -----------v3------------
        # if goal < self.worst_state:
        #     costs = -math.exp(self.worst_state - goal)
        # else:
        #     costs = goal
        #------------v4------------
        # costs = 0
        # if goal < pre_latency:
        #     costs -= 5
        # else:
        #     costs += 5
        #----------- v5,v6------------
        costs = 0
        if goal < pre_latency:
            costs -= 1.0
        elif goal > pre_latency:
            costs += 1.5
        else:
            pass

        new_dnn_slc = dnn_slc
        new_rs_alloc = rs_alloc
        bw = np.full((1, self.num_device), self.bandwidth)
        dl = np.full((1, self.num_device), self.delay)
        dc = self.device_param

        self.state = np.vstack((new_rs_alloc, new_dnn_slc,
                                bw, dl, dc))
        # return state, reward, done, _
        if is_train:
            return self._get_obs(), -costs, goal, done, [rs_alloc, dnn_slc, bw, dl]
        else:
            exit_slt_bin = []
            for k in range(self.num_device):
                tmp = self._get_optimal_exits(dnn_slc[k])
                exit_slt_bin.append(tmp)
            return self._get_obs(), -costs, goal, done, [rs_alloc, dnn_slc, exit_slt_bin]

    def reset(self, network_init_=None, randomly_init_=True):
        s_width = self.num_device
        rs_alloc = np.full((1, s_width), 0.33)
        dnn_slc = np.full((1, s_width), 0)
        # rs_alloc = np.random.rand(1, s_width)
        # rs_alloc = np.clip(rs_alloc/np.sum(rs_alloc), 0, 1)
        # dnn_slc = np.array(np.random.randint(self.num_dnn_layers, size=(1, s_width)), dtype=np.int32)
        # exit_slt = self._get_optimal_exits(dnn_slc)
        bw_list = [i for i in range(1, 6)]
        dl_list = [i for i in range(1, 11)]
        if randomly_init_:
            # bw = np.random.random(1) * 20 * 1024
            # dl = np.random.random(1) / 5
            bw = np.array([random.choice(bw_list)]) * 1024
            dl = np.array([random.choice((dl_list))]) / 1000
            self.bandwidth = bw[0]
            self.delay = dl[0]
        else:
            bw = np.array([network_init_[0]]) # input bandwidth
            dl = np.array([network_init_[1]]) # input delay
            self.bandwidth = bw[0]
            self.delay = dl[0]
        # resize
        bw = np.full((1, s_width), bw)
        dl = np.full((1, s_width), dl)
        dc = self.device_param
        # bw = np.full((1, self.num_device), self.bandwidth)
        # dl = np.full((1, self.num_device), self.delay)

        # randomly reset state
        self.state = np.vstack((rs_alloc, dnn_slc,
                                bw, dl, dc))
        self.worst_state = self._get_worst_state()
        self.last_u = None
        return self._get_obs()

    # get observation
    def _get_obs(self):
        u = np.vstack((self.state[0], self.state[1]))
        latency, l_local, l_trans, l_edge = self._get_cost(u, is_step=False)
        obs = np.vstack((latency, l_local, l_trans, l_edge,
                         self.state[2]/20480, self.state[3]*1000/105))  # size=(6, NUM_DEVICE)
        return obs.reshape(-1)

    def render(self):
        return self.worst_state

