import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import random
from scipy.linalg import hadamard, eig
import copy
import gc
import os
import time
from tqdm import tqdm
import json
import scipy.sparse.linalg as linalg
from scipy.sparse import csc_matrix
import copy
from numba import jit
import time
# 汉明距离转内积：<> = bit-2d
# 内积转汉明距离：d = 1/2(bit-<>)

def get_margin(bit, n_class):
    # 1. 计算d_max
    L = bit
    right = (2 ** L) / n_class
    # print(f"border is {right}")
    d_min = 0
    d_max = 0
    for j in range(2 * L + 4):
        dim = j
        sum_1 = 0
        sum_2 = 0
        for i in range((dim - 1) // 2 + 1):
            sum_1 += comb(L, i)
        for i in range((dim) // 2 + 1):
            sum_2 += comb(L, i)
        if sum_1 <= right and sum_2 > right:
            d_min = dim
    for i in range(2 * L + 4):
        dim = i
        sum_1 = 0
        sum_2 = 0
        for j in range(dim):
            sum_1 += comb(L, j)
        for j in range(dim - 1):
            sum_2 += comb(L, j)
        if sum_1 >= right and sum_2 < right:
            # `print(f"sum 1 is {sum_1}")
            # print(f"sum 2 is {sum_2}")`
            d_max = dim
            break
    # 2. 计算alpha_neg和alpha_pos
    alpha_neg = L - 2 * d_max
    # beta_neg = L - 2 * d_min
    alpha_pos = L
    # print(d_min)
    return d_max, d_min

def CSQ_init(n_class, bit):
    h_k = hadamard(bit)
    h_2k = np.concatenate((h_k, -h_k), 0)
    hash_center = h_2k[:n_class]

    if h_2k.shape[0] < n_class:
        hash_center = np.resize(hash_center, (n_class, bit))
        for k in range(5):
            for index in range(h_2k.shape[0], n_class):
                ones = np.ones(bit)
                ones[random.sample(list(range(bit)), bit // 2)] = -1
                hash_center[index] = ones
            c = []
            for i in range(n_class):
                for j in range(i, n_class):
                        c.append(sum(hash_center[i] != hash_center[j]))
            c = np.array(c)
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                break
    return hash_center

def init_hash(n_class, bit):
    hash_centers = -1 + 2 * np.random.random((n_class, bit))
    hash_centers = np.sign(hash_centers)
    return hash_centers

@jit(nopython=True)
def cal_Cx(x, H):
    return np.dot(H, x)

@jit(nopython=True)
def cal_M(H):
    return np.dot(H.T, H)/H.shape[0]

@jit(nopython=True)
def cal_b(H):
    """
    求H中所有哈希中心的均值
    """
    return np.dot(np.ones(H.shape[0], dtype=np.float64), H) / H.shape[0]

@jit(nopython=True)
def cal_one_hamm(b, H):
    temp = 0.5 * (b.shape[0] - np.dot(H, b))
    return temp.mean() + temp.min()-temp.var(), temp.min()

@jit(nopython=True)
def cal_hamm(H):
    dist = []
    for i in range(H.shape[0]):
        for j in range(i+1, H.shape[0]):
                TF = np.sum(H[i] != H[j])
                dist.append(TF)
    dist = np.array(dist)
    st = dist.mean() + dist.min() - dist.var()
    # print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}")
    return st, dist.mean(), dist.min(), dist.var(), dist.max()

@jit(nopython=True)
def in_range(z1, z2, z3, bit):
    """
    截断误差
    """
    flag = True
    for item in z1:
        if item < -1 and item > 1:
            flag = False
            return flag
    for item in z3:
        if item < 0:
            flag = False
            return flag
    res = 0
    for item in z2:
        res += item**2
    if abs(res - bit) > 0.001:
        flag = False
        return flag
    return flag

@jit(nopython=True)
def get_min(b, H):
    temp = []
    for i in range(H.shape[0]):
        TF = np.sum(b != H[i])
        temp.append(TF)
    temp = np.array(temp)
    # print(temp.min())
    return temp.min()

@jit(nopython=True)
def Lp_box_one(b, H, d_max, n_class, bit, rho, gamma, error):
    """
    要优化的哈希中心为x
    """
    b = b.astype(np.float64)
    H = H.astype(np.float64)
    # 计算下界
    d = bit - 2 * d_max

    # 先优化一个哈希中心
    M = cal_M(H) # M的维度是 n x n
    C = cal_b(H) # b的维度是 n x 1
    out_iter = 1000
    in_iter = 10
    upper_rho = 1e7
    learning_fact = 1.09
    count = 0
    best_eval, best_min = cal_one_hamm(np.sign(b), H)
    best_B = b
    # 每一次迭代的初始参数都是一样的
    #将辅助变量z1，z2初始化为输入哈希中心
    z1 = b.copy() 
    z2 = b.copy()
    # 初始化每个输入哈希中心与其他哈希中心的内积，这些内积离下界的差距，维度为 m-1
    z3 = d - cal_Cx(np.sign(b), H)
    #对拉格朗日乘子进行随机初始化
    y1 = np.random.rand(bit)
    y2 = np.random.rand(bit)
    y3 = np.random.rand(n_class-1)
    # 转换数据类型
    z1 = z1.astype(np.float64)
    z2 = z2.astype(np.float64)
    z3 = z3.astype(np.float64)
    y1 = y1.astype(np.float64)
    y2 = y2.astype(np.float64)
    y3 = y3.astype(np.float64)

    # 外层迭代
    for e in range(out_iter):
        for ei in range(in_iter):
            # 更新x
            left = ((rho+rho) * np.eye(bit, dtype=np.float64) + rho * np.dot(H.T, H))
            left = left.astype(np.float64)
            right = (rho * z1 + rho * z2 + rho * np.dot(H.T, (d - z3)) - y1 - y2 - np.dot(H.T, y3) - C)
            right = right.astype(np.float64)
            b = np.dot(np.linalg.inv(left), right)

            # 更新z1
            z1 = b + 1/rho * y1
            # 更新z2
            z2 = b + 1/rho * y2
            # 更新z3
            z3 = d - np.dot(H, b) - 1/rho * y3

            if in_range(z1, z2, z3, bit):
                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)
                break
            else:
                z1[z1 > 1] = 1
                z1[z1 < -1] = -1

                norm_x = np.linalg.norm(z2)
                z2 = np.sqrt(bit) * z2 / norm_x

                z3[z3 < 0] = 0

                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)
        # 更新rho
        rho = min(learning_fact * rho, upper_rho)
        if rho == upper_rho:
            count += 1
            eval, mini = cal_one_hamm(np.sign(b), H)
            if eval > best_eval or mini > best_min:
                best_eval = eval
                best_min = mini
                best_B = np.sign(b)
        if max(np.linalg.norm(b-z1, np.inf), max(np.linalg.norm(b-z2, np.inf), np.linalg.norm(np.dot(H, b) + z3 - d, np.inf))) < error:
            break
        if count == 100:
            # best_B = np.sign(b)
            break
    # best_B = np.sign(b)
    return best_B, H

@jit(nopython=True)
def Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, error, best_st):
    count = 0
    for oo in range(50):
        for i in range(n_class):
            H = np.vstack((B[:i], B[i+1:])) # m-1 x n
            B[i], _ = Lp_box_one(B[i], H, d_max, n_class, bit, rho, gamma, error)
        eval_st, eval_mean, eval_min, eval_var, eval_max = cal_hamm(B)
        print(eval_st, eval_min, eval_mean, eval_var, eval_max)
        if eval_st > best_st:
            best_st = eval_st
            best_B = B.copy()
            count = 0
        else:
            count += 1
        if eval_min >= d_max:
            break
    return best_B

if __name__ == '__main__':
    for bit in [32]:
        n_class = 196
        initWithCSQ = True
        if bit == 48:
            initWithCSQ = False
        d_max, d_min = get_margin(bit, n_class)
        d_max = d_max
        print(f"d_max is {d_max}, d_min is {d_min}")
        # 参数初始化
        rho = 1e-5
        gamma = (1+5**0.5)/2
        error = 1e-6
        # 初始化哈希中心
        random.seed(40)
        np.random.seed(40)
        d = bit - 2 * d_max
        if initWithCSQ:
            B = CSQ_init(n_class, bit) # initialize with CSQ
        else:
            B = init_hash(n_class, bit) # random initialization

        # 初始评价指标
        best_st, best_mean, best_min, best_var, best_max = cal_hamm(B)
        best_B = copy.deepcopy(B)
        count = 0
        error_index = {}
        print(f"best_st is {best_st}, best_min is {str(best_min)}, best_mean is {best_mean}, best_var is {best_var}, best_max is {str(best_max)}")
        best_st = 0
        print(f"eval st, eval min, eval mean, eval var, eval max")
        begin = time.time()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(begin))
        print(time_string)
        best_B = Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, error, best_st)
        end = time.time()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
        print(time_string)
        ev_st, ev_mean, ev_min, ev_var, ev_max = cal_hamm(best_B)
        print(f"ev_st is {ev_st}, ev_min is {str(ev_min)}, ev_mean is {ev_mean}, ev_var is {ev_var}, ev_max is {str(ev_max)}")
        np.save(f'./centerswithoutVar/CSQ_init_{initWithCSQ}_{n_class}_{bit}.npy', best_B)

    
            
