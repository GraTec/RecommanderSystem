import numpy as np
import time
from math import sqrt


def computeSSE(m, p, q, lamb, b_user, b_item, sigma):
    [m_row, m_col, m_val] = m[0:3]
    length = len(m_row)
    SSE = 0
    for i in range(0, length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])-sigma-b_user[m_row[i]]-b_item[m_col[i]]) ** 2+lamb * \
            (np.linalg.norm(p[m_row[i], :])**2 + np.linalg.norm(q[:, m_col[i]])
             ** 2+np.linalg.norm(b_user)**2+np.linalg.norm(b_item)**2)
    return SSE


def biasSVD(m, a, maxK, lamb, eps):
    print('Initializing...')
    [m_row, m_col, m_val, row, col] = m
    length = len(m_row)
    step = 1
    p = np.random.rand(row, maxK)
    q = np.random.rand(maxK, col)
    b_user, b_item = np.random.rand(row), np.random.rand(col)
    sigma = sum(m_val)/length
    # 初始化完毕
    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma)
    RMSE = sqrt(SSE/length)
    # 计算初始误差完毕
    print('Initialization finished. RMSE=', RMSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'RMSE=', RMSE, end='\r')
        # time.sleep(0.5)
        b2_user = b_user
        b2_item = b_item
        for i in range(0, length):  # 梯度下降法，每次迭代遍历所有变量
            user = m_row[i]
            item = m_col[i]
            _sum = m_val[i]-np.dot(p[user, :], q[:, item]) - \
                sigma-b_user[user]-b_item[item]
            ###### 更新P,Q ######
            tmp = p[user, :]  # 保存前一个状态的P_U
            p[user, :] = (1-a*lamb/RMSE/length)*p[user, :] + \
                a*_sum*q[:, item]/RMSE/length
            q[:, item] = (1-a*lamb/RMSE/length) * \
                q[:, item]+a*_sum*tmp/RMSE/length
            # 更新偏置 b_user 和 b_item ######
            b_user[user] = b_user[user]+a * \
                (_sum-lamb*b_user[user])/RMSE/length
            b_item[item] = b_item[item]+a * \
                (_sum-lamb*b_item[item])/RMSE/length

        newSSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma)
        newRMSE = sqrt(newSSE/length)  # 计算新的误差
        if abs(newRMSE-RMSE) <= 10**-5:
            print('', end='\n')
            return [p, q, sigma, b_user, b_item]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q, sigma, b_user, b_item]
