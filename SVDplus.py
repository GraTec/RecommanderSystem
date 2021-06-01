import numpy as np
import time
from math import sqrt


def sumy(y, implicit, user):  # 计算(Sum_j yj) 和 (Sum_j \|y_j\|^2)
    vec = np.zeros((1, len(y[0])))
    norm = 0
    for item in implicit[user]:
        vec = vec+y[item]
        norm += np.linalg.norm(y[item])**2
    return [vec, norm]


def computeSSE(m, p, q, lamb, b_user, b_item, sigma, y, implicit):
    [m_row, m_col, m_val] = m[0:3]
    length = len(m_row)
    SSE = 0
    for i in range(0, length):
        [vec, norm] = sumy(y, implicit, m_row[i])
        SSE += (m_val[i]-np.dot(p[m_row[i], :]+vec/max(sqrt(len(y[m_row[i]])), 1), q[:, m_col[i]])-sigma-b_user[m_row[i]]-b_item[m_col[i]]) ** 2+lamb * \
            (np.linalg.norm(p[m_row[i], :])**2 + np.linalg.norm(q[:, m_col[i]])
             ** 2+np.linalg.norm(b_user)**2+np.linalg.norm(b_item)**2 + norm)
    return SSE


def SVDplus(m, a, maxK, lamb, implicit):
    print('Initializing...')
    [m_row, m_col, m_val, row, col] = m
    col = max(col, (np.max(implicit)+1))
    length = len(m_row)
    step = 1
    p = np.random.rand(row, maxK)
    q = np.random.rand(maxK, col)
    y = np.random.rand(col, maxK)
    b_user, b_item = np.random.rand(row), np.random.rand(col)
    sigma = sum(m_val)/length
    # 初始化完毕
    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma, y, implicit)
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
            [vec, norm] = sumy(y, implicit, user)
            _sum = m_val[i]-np.dot(p[user, :]+vec/max(sqrt(len(y[user, :])), 1),
                                   q[:, item]) - sigma-b_user[user]-b_item[item]
            ###### 更新P,Q ######
            tmp = p[user, :]  # 保存前一个状态的P_U
            tmq = q[:, item]  # 保存前一个状态的Q_I
            p[user, :] = (1-a*lamb/RMSE/length)*p[user, :] + \
                a*_sum*q[:, item]/RMSE/length
            q[:, item] = (1-a*lamb/RMSE/length) * q[:, item] +\
                a*_sum*(tmp+vec/max(sqrt(len(y[user, :])), 1))/RMSE/length
            # 更新偏置 b_user 和 b_item ######
            b_user[user] = b_user[user]+a * \
                (_sum-lamb*b_user[user])/RMSE/length
            b_item[item] = b_item[item]+a * \
                (_sum-lamb*b_item[item])/RMSE/length
            # 更新 y_j ######
            if item in implicit[user]:
                y[item, :] = y[item, :]+a*(_sum*np.transpose(tmq)/max(
                    sqrt(len(y[user, :])), 1)-lamb*y[item, :])/RMSE/length
            # for k in range(0, maxK):
            #     tmp = p[user][k]
            #     p[user][k] = p[user][k]+a * \
            #         (_sum * q[k][item]-lamb*p[user][k])/RMSE
            #     q[k][item] = q[k][item]+a * \
            #         (_sum*(tmp+vec[0, k]/max(sqrt(len(implicit[user])), 1)
            #                ) - lamb*q[k][item])/RMSE
            # if item in implicit[user]:
            #     y[item, :] = y[item, :]+a * \
            #         (_sum*np.transpose(q_item) /
            #          max(sqrt(len(y[user, :])), 1)-lamb*y[item, :])/RMSE

        newSSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma, y, implicit)
        newRMSE = sqrt(newSSE/length)  # 计算新的误差
        if abs(newRMSE-RMSE) <= 10**-5:
            print('', end='\n')
            return [p, q, sigma, b_user, b_item, y]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q, sigma, b_user, b_item, y]
