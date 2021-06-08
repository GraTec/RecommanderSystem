import numpy as np
import time
from math import sqrt


def computeSSE(m, p, q, lamb, b_user, b_item, sigma, omega_u, omega_i, w_u, w_i):
    [m_row, m_col, m_val] = m[0:3]
    length = len(m_row)
    SSE = 0
    for i in range(0, length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :]+omega_u[m_row[i]]*w_u[m_row[i]], q[:, m_col[i]]+omega_i[m_col[i]]*w_i[m_col[i]])-sigma-b_user[m_row[i]]-b_item[m_col[i]]) ** 2+lamb * \
            (np.linalg.norm(p[m_row[i], :])**2 + np.linalg.norm(q[:, m_col[i]])
             ** 2+np.linalg.norm(b_user)**2+np.linalg.norm(b_item)**2+np.linalg.norm(w_u)**2+np.linalg.norm(w_i)**2)
    return SSE


def biasSVDweight(m, a, maxK, lamb):
    print('Initializing...')
    [m_row, m_col, m_val, row, col] = m
    length = len(m_row)
    [b_user, b_item, sigma] = [np.zeros(row), np.zeros(col), sum(m_val)/length]
    step = 1
    p = np.random.rand(row, maxK)
    q = np.random.rand(maxK, col)
    b_user, b_item = np.random.rand(row), np.random.rand(col)
    omega_u = np.zeros(row)  # 用户的评论数
    omega_i = np.zeros(col)  # 物品的评论数
    w_u = np.random.rand(row)  # 用户的权重
    w_i = np.random.rand(col)  # 物品的权重
    for i in range(length):
        omega_u[m_row[i]] += 1
        omega_i[m_col[i]] += 1
    omega_u = omega_u/row  #
    omega_i = omega_i/col  # 经验函数处理
    # 初始化完毕
    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q, lamb, b_user, b_item,
                     sigma, omega_u, omega_i, w_u, w_i)
    RMSE = sqrt(SSE/length)
    print('Initialization finished. RMSE=', RMSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'RMSE=', RMSE, end='\r')
        # time.sleep(0.5)
        for i in range(0, length):
            user = m_row[i]
            item = m_col[i]
            _sum = m_val[i]-np.dot(p[user, :]+omega_u[user]*w_u[user], q[:, item]+omega_i[item]*w_i[item]) - \
                sigma-b_user[user]-b_item[item]
            ###### 更新P,Q ######
            tmp = p[user, :]  # 保存前一个状态的P_U
            p[user, :] = (1-a*lamb/RMSE/length)*p[user, :] + \
                a*_sum*(q[:, item]+omega_u[user]*w_u[user])/RMSE/length
            q[:, item] = (1-a*lamb/RMSE/length) * q[:, item] +\
                a*_sum*(tmp+omega_i[item]*w_i[item])/RMSE/length
            # 更新偏置 b_user 和 b_item ######
            b_user[user] = b_user[user]+a * \
                (_sum-lamb*b_user[user])/RMSE/length
            b_item[item] = b_item[item]+a * \
                (_sum-lamb*b_item[item])/RMSE/length
            # 更新权重 w_u 和 w_i ######
            w_u[user] = w_u[user]+a * \
                (_sum*omega_u[user]-lamb*w_u[user])/RMSE/length
            w_i[item] = w_i[item]+a * \
                (_sum*omega_i[item]-lamb*w_i[item])/RMSE/length

        # for i in m_row:
        #     p[i][k] = p[i][k]+2*a*_sum*q[k][item]
        # for j in m_col:
        #     q[k][j] = q[k][j]+2*a*_sum*tmp
        newSSE = computeSSE(m, p, q, lamb, b_user, b_item,
                            sigma, omega_u, omega_i, w_u, w_i)
        newRMSE = sqrt(newSSE/length)
        if abs(newRMSE-RMSE) <= 10**-5:
            print('', end='\n')
            return [p, q, sigma, b_user, b_item, omega_u, omega_i, w_u, w_i]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q, sigma, b_user, b_item, omega_u, omega_i, w_u, w_i]
