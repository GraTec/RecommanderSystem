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


def biasSVD(m, a, maxK, lamb):
    print('Initializing...')
    [m_row, m_col, m_val, row, col] = m
    length = len(m_row)
    [b_user, b_item, sigma] = [np.zeros(row), np.zeros(col), sum(m_val)/length]
    step = 1
    p = np.zeros((row, maxK))
    q = np.zeros((maxK, col))
    for k in range(0, maxK):
        for i in m_row:
            p[i][k] = np.random.rand()
        for j in m_col:
            q[k][j] = np.random.rand()
    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma)
    RMSE = sqrt(SSE/length)
    print('Initialization finished. RMSE=', RMSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'RMSE=', RMSE, end='\r')
        # time.sleep(0.5)
        b2_user = b_user
        b2_item = b_item
        for i in range(0, length):
            user = m_row[i]
            item = m_col[i]
            _sum = m_val[i]-np.dot(p[user, :], q[:, item]) - \
                sigma-b_user[user]-b_item[item]
            for k in range(0, maxK):
                tmp = p[user][k]
                p[user][k] = p[user][k]+a * \
                    (_sum * q[k][item]-lamb*p[user][k])/RMSE
                q[k][item] = q[k][item]+a*(_sum*tmp-lamb*q[k][item])/RMSE
            b_user[user] = b_user[user]+a * \
                (_sum-lamb*b_user[user])/RMSE
            b_item[item] = b_item[item]+a * \
                (_sum-lamb*b_item[item])/RMSE
        # for i in m_row:
        #     p[i][k] = p[i][k]+2*a*_sum*q[k][item]
        # for j in m_col:
        #     q[k][j] = q[k][j]+2*a*_sum*tmp
        newSSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma)
        newRMSE = sqrt(newSSE/length)
        if abs(newRMSE-RMSE) <= 10**-5:
            print('', end='\n')
            return [p, q, sigma, b_user, b_item]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q, sigma, b_user, b_item]
