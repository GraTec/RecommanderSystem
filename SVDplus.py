import numpy as np
import time
from math import sqrt


def sumy(y, implicit, user):
    length = len(implicit[user])
    vec = np.zeros((1, len(y[0])))
    norm = 0
    for j in implicit[user]:
        vec = vec+y[j]
        norm += np.linalg.norm(y[j])**2
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
    length = len(m_row)
    [b_user, b_item, sigma] = [np.zeros(row), np.zeros(col), sum(m_val)/length]
    step = 1
    p = np.random.rand(row, maxK)
    q = np.random.rand(maxK, col)
    y = np.random.rand(col, maxK)

    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma, y, implicit)
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
            [vec, norm] = sumy(y, implicit, user)
            _sum = m_val[i]-np.dot(p[user, :]+vec/max(sqrt(len(y[user, :])), 1),
                                   q[:, item]) - sigma-b_user[user]-b_item[item]
            q_item = q[:, item]
            for k in range(0, maxK):
                tmp = p[user][k]
                p[user][k] = p[user][k]+a * \
                    (_sum * q[k][item]-lamb*p[user][k])/RMSE
                q[k][item] = q[k][item]+a * \
                    (_sum*(tmp+vec[0, k]/max(sqrt(len(implicit[user])), 1)
                           ) - lamb*q[k][item])/RMSE

            b_user[user] = b_user[user]+a * \
                (_sum-lamb*b_user[user])/RMSE
            b_item[item] = b_item[item]+a * \
                (_sum-lamb*b_item[item])/RMSE
            if item in implicit[user]:
                y[item, :] = y[item, :]+a * \
                    (_sum*np.transpose(q_item) /
                     max(sqrt(len(y[user, :])), 1)-lamb*y[item, :])/RMSE
        # for i in m_row:
        #     p[i][k] = p[i][k]+2*a*_sum*q[k][item]
        # for j in m_col:
        #     q[k][j] = q[k][j]+2*a*_sum*tmp
        newSSE = computeSSE(m, p, q, lamb, b_user, b_item, sigma, y, implicit)
        newRMSE = sqrt(newSSE/length)
        if abs(newRMSE-RMSE) <= 10**-5:
            print('', end='\n')
            return [p, q, sigma, b_user, b_item, y]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q, sigma, b_user, b_item, y]
