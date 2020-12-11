import numpy as np
import time
from math import sqrt


def computeSSE(m, p, q):
    SSE = 0
    for i in range(0, length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])) ** 2
    return SSE


def SGD(m, a, maxK):
    print('Initializing...')
    [m_row, m_col, m_val] = m
    row = max(m_row)+1
    col = max(m_col)+1
    length = len(m_row)
    step = 1
    p = np.zeros((row, maxK))
    q = np.zeros((maxK, col))
    for k in range(0, maxK):
        for i in m_row:
            p[i][k] = 1
        for j in m_col:
            q[k][j] = 1
    print('Computing first SSE...')
    SSE = computeSSE(m, p, q)
    RMSE = sqrt(SSE/length)
    print('Initialization finished. RMSE=', RMSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'RMSE=', RMSE, end='\r')
        # time.sleep(0.5)
        for i in range(0, length):
            user = m_row[i]
            item = m_col[i]
            _sum = m_val[i]-np.dot(p[user, :], q[:, item])
            for k in range(0, maxK):
                tmp = p[user][k]
                p[user][k] = p[user][k]+a*_sum*q[k][item]/RMSE
                q[k][item] = q[k][item]+a*_sum*tmp/RMSE
                # for i in m_row:
                #     p[i][k] = p[i][k]+2*a*_sum*q[k][item]
                # for j in m_col:
                #     q[k][j] = q[k][j]+2*a*_sum*tmp
        newSSE = computeSSE(m, p, q)
        newRMSE = sqrt(newRMSE/length)
        if abs(newRMSE-RMSE) <= 10**-4:
            print('', end='\n')
            return [p, q, RMSE]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q, RMSE]
