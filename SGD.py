import numpy as np
import time


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
    SSE = 0
    for i in range(0, length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])) ** 2
    print('Initialization finished. SSE=', SSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'SSE=', SSE, end='\r')
        # time.sleep(0.5)
        random = np.random.randint(length)
        user = m_row[random]
        item = m_col[random]
        _sum = m_val[random]-np.dot(p[user, :], q[:, item])
        while abs(_sum) <= 10**-8:
            random = np.random.randint(length)
            user = m_row[random]
            item = m_col[random]
            _sum = m_val[random]-np.dot(p[user, :], q[:, item])
        for k in range(0, maxK):
            tmp = p[user][k]
            p[user][k] = p[user][k]+2*a*_sum*q[k][item]
            q[k][item] = q[k][item]+2*a*_sum*tmp
            # for i in m_row:
            #     p[i][k] = p[i][k]+2*a*_sum*q[k][item]
            # for j in m_col:
            #     q[k][j] = q[k][j]+2*a*_sum*tmp
        newSSE = 0
        for i in range(0, length):
            newSSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])) ** 2
        if abs(newSSE-SSE) <= 10**-8:
            print('', end='\n')
            return [p, q, SSE]
        step += 1
        SSE = newSSE
    print('', end='\n')
    return [p, q, SSE]
