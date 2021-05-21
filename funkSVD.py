import numpy as np
import time
from math import sqrt


def computeSSE(m, p, q):  # 计算computeSSE
    [m_row, m_col, m_val] = m[0:3]  # 从m中读取数据
    length = len(m_row)
    SSE = 0
    for i in range(0, length):  # 遍历计算SSE
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])) ** 2
    return SSE


def funkSVD(m, a, maxK, eps):  # m为评分矩阵，a为alpha参数，maxK为k的参数，eps为终止条件
    print('Initializing...')
    [m_row, m_col, m_val, row, col] = m
    length = len(m_row)
    step = 1
    p = np.zeros((row, maxK))
    q = np.zeros((maxK, col))
    for k in range(0, maxK):
        for i in m_row:
            p[i][k] = np.random.rand()
        for j in m_col:
            q[k][j] = np.random.rand()
    # 初始化完毕
    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q)
    RMSE = sqrt(SSE/length)
    # 初始RMSE误差计算完毕
    print('Initialization finished. RMSE=', RMSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'RMSE=', RMSE, end='\r')
        # time.sleep(0.5)
        for i in range(0, length):  # 每一次迭代遍历所有变量
            user = m_row[i]
            item = m_col[i]
            _sum = m_val[i]-np.dot(p[user, :], q[:, item])
            for k in range(0, maxK):  # 梯度下降法，对P,Q进行更新
                tmp = p[user][k]
                p[user][k] = p[user][k]+a*_sum*q[k][item]/RMSE
                q[k][item] = q[k][item]+a*_sum*tmp/RMSE
                # for i in m_row:
                #     p[i][k] = p[i][k]+2*a*_sum*q[k][item]
                # for j in m_col:
                #     q[k][j] = q[k][j]+2*a*_sum*tmp
        newSSE = computeSSE(m, p, q)
        newRMSE = sqrt(newSSE/length)
        if abs(newRMSE-RMSE) <= eps:  # 迭代终止判断
            print('', end='\n')
            return [p, q]
        step += 1
        RMSE = newRMSE
    print('', end='\n')
    return [p, q]


if __name__ == "__main__":
    # m` = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [
    #  `            2, 0, 4, 2, 0], [0, 4, 0, 1, 0], [0, 4, 0, 0, 1]])
    # m = [[0, 0, 1, 2, 2, 2, 3, 3, 4, 4], [0, 4, 2, 0, 2,
    #                                       3, 1, 3, 1, 4], [1, 2, 3, 2, 4, 2, 4, 1, 4, 1], 5, 5]
    m = [[0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4], [0, 4, 2, 0, 2,
                                             3, 1, 3, 1, 4, 0], [1, 2, 3, 2, 4, 2, 4, 1, 4, 1, 5], 5, 5]

    p, q = funkSVD(m, 0.01, 5, 1e-5)
    print(p)
    print(q)
    print(p@q)
