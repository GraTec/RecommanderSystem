import numpy as np
import time
from math import sqrt


def computeSSE(m, p, q, lamb):  # 计算SSE，加入正则化系数
    [m_row, m_col, m_val] = m[0:3]
    length = len(m_row)
    SSE = 0
    for i in range(0, length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])) ** 2+lamb * \
            (np.linalg.norm(p[m_row[i], :])**2 +
             np.linalg.norm(q[:, m_col[i]])**2)
    return SSE


def rfunkSVD(m, a, maxK, lamb, eps):  # m为评分矩阵，a为alpha参数，maxK为k的参数，lamb为正则化参数，eps为终止条件
    print('Initializing...')
    [m_row, m_col, m_val, row, col] = m
    length = len(m_row)
    step = 1
    p = np.random.rand(row, maxK)
    q = np.random.rand(maxK, col)
    # 初始化完毕
    print('Computing first RMSE...')
    SSE = computeSSE(m, p, q, lamb)
    RMSE = sqrt(SSE/length)
    # 初始RMSE误差计算完毕
    print('Initialization finished. RMSE=', RMSE, '. Factoring matrix...')
    while step <= 10**5:
        print('Running step', step, 'RMSE=', RMSE, end='\r')
        # time.sleep(0.5)
        for i in range(0, length):  # 梯度下降法，每一次迭代遍历所有变量
            user = m_row[i]
            item = m_col[i]
            _sum = m_val[i]-np.dot(p[user, :], q[:, item])
            ###### 更新P,Q ######
            tmp = p[user, :]  # 保存前一个状态的P_U
            p[user, :] = (1-a*lamb/RMSE/length)*p[user, :] + \
                a*_sum*q[:, item]/RMSE/length
            q[:, item] = (1-a*lamb/RMSE/length)*q[:, item] + \
                a*_sum*tmp/RMSE/length

        newSSE = computeSSE(m, p, q, lamb)
        newRMSE = sqrt(newSSE/length)  # 计算新的误差
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
    m = [[0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4], [0, 4, 2, 0, 2,
                                             3, 1, 3, 1, 4, 0], [1, 2, 3, 2, 4, 2, 4, 1, 4, 1, 5], 5, 5]
    p, q = rfunkSVD(m, 0.01, 5, 0.2, 1e-6)
    print(p)
    print(q)
    print(p@q)
