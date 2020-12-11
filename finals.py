import numpy as np
from SGD import SGD
from loadData import loadData


def test():
    m_row = [0, 0, 1, 3]
    m_col = [0, 4, 2, 1]
    m_val = [1, 2, 3, 4]
    [p, q, SSE] = SGD([m_row, m_col, m_val], 0.01, 3)
    print(p.dot(q), SSE)


def finalTest():
    [m, implicit] = loadData()
    print(len(m[0]))
    [p, q, SSE] = SGD([m[0][:10**3], m[1][:10**3], m[2][:10**3]], 0.025, 10)
    print(SSE)


test()
finalTest()
