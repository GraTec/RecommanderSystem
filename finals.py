import numpy as np
from SGD import SGD
from loadData import loadData
from nSGD import nSGD
from biasSVD import biasSVD
from math import sqrt
import random
from dataSplit import dataSplit, getTrainTest


def test():
    m_row = [0, 0, 1, 3]
    m_col = [0, 4, 2, 1]
    m_val = [1, 2, 3, 4]
    [p, q] = SGD([m_row, m_col, m_val, 5, 5], 0.01, 3)
    print(p.dot(q))
    [p, q] = nSGD([m_row, m_col, m_val, 5, 5], 0.01, 3, 0.001)
    print(p.dot(q))


def computeRMSE(m, p, q):
    [m_row, m_col, m_val] = m
    length = len(m_row)
    SSE = 0
    for i in range(length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]])) ** 2
    RMSE = sqrt(SSE/length)
    return RMSE


def computeRMSEbias(m, p, q, sigma, b_user, b_item):
    [m_row, m_col, m_val] = m
    length = len(m_row)
    SSE = 0
    for i in range(length):
        SSE += (m_val[i]-np.dot(p[m_row[i], :], q[:, m_col[i]]) -
                sigma-b_user[m_row[i]]-b_item[m_col[i]]) ** 2
    RMSE = sqrt(SSE/length)
    return RMSE


def finalTest():
    # Reading data
    [m, implicit] = loadData()
    [m_row, m_col, m_val] = m
    [implicit_row, implicit_col, implicit_val] = implicit
    length = len(m_row)
    # Reading finished
    # Split data
    dimMax = 10**3
    m2 = dataSplit(m, [dimMax, dimMax])
    [m2_row, m2_col, m2_val] = m2
    # Split finished
    [m_train, m_test] = getTrainTest(m2)
    [p, q] = SGD(m_train, 0.001, 10)
    print(computeRMSE(m_test, p, q))
    # print(p)

    [p, q] = nSGD(m_train, 0.005, 10, 0.1)
    print(computeRMSE(m_test, p, q))
    # print(p)

    [p, q, sigma, b_user, b_item] = biasSVD(m_train, 0.0025, 10, 0.5)
    print(computeRMSEbias(m_test, p, q, sigma, b_user, b_item))


# test()
finalTest()
