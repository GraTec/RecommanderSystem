import numpy as np
from SGD import SGD
from loadData import loadData
from nSGD import nSGD
from biasSVD import biasSVD
from biasSVDweight import biasSVDweight
from SVDplus import SVDplus
from math import sqrt
import random
from dataSplit import dataSplit, getTrainTest, implicitSplit


def test():
    m_row = [0, 0, 1, 4]
    m_col = [0, 4, 2, 1]
    m_val = [1, 2, 3, 4]
    [p, q] = SGD([m_row, m_col, m_val, 5, 5], 0.01, 3)
    print(p, q)
    print(p.dot(q))
    # [p, q] = nSGD([m_row, m_col, m_val, 5, 5], 0.01, 3, 0.001)
    # print(p.dot(q))


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


def sumy(y, implicit, user):
    length = len(implicit[user])
    vec = np.zeros((1, len(y[0])))
    norm = 0
    for j in implicit[user]:
        vec = vec+y[j]
        norm += np.linalg.norm(y[j])**2
    return [vec, norm]


def computeRMSEplus(m, p, q, sigma, b_user, b_item, y, implicit):
    [m_row, m_col, m_val] = m
    length = len(m_row)
    SSE = 0
    for i in range(0, length):
        [vec, norm] = sumy(y, implicit, m_row[i])
        SSE += (m_val[i]-np.dot(p[m_row[i], :]+vec/max(sqrt(len(y[m_row[i]])), 1),
                                q[:, m_col[i]])-sigma-b_user[m_row[i]]-b_item[m_col[i]]) ** 2
    RMSE = sqrt(SSE/length)
    return RMSE


def finalTest():
    # Reading data
    [m, implicit] = loadData()
    # Reading finished
    # Split data
    dimMax = 10**3
    m1 = dataSplit(m, [dimMax, dimMax])
    [m1_train, m1_test] = getTrainTest(m1)
    m1 = dataSplit(m, [2*dimMax, 2*dimMax])
    [m2_train, m2_test] = getTrainTest(m1)
    m1 = dataSplit(m, [3*dimMax, 3*dimMax])
    [m3_train, m3_test] = getTrainTest(m1)
    implicit1 = implicitSplit(implicit, [dimMax, dimMax])
    implicit2 = implicitSplit(implicit, [2*dimMax, 2*dimMax])
    implicit3 = implicitSplit(implicit, [3*dimMax, 3*dimMax])

    # Split finished
    [p1, q1] = SGD(m1_train, 0.001, 10)
    [p2, q2] = SGD(m2_train, 0.001, 10)
    [p3, q3] = SGD(m3_train, 0.001, 10)
    RMSE_SGD = [0, 0, 0]
    RMSE_SGD[0] = computeRMSE(m1_test, p1, q1)
    RMSE_SGD[1] = computeRMSE(m2_test, p2, q2)
    RMSE_SGD[2] = computeRMSE(m3_test, p3, q3)
    print(RMSE_SGD)
    # print(p)

    [p1, q1] = nSGD(m1_train, 0.005, 10, 0.1)
    [p2, q2] = nSGD(m2_train, 0.005, 10, 0.1)
    [p3, q3] = nSGD(m3_train, 0.005, 10, 0.1)
    RMSE_nSGD = [0, 0, 0]
    RMSE_nSGD[0] = computeRMSE(m1_test, p1, q1)
    RMSE_nSGD[1] = computeRMSE(m2_test, p2, q2)
    RMSE_nSGD[2] = computeRMSE(m3_test, p3, q3)
    print(RMSE_nSGD)
    # print(p)

    [p1, q1, sigma1, b_user1, b_item1] = biasSVD(m1_train, 0.0001, 10, 0.1)
    [p2, q2, sigma2, b_user2, b_item2] = biasSVD(m2_train, 0.0001, 10, 0.1)
    [p3, q3, sigma3, b_user3, b_item3] = biasSVD(m3_train, 0.0001, 10, 0.1)
    RMSE_biasSVD = [0, 0, 0]
    RMSE_biasSVD[0] = computeRMSEbias(
        m1_test, p1, q1, sigma1, b_user1, b_item1)
    RMSE_biasSVD[1] = computeRMSEbias(
        m2_test, p2, q2, sigma2, b_user2, b_item2)
    RMSE_biasSVD[2] = computeRMSEbias(
        m3_test, p3, q3, sigma3, b_user3, b_item3)
    print(RMSE_biasSVD)

    [p1, q1, sigma1, b_user1, b_item1] = biasSVDweight(
        m1_train, 0.0001, 10, 0.1)
    [p2, q2, sigma2, b_user2, b_item2] = biasSVDweight(
        m2_train, 0.0001, 10, 0.1)
    [p3, q3, sigma3, b_user3, b_item3] = biasSVDweight(
        m3_train, 0.0001, 10, 0.1)
    RMSE_biasSVDweight = [0, 0, 0]
    RMSE_biasSVDweight[0] = computeRMSEbias(
        m1_test, p1, q1, sigma1, b_user1, b_item1)
    RMSE_biasSVDweight[1] = computeRMSEbias(
        m2_test, p2, q2, sigma2, b_user2, b_item2)
    RMSE_biasSVDweight[2] = computeRMSEbias(
        m3_test, p3, q3, sigma3, b_user3, b_item3)
    print(RMSE_biasSVDweight)

    with open('./result', 'w') as f:
        f.write(str(RMSE_SGD))
        f.write(str(RMSE_nSGD))
        f.write(str(RMSE_biasSVD))
        f.write(str(RMSE_biasSVDweight))
        f.close()


def plusTest():
    # Reading data
    [m, implicit] = loadData()
    # Reading finished
    # Split data
    dimMax = 10**3
    m1 = dataSplit(m, [dimMax, dimMax])
    [m1_train, m1_test] = getTrainTest(m1)
    m1 = dataSplit(m, [2*dimMax, 2*dimMax])
    [m2_train, m2_test] = getTrainTest(m1)
    m1 = dataSplit(m, [3*dimMax, 3*dimMax])
    [m3_train, m3_test] = getTrainTest(m1)
    m1_train[3] = dimMax
    m1_train[4] = dimMax
    m2_train[3] = 2*dimMax
    m2_train[4] = 2*dimMax
    m3_train[3] = 3*dimMax
    m3_train[4] = 3*dimMax
    implicit1 = implicitSplit(implicit, [dimMax, dimMax])
    implicit2 = implicitSplit(implicit, [2*dimMax, 2*dimMax])
    implicit3 = implicitSplit(implicit, [3*dimMax, 3*dimMax])

    [p1, q1, sigma1, b_user1, b_item1, y] = SVDplus(
        m1_train, 0.0001, 10, 0.1, implicit1)
    [p2, q2, sigma2, b_user2, b_item2, y] = SVDplus(
        m2_train, 0.0001, 10, 0.01, implicit2)
    [p3, q3, sigma3, b_user3, b_item3, y] = SVDplus(
        m3_train, 0.0001, 10, 0.01, implicit3)
    RMSE_SVDplus = [0, 0, 0]
    RMSE_SVDplus[0] = computeRMSEplus(
        m1_test, p1, q1, sigma1, b_user1, b_item1, y, implicit1)
    RMSE_SVDplus[1] = computeRMSEplus(
        m2_test, p2, q2, sigma2, b_user2, b_item2, y, implicit2)
    RMSE_SVDplus[2] = computeRMSEplus(
        m3_test, p3, q3, sigma3, b_user3, b_item3, y, implicit3)
    print(RMSE_SVDplus)


# test()
# finalTest()
plusTest()
