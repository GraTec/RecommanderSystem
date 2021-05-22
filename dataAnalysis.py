import csv
import re
import numpy as np
import scipy.sparse as ss
from loadData import loadData

############# 用于分析数据集 ###############
# 数据集格式："username";"bookName";"rate" #
# 用户信息从上到下一定是递增的               #
###########################################


def biasAnalysis():  # 分析有多少用户有偏置
    [[m_row, m_col, m_val], implicit] = loadData()
    length = len(m_row)
    posCnt, negCnt = 0, 0  # 偏好好评和偏好差评的人数
    minNum = 5  # 至少需要minNum个评价才能认为有偏置
    i = 0
    while i < length:  # 遍历所有用户
        if m_val[i] == 0:  # 隐含信息，不统计
            i += 1
            continue
        if m_val[i] >= 4:  # 用户偏好好评
            flag = True
            num = 1
            for j in range(i+1, length):  # 遍历该用户的剩余评价
                if m_row[i] != m_row[j]:  # 该用户遍历完毕
                    break
                if m_val[j] < 4 and m_val[j] > 0:  # 该用户不偏好好评
                    flag = False
                    break
                num += 1
            if flag and num >= minNum:
                posCnt += 1
        elif m_val[i] <= 2 and m_val[i] > 0:  # 用户偏好差评
            flag = True
            num = 1
            for j in range(i+1, length):  # 遍历该用户的剩余评价
                if m_row[i] != m_row[j]:  # 该用户遍历完毕
                    break
                if m_val[j] > 2:  # 该用户不偏好差评
                    flag = False
                    break
                num += 1
            if flag and num >= minNum:
                negCnt += 1
        for j in range(i, length):  # 跳到下一个用户
            if m_row[i] != m_row[j]:
                i = j
                break
            if j == length-1:
                i = length
    return [m_row[-1], posCnt, negCnt]


def weightAnalysis():  # 分析不同用户的评价数
    [[m_row, m_col, m_val], implicit] = loadData()
    length = len(m_row)
    i = 0
    rateNum = np.zeros(10)  # rateNum[i]代表给了小于i*1000个评价的用户数
    rateNumLess1000 = np.zeros(10)  # 对于小于1000评价的，以100为一个分层
    rateNumLess10 = 0  # 评论数小于10的用户
    rateNumEqual1 = 0  # 评论数为1的用户
    while i < length:  # 遍历所有用户
        if m_val[i] == 0:  # 隐含信息，不统计
            i += 1
            continue
        rateCnt = 0  # 评价数计算
        for j in range(i, length):  # 遍历该用户的评价
            if m_row[i] != m_row[j]:  # 该用户遍历完毕
                break
            if m_val[j] == 0:  # 隐含信息，不统计
                continue
            rateCnt += 1
        if rateCnt == 1:
            rateNumEqual1 += 1
        if rateCnt <= 10:
            rateNumLess10 += 1
        if rateCnt <= 1000:
            rateNumLess1000[int(rateCnt/100)] += 1
        else:
            rateNum[int(rateCnt/1000)] += 1
        for j in range(i, length):  # 跳到下一个用户
            if m_row[i] != m_row[j]:
                i = j
                break
            if j == length-1:
                i = length
    return [rateNumEqual1, rateNumLess10, rateNumLess1000, rateNum]


if __name__ == "__main__":
    # print(loadData())
    # print(biasAnalysis())
    print(weightAnalysis())
