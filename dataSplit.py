import random


def dataSplit(m, dim):
    [m_row, m_col, m_val] = m
    [rowMax, colMax] = dim
    [m2_row, m2_col, m2_val] = [[], [], []]
    length = len(m_row)
    for i in range(0, length):
        if m_row[i] < rowMax and m_col[i] < colMax:
            m2_row.append(m_row[i])
            m2_col.append(m_col[i])
            m2_val.append(m_val[i])
    return [m2_row, m2_col, m2_val]


def implicitSplit(implicit, dim):
    [rowMax, colMax] = dim
    im2 = [[] for i in range(0, rowMax)]
    for i in range(0, rowMax):
        for j in range(0, len(implicit[i])):
            if implicit[i][j] < colMax:
                im2[i].append(implicit[i][j])
    return im2


def getTrainTest(m, rate=0.8):
    [m_row, m_col, m_val] = m
    length = len(m_row)
    dataList = list(range(0, length))
    random.shuffle(dataList)
    row = max(m_row)+1
    col = max(m_col)+1
    m_train = [[], [], [], row, col]
    m_test = [[], [], []]
    for i in range(0, int(length*rate)):
        m_train[0].append(m_row[dataList[i]])
        m_train[1].append(m_col[dataList[i]])
        m_train[2].append(m_val[dataList[i]])
    for i in range(int(length*rate), length):
        m_test[0].append(m_row[dataList[i]])
        m_test[1].append(m_col[dataList[i]])
        m_test[2].append(m_val[dataList[i]])
    return [m_train, m_test]
