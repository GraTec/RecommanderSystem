import csv
import re
import numpy as np
import scipy.sparse as ss


def loadData():
    userSet = {}
    userId = 0
    bookSet = {}
    bookId = 0
    with open('./BX-CSV-Dump/dataset.csv', newline='') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            row = row[0].split(';')
            if row[0] == '':
                break
            [user, book, rate] = row

            if user not in userSet.keys():
                userSet[user] = userId
                userId += 1
            if book not in bookSet.keys():
                bookSet[book] = bookId
                bookId += 1
    # print(userId, bookId)
    # m = np.zeros((userId, bookId))
    # implicit = np.zeros((userId, bookId))
    [m_row, m_col, m_val] = [[], [], []]
    implicit = [[] for i in range(0, bookId)]
    with open('./BX-CSV-Dump/dataset.csv', newline='') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            row = row[0].split(';')
            if row[0] == '':
                break
            [user, book, rate] = row
            user = userSet[user]
            book = bookSet[book]
            rate = rate.replace('"', '')
            if int(rate) == 0:
                implicit[user].append(book)
            else:
                m_row.append(user)
                m_col.append(book)
                m_val.append(int(rate))
    # m = ss.coo_matrix((m_val, (m_row, m_col)))
    # implicit = ss.coo_matrix((implicit_val, (implicit_row, implicit_col)))
    return [[m_row, m_col, m_val], implicit]


if __name__ == "__main__":
    print(loadData())
