import random
# import combinations
import numpy as np
# from collections import Counter
# import histogram


q = 20                  # number of questions
t = 10                     # number of selected questions in the test
s = 10                    # number of students
SQ = []
dist = 0
r = 0
    # np.ceil((s*t)/q)
avg = 0
    # int(np.ceil(s/q))
dist = 0
    # [(i*(i-1)//2) for i in range(s+1)]

# print matrix
def print_matrix(matrix):
    print("\n")
    for i in range(len(matrix)):
        row = "["
        for j in range(len(matrix[i])-1):
            row += str(matrix[i][j]) + ', '
        row += str(matrix[i][len(matrix[i])-1])
        print(row + ']')


# summarize matrix columns
def sum_columns(matrix):
    sum = []
    for i in range(q):
        col_sum = 0
        for j in range(s):
            col_sum += matrix[j][i]
        sum.append(col_sum)
    return sum

# def print_histogram(QT, zlim):
#     xpos = []
#     ypos = []
#     dz = []
#
#     for i in range(q):
#         for j in range(t):
#             xpos.append(i+0.25)
#             ypos.append(j+0.25)
#             dz.append(QT[i][j])
#
#     histogram.Plot_3D_Histogram(xpos, ypos, dz, zlim)

# generate binary matrix
def SQ_random_generator():
    for i in range(s):
        tempArr = []
        r_sum = 0
        for j in range(q):
            r = 1
            r_sum += r
            if r_sum <= t:
                tempArr.append(r)
            else:
                tempArr.append(0)
        random.shuffle(tempArr)
        SQ.append(tempArr)

# shuffle the questions that minimize SQ-distance
def SQ_optimization(matrix):
    SQ = matrix
    count_col = sum_columns(SQ)
    max_el = max(count_col)
    min_el = min(count_col)
    while abs(min_el - max_el) > 1:
        max_indx = count_col.index(max_el)
        min_indx = count_col.index(min_el)

        i = 0
        while SQ[i][max_indx] != 1 or SQ[i][min_indx] != 0: i += 1

        SQ[i][max_indx] = 0
        SQ[i][min_indx] = 1
        count_col = sum_columns(SQ)

        max_el = max(count_col)
        min_el = min(count_col)

# SQ-distance that identicates overall number of common questions
# def SQ_distance(arr):
#     dist = 0
#     for i in arr:
#         dist += (i * (i-1))//2
#     return dist

# converts SQ-matrix into ST-matrix
def SQ_to_ST(SQ):
    ST = []
    for i in SQ:
        row = []
        for j in range(len(i)):
            if i[j]:
                row.append(j)
        ST.append(row)
    return np.array(ST)

def ST_shuffle(ST):
    np.random.shuffle(ST)
    ST_permutations = np.empty([ST.shape[0], ST.shape[1]], dtype=int)
    for i in range(ST.shape[0]):
        np.random.seed(i)
        ST_permutations[i] = np.random.permutation(ST[i])
    return ST_permutations

# def ST_to_QT(ST):
#     QT = np.zeros([q, t], dtype=int)
#     for i in range(s):
#         for j in range(t):
#             QT[ ST[i,j], j ] += 1
#     return QT


def QT_optimization(QT, ST):

    loop = True
    loop_iter = 0

    while loop:
        loop_iter += 1
        # print("loop_iter = ", loop_iter)
        loop = False
        iter = 0
        for k in range(s):
            for i in range(t-1):
                for j in range(i+1, t):

                    if check_block(QT[ST[k,i], i], QT[ST[k,i], j], QT[ST[k,j], i], QT[ST[k,j], j]) > 0:
                        QT[ST[k, i], i] -= 1
                        QT[ST[k, i], j] += 1
                        QT[ST[k, j], i] += 1
                        QT[ST[k, j], j] -= 1
                        temp = ST[k, i]
                        ST[k, i] = ST[k, j]
                        ST[k, j] = temp
                        loop = True
                        iter += 1
    return iter

def check_block(A, B, C, D):
    if A < 1 or D < 1:
        result = -1
    else:
        result = dist[A] + dist[B] + dist[C] + dist[D] - dist[A-1] - dist[B+1] - dist[C+1] - dist[D-1]
    return result


def setSTMatrix(data):
    q = len(data["listOfQ"])
    t = data["T"]
    s = len(data["listOfS"])
    r = np.ceil((s*t)/q)
    avg = int(np.ceil(s/q))
    dist = [(i * (i - 1) // 2) for i in range(s + 1)]

    SQ_random_generator()
    SQ_optimization(SQ)
    ST = SQ_to_ST(SQ)
    ST = ST_shuffle(ST)
    # print_matrix(ST)

    return_dict = {}
    for i in range(s):
        temp_arr = []
        for j in range(t):
            temp_arr.append(data["listOfQ"][ST[i][j]])
        return_dict[data["listOfS"][i]] = temp_arr

    # for i in return_dict:
    #     print(i, return_dict[i])

    return return_dict

