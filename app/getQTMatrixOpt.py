import random
import numpy as np
import numpy.random

# summarize matrix columns
def sum_columns(matrix):
    sum = []
    for i in range(q):
        col_sum = 0
        for j in range(s):
            col_sum += matrix[j][i]
        sum.append(col_sum)
    return sum


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
def SQ_distance(arr):
    dist = 0
    for i in arr:
        dist += (i * (i - 1)) // 2
    return dist


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


def ST_to_QT(ST):
    QT = np.zeros([q, t], dtype=int)
    for i in range(s):
        for j in range(t):
            QT[ST[i, j], j] += 1
    return QT


def QT_optimization(QT, ST):
    loop = True
    loop_iter = 0
    iter = 0
    while loop:
        loop_iter += 1
        loop = False
        for k in range(s):
            for i in range(t - 1):
                for j in range(i + 1, t):

                    if check_block(QT[ST[k, i], i], QT[ST[k, i], j], QT[ST[k, j], i], QT[ST[k, j], j]) > 0:
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
        result = dist[A] + dist[B] + dist[C] + dist[D] - dist[A - 1] - dist[B + 1] - dist[C + 1] - dist[D - 1]
    return result


def QT_optimization2(QT, ST):
    loop = True

    bricks = generate_bricks(QT)

    while len(bricks) and loop:
        if not update_bricks(bricks, QT, ST):
            loop = False


def find_row(ST, ilist, jlist, irow, jrow):
    ilist2 = []
    jlist2 = []
    for i in ilist:
        if jrow not in np.array(ST[i]):
            ilist2.append(0)
        else:
            ilist2.append(1)
    for j in jlist:
        if irow not in np.array(ST[j]):
            jlist2.append(0)
        else:
            jlist2.append(1)

    return ilist2, jlist2


def update_bricks(bricks, QT, ST):
    bricks = generate_bricks(QT)

    for i, (irow, icol) in enumerate(bricks):
        for j, (jrow, jcol) in enumerate(bricks):
            if j > i:
                if (QT[jrow, icol] < avg and QT[irow, jcol] <= avg) or (QT[jrow, icol] <= avg and QT[irow, jcol] < avg):
                    ilist = []
                    jlist = []
                    for k in range(s):
                        if ST[k, icol] == irow:
                            ilist.append(k)
                        if ST[k, jcol] == jrow:
                            jlist.append(k)

                    ilist2, jlist2 = find_row(ST, ilist, jlist, irow, jrow)

                    if 0 in ilist2 and 0 in jlist2:
                        iswap = ilist2.index(0)
                        jswap = jlist2.index(0)

                        ST_swap(ST, QT, ilist[iswap], icol, jlist[jswap], jcol)

                        bricks = generate_bricks(QT)
                        return 1
                    else:
                        if 0 in ilist2:

                            iswap = ilist2.index(0)

                            if ilist[iswap] in jlist:

                                ST_swap(ST, QT, ilist[iswap], icol, ilist[iswap], jcol)

                                bricks = generate_bricks(QT)
                                return 1
                            else:
                                for k in jlist:
                                    kcol = list(ST[k]).index(irow)
                                    for l in range(s):
                                        if l not in jlist:
                                            if irow not in ST[l] and ST[l, kcol] not in ST[k]:

                                                ST_swap(ST, QT, k, kcol, l, kcol)
                                                ST_swap(ST, QT, ilist[iswap], icol, k, jcol)

                                                bricks = generate_bricks(QT)
                                                return 1

                        if 0 in jlist2:
                            jswap = jlist2.index(0)

                            if jlist[jswap] in ilist:

                                ST_swap(ST, QT, jlist[jswap], icol, jlist[jswap], jcol)

                                bricks = generate_bricks(QT)
                                return 1
                            else:
                                for k in ilist:
                                    kcol = list(ST[k]).index(jrow)
                                    for l in range(s):
                                        if l not in ilist:
                                            if jrow not in ST[l] and ST[l, kcol] not in ST[k]:

                                                ST_swap(ST, QT, k, kcol, l, kcol)
                                                ST_swap(ST, QT, jlist[jswap], jcol, k, icol)

                                                bricks = generate_bricks(QT)
                                                return 1
                        if 0 not in ilist2 and 0 not in jlist2:
                            if QT_optimization(QT, ST) == 0:
                                for k in ilist:
                                    jindex = list(ST[k]).index(jrow)
                                    tempcol = [row[jindex] for row in ST]
                                    for l in range(s):
                                        if l != k:
                                            if tempcol[l] not in ST[k] and jrow not in ST[l]:
                                                ST_swap(ST, QT, k, jindex, l, jindex)
                                                bricks = generate_bricks(QT)
                                                return 1

                                for k in jlist:
                                    iindex = list(ST[k]).index(irow)
                                    tempcol = [row[iindex] for row in ST]
                                    for l in range(s):
                                        if l != k:
                                            if tempcol[l] not in ST[k] and irow not in ST[l]:
                                                ST_swap(ST, QT, k, iindex, l, iindex)
                                                bricks = generate_bricks(QT)
                                                return 1
                            else:
                                bricks = generate_bricks(QT)
                                return 1

    for k, (krow, kcol) in enumerate(bricks):

        ilist = []
        for i in range(s):
            if ST[i, kcol] == krow:
                ilist.append(i)

        jlist = []
        for j in range(q):
            if QT[j, kcol] < avg and QT[j, kcol] > 0:
                jlist.append(j)

        for i in ilist:
            for j in jlist:
                if j not in ST[i]:
                    ST[i, kcol] = j
                    QT[krow, kcol] -= 1
                    QT[j, kcol] += 1

                    return 1
    return 0


def remove_bricks(QT, bricks, i, irow, icol, j, jrow, jcol):
    bricks = generate_bricks(QT)


def ST_swap(ST, QT, irow, icol, jrow, jcol):
    mrow = ST[irow, icol]
    mcol = icol
    nrow = ST[jrow, jcol]
    ncol = jcol

    temp = ST[irow, icol]
    ST[irow, icol] = ST[jrow, jcol]
    ST[jrow, jcol] = temp

    QT[mrow, mcol] -= 1
    QT[mrow, ncol] += 1
    QT[nrow, mcol] += 1
    QT[nrow, ncol] -= 1


def generate_bricks(QT):
    bricks = []
    for ix, i in enumerate(QT):
        for jx, j in enumerate(i):
            if j > avg:
                bricks.append((ix, jx))
    return bricks


def setSTMatrix(data):
    global q, t, s, SQ, r, avg, dist, c, zlim

    q = len(data["listOfQ"])
    t = data["T"]
    s = len(data["listOfS"])

    SQ = []

    r = np.ceil((s * t) / q)  # how many times questions are repeated
    avg = int(np.ceil(s / q))

    dist = [(i * (i - 1) // 2) for i in range(s + 1)]

    c = 0
    zlim = 0

    SQ_random_generator()
    SQ_optimization(SQ)
    ST = SQ_to_ST(SQ)
    ST = ST_shuffle(ST)
    QT = ST_to_QT(ST)
    n_iter = QT_optimization(QT, ST)
    zlim = np.amax(QT)
    n_iter2 = QT_optimization2(QT, ST)

    return_dict = {}
    for i in range(s):
        temp_arr = []
        for j in range(t):
            temp_arr.append(data["listOfQ"][ST[i][j]])
        return_dict[data["listOfS"][i]] = temp_arr

    return return_dict
