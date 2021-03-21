from __future__ import print_function
from numpy import linalg as la
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import math


def sparse_read(fn, dx, dy):
    """
    read the sparse matrix from file
    :param fn: file name of date
    :param dx: dimension of X
    :param dy:  dimension of Y
    :param mtype: data type of matrix
    :return: the sparse matrix
    """
    f = open(fn, 'r')
    fx = []    # X-axis
    fy = []    # Y-axis
    fd = []    # data

    for line in f:
        tlist = line.strip().split('\t')
        fx.append(int(tlist[0]))
        fy.append(int(tlist[1]))
        fd.append(float(tlist[2]))

    x = csr_matrix((fd, (fx, fy)), shape=(dx, dy), dtype='float')
    f.close()
    return x


def l21_norm(a):
    """
    L2,1 norm of a. Square root of sum of squares of all elements in a column, and sum them
    It is the transposition of universal l2,1 norm
    :param a: the matrix
    :return:L2,1 norm
    """
    return np.sum(la.norm(a, ord=2, axis=0))


def derivative_l21_norm(a):
    """
    During derivative, the transposition of a is processed
    Then, return the transposition of result
    :param a:
    :return:
    """
    t = a.T
    x, y = t.shape
    row_2norm = la.norm(t, ord=2, axis=1)
    row_2norm = 1./row_2norm
    coefficient_matrix = np.zeros((x, x))
    for i in range(x):
        coefficient_matrix[i][i] = row_2norm[i]
    res = np.dot(coefficient_matrix, t)
    return res.T


def laplace_matrix(a):
    """
    get the laplace matrix of square matrix
    :param a:
    :return:
    """
    x, y = a.shape
    if x != y:
        print("The input of laplace_matrix is not a square matrix!")
    d = np.zeros((x, y))
    for i in range(x):
        summ = 0
        for j in range(y):
            summ += a[i][j]
        d[i][i] = summ
    res = d - a
    return res


def normalize_matrix_columns(w):
    """
    Normalize a matrix columns to unit norm
    :param w: matrix to be normalized
    :type w: numpy.array
    :returns: normalized matrix
    """
    #print(np.linalg.norm(w, ord=2, axis=1))
    return (w.T / np.linalg.norm(w, ord=2, axis=1)).T


def regularized_laplace_matrix(a):
    """
    get the laplace matrix of square matrix
    :param a:
    :return:
    """
    x, y = a.shape
    if x != y:
        print("The input of laplace_matrix is not a square matrix!")

    d = np.zeros((x, y))
    for i in range(x):
        summ = 0
        for j in range(y):
            summ += a[i][j]
        d[i][i] = summ

    d05 = np.zeros((x, y))
    for i in range(x):
        d05[i][i] = d[i][i] ** (-0.5)

    res = d05.dot(d - a).dot(d05)
    return res


# below are for sparse
def sparse_write(xi, fn):
    assert isinstance(xi, csr_matrix)
    eps = 1e-8
    x, _ = xi.shape
    xiout = open(fn, 'w')
    for i in range(x):
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            if abs(xi.data[j]) > eps:
                xiout.write(str(i) + '\t' + str(xi.indices[j]) + '\t' + str(xi.data[j]) + '\n')
    xiout.close()





def sparse_normalized_row(xi):
    assert isinstance(xi, csr_matrix)
    x_sum = xi.sum(axis=1)
    print(type(x_sum))
    # eps = 1e-8
    x, _ = xi.shape
    for i in range(x):
        # if abs(x_sum[i]) > eps:
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            xi.data[j] /= x_sum[i]
    return xi


def sparse_model_normalized_row(xi):
    assert isinstance(xi, csr_matrix)
    # eps = 1e-8
    x, _ = xi.shape
    for i in range(x):
        sum = 0
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            sum += np.square(xi.data[j])
        sum = np.sqrt(sum)
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            xi.data[j] /= sum
    return xi

def sparse_log10_normalized(xi):
    assert isinstance(xi, csr_matrix)
    # eps = 1e-8
    x, _ = xi.shape
    for i in range(x):
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            xi.data[j] = math.log10(xi.data[j])
    return xi


def sparse_min_max_normalized_row(xi):
    """
    
    :param xi:
    :return:
    """
    assert isinstance(xi, csr_matrix)
    m, n = xi.shape
    eps = 1e-8
    x_min = []
    x_max = []
    for i in range(m):
        if xi.indptr[i+1] - xi.indptr[i] < 1:
            x_max.append(eps)
            x_min.append(eps)
        else:
            maxx = xi.data[xi.indptr[i]]
            minn = xi.data[xi.indptr[i]]
            for j in range(xi.indptr[i]+1, xi.indptr[i+1]):
                if minn > xi.data[j]:
                    minn = xi.data[j]
                if maxx < xi.data[j]:
                    maxx = xi.data[j]
            x_max.append(maxx)
            x_min.append(minn)

    for i in range(m):
        diff = x_max[i] - x_min[i]
        if diff < eps:
            if xi.indptr[i + 1] - xi.indptr[i] == 1:
                xi.data[xi.indptr[i]] = 1
            elif xi.indptr[i + 1] - xi.indptr[i] > 1:
                avg = 1.0 / (xi.indptr[i + 1] - xi.indptr[i])
                avg = max(avg, eps)
                for j in range(xi.indptr[i], xi.indptr[i + 1]):
                    xi.data[j] = avg
        else:
            for j in range(xi.indptr[i], xi.indptr[i+1]):
                xi.data[j] = (xi.data[j] - x_min[i]) / diff

    return xi



def sparse_transpose(xi):
    """
    csr_matrix.transpose() return the csc_matrix rather than csr_matrix, so the calculation based on csr is not suit after transpose
    :param xi:
    :return: the transpose csr_matrix
    """
    assert isinstance(xi, csr_matrix)
    x, y = xi.shape
    tx = []
    ty = []
    td = []

    for i in range(x):
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            tx.append(i)
            ty.append(xi.indices[j])
            td.append(xi.data[j])

    return csr_matrix((td, (ty, tx)), shape=(y, x), dtype='float')


def sparse_tl21_norm(xi):
    """
    L2,1 norm of xi. Square root of sum of squares of all elements in a column, and sum them
    It is the Transposition of universal l2,1 norm
    :param xi: the matrix
    :return: L2,1 norm
    """
    assert isinstance(xi, csr_matrix)
    res = 0
    t = sparse_transpose(xi)
    x, _ = t.shape
    for i in range(x):
        inside_res = 0
        if t.indptr[i+1] - t.indptr[i] == 0:
            continue
        for j in range(t.indptr[i], t.indptr[i+1]):
            inside_res += np.square(t.data[j])
        res += np.sqrt(inside_res)
    return res


def sparse_regularized_laplace(xi):
    """
    get the laplace matrix of square matrix
    :param xi:
    :return:
    """
    assert isinstance(xi, csr_matrix)
    x, y = xi.shape
    if x != y:
        print("The input of laplace_matrix is not a square matrix!")

    tx = []
    ty = []
    td = []
    for i in range(x):
        summ = 0
        if xi.indptr[i+1] - xi.indptr[i] == 0:
            continue
        for j in range(xi.indptr[i], xi.indptr[i+1]):
            summ += xi.data[j]
        tx.append(i)
        ty.append(i)
        td.append(summ)

    d = csr_matrix((td, (tx, ty)), shape=(x, y), dtype='float')
    for i in range(len(td)):
        td[i] = td[i] ** (-0.5)
    d05 = csr_matrix((td, (tx, ty)), shape=(x, y), dtype='float')

    return d05.dot(d-xi).dot(d05)


def sparse_derivative_tl21_norm(xi):
    """
    During derivative, the transposition of a is processed
    Then, return the transposition of result
    :param xi:
    :return:
    """
    assert isinstance(xi, csr_matrix)
    t = sparse_transpose(xi)
    x, y = t.shape

    eps = 1e-8
    tx = []
    ty = []
    td = []
    for i in range(x):
        inside_res = 0
        if t.indptr[i+1] - t.indptr[i] == 0:
            continue
        for j in range(t.indptr[i], t.indptr[i+1]):
            inside_res += np.square(t.data[j])
        if abs(inside_res) < eps:
            continue
        tx.append(i)
        ty.append(i)
        td.append(1.0/np.sqrt(inside_res))

    coefficient_matrix = csr_matrix((td, (tx, ty)), shape=(x, x), dtype='float')

    return sparse_transpose(sparse_dot(coefficient_matrix, t))


def sparse_plus_eps(xi):
    """
    plus a very small number for every nonzero
    :param xi:
    :return:
    """
    assert isinstance(xi, csr_matrix)
    eps = 1e-8
    for i in range(len(xi.data)):
        xi.data[i] += 1e-8
    return xi


def sparse_dot(x1, x2):
    """
    csr_matrix.dot(other_array)
    :param x1:
    :param x2:
    :return:
    """
    assert isinstance(x1, csr_matrix)
    assert isinstance(x2, csr_matrix)
    x, r1 = x1.shape
    r2, y = x2.shape
    r = r1
    if r1 != r2:
        print('The wrong size for dot multiplication: (%d, %d), (%d, %d)' % (x, r1, r2, y))
    # print('The dot dimension is (%d, %d)' % (x, y))  ###
    eps = 1e-8

    tx = []
    ty = []
    td = []
    midres = {}
    for i in range(x):
        for j in range(x1.indptr[i], x1.indptr[i+1]):
            for k in range(x2.indptr[x1.indices[j]], x2.indptr[x1.indices[j]+1]):
                inres = x1.data[j]*x2.data[k]
                # if abs(inres) > eps:
                key = (i, x2.indices[k])
                if key not in midres:
                    midres[key] = inres
                else:
                    midres[key] = midres[key] + inres
    for k, v in midres.items():
        tx.append(k[0])
        ty.append(k[1])
        td.append(v)

    return csr_matrix((td, (tx, ty)), shape=(x, y), dtype='float')


def sparse_elementwise_division(x1, x2):
    assert isinstance(x1, csr_matrix)
    assert isinstance(x2, csr_matrix)
    m1, n1 = x1.shape
    m2, n2 = x2.shape
    if m1 != m2 or n1 != n2:
        print('The wrong size for elementwise division')

    eps = 1e-8
    tx = []
    ty = []
    td = []
    for i in range(m1):
        j = x1.indptr[i]
        k = x2.indptr[i]
        while j < x1.indptr[i+1] and k < x2.indptr[i+1]:
            if x1.indices[j] == x2.indices[k]:
                # if abs(x2.data[k]) > eps:
                tx.append(i)
                ty.append(x1.indices[j])
                td.append(x1.data[j] / x2.data[k])
                k += 1
                j += 1
            elif x1.indices[j] > x2.indices[k]:
                k += 1
            elif x1.indices[j] < x2.indices[k]:
                j += 1

    return csr_matrix((td, (tx, ty)), shape=(m1, n1), dtype='float')


def sparse_elementwise_product(x1, x2):
    assert isinstance(x1, csr_matrix)
    assert isinstance(x2, csr_matrix)
    m1, n1 = x1.shape
    m2, n2 = x2.shape
    if m1 != m2 or n1 != n2:
        print('The wrong size for elementwise division')

    tx = []
    ty = []
    td = []
    for i in range(m1):
        j = x1.indptr[i]
        k = x2.indptr[i]
        while j < x1.indptr[i+1] and k < x2.indptr[i+1]:
            if x1.indices[j] == x2.indices[k]:
                tx.append(i)
                ty.append(x1.indices[j])
                td.append(x1.data[j] * x2.data[k])
                k += 1
                j += 1
            elif x1.indices[j] > x2.indices[k]:
                k += 1
            elif x1.indices[j] < x2.indices[k]:
                j += 1

    return csr_matrix((td, (tx, ty)), shape=(m1, n1), dtype='float')


def sparse_grad_product(x1, x2, a, b):
    """
    Very Important Note: x1 is csR_matrix, x2 is csC_matrix
    :param x1:
    :param x2:
    :param a:
    :param b:
    :return: element (a, b) of x1*x2
    """
    assert isinstance(x1, csr_matrix)
    assert isinstance(x2, csc_matrix)
    _, r1 = x1.shape
    r2, _ = x2.shape
    if r1 != r2:
        print('The wrong size for grad_product')

    res = 0
    i = x1.indptr[a]
    j = x2.indptr[b]
    while i < x1.indptr[a+1] and j < x2.indptr[b+1]:
        if x1.indices[i] == x2.indices[j]:
            res += x1.data[i] * x2.data[j]
            i += 1
            j += 1
        elif x1.indices[i] > x2.indices[j]:
            j += 1
        elif x1.indices[i] < x2.indices[j]:
            i += 1

    return res


def sparse_double_fnorm(x1, x3):
    """
    obtain F-norm of x1*x3 without saving x1*x3 to avoid memory overflow(||x1*x2||)
    :param x1:
    :param x3:
    :return:
    """
    assert isinstance(x1, csr_matrix)
    assert isinstance(x3, csr_matrix)
    m, r1 = x1.shape
    r2, n = x3.shape
    if r1 != r2:
        print('The wrong size for grad_product')

    x2 = x3.tocsc()
    res = 0
    for i in range(m):
        for j in range(n):
            inres = sparse_grad_product(x1, x2, i, j)
            res += np.square(inres)

    return np.sqrt(res)


def sparse_middle_1norm(x0, x1, x3):
    """
    sum the all elements of array x0*x1*x2. sparser x0, less running price
    :param x0:
    :param x1:
    :param x3:
    :return:
    """
    assert isinstance(x0, csr_matrix)
    assert isinstance(x1, csr_matrix)
    assert isinstance(x3, csr_matrix)
    # check 3 dimension
    m, r1 = x0.shape
    r2, r3 = x1.shape
    r4, n = x3.shape
    if r1 != r2 or r3 != r4:
        print('The wrong size for sparse_middle_1norm: (%d, %d), (%d, %d), (%d, %d).' % (m, r1, r2, r3, r4, n))

    x2 = x3.tocsc()
    res = 0

    for i in range(m):
        for u in range(x0.indptr[i], x0.indptr[i+1]):
            j = x0.indices[u]
            for k in range(n):
                res += x0.data[u] * sparse_grad_product(x1, x2, j, k)

    return res


def sparse_tuple_fnorm(x0, x1, x3):
    """
    obtain F-norm of x0-x1*x3 without saving x1*x3 to avoid memory overflow
    :param x0:
    :param x1:
    :param x3:
    :return:
    """
    assert isinstance(x0, csr_matrix)
    assert isinstance(x1, csr_matrix)
    assert isinstance(x3, csr_matrix)
    r2, r1 = x0.shape
    m, r3 = x1.shape
    r4, n = x3.shape
    if m != r2 or r3 != r4 or r1 != n:
        print('The wrong size for sparse_middle_1norm: (%d, %d), (%d, %d), (%d, %d).' % (r2, r1, m, r3, r4, n))

    # x2 is csC_matrix of x3
    x2 = x3.tocsc()
    res = 0
    for i in range(m):
        k = x0.indptr[i]
        for j in range(n):
            inres = sparse_grad_product(x1, x2, i, j)
            if j == x0.indices[k] and k < x0.indptr[i+1]:
                inres = x0.data[k] - inres
                k += 1
            res += np.square(inres)

    return np.sqrt(res)


def sparse_remove_lower(x1, thr):
    """
    remove the element which lower than threshold
    :param x1:
    :param thr:
    :return:
    """
    assert isinstance(x1, csr_matrix)
    m, n = x1.shape
    tx = []
    ty = []
    td = []
    for i in range(m):
        for j in range(x1.indptr[i], x1.indptr[i+1]):
            if x1.data[j] >= thr:
                tx.append(i)
                ty.append(x1.indices[j])
                td.append(x1.data[j])
    res = csr_matrix((td, (tx, ty)), shape=(m, n), dtype='float')
    return res


def sparse_set_min(x1, thr):
    """
    Set the element which lower than threshold to threshold
    :param x1:
    :param thr:
    :return:
    """
    assert isinstance(x1, csr_matrix)
    m, n = x1.shape
    tx = []
    ty = []
    td = []
    for i in range(m):
        for j in range(x1.indptr[i], x1.indptr[i+1]):
            tx.append(i)
            ty.append(x1.indices[j])
            if x1.data[j] >= thr:
                td.append(x1.data[j])
            else:
                td.append(thr)

    res = csr_matrix((td, (tx, ty)), shape=(m, n), dtype='float')
    return res

def sparse_trace(x1, x3):
    """
    calculate the trace of X1*X3
    :param x1: 
    :param x3: 
    :return: 
    """
    assert isinstance(x1, csr_matrix)
    assert isinstance(x3, csr_matrix)
    m, r1 = x1.shape
    r2, n = x3.shape
    if r1 != r2 or m != n:
        print('The wrong dimensions of sparse_trace: (%d, %d), (%d, %d)' % (m, r1, r2, n))

    # x2 = x3.tocsc()
    tx = []
    ty = []
    td = []
    for kk in range(len(x3.indptr) - 1):
        for j in range(x3.indptr[kk], x3.indptr[kk + 1]):
            tx.append(kk)
            ty.append(x3.indices[j])
            td.append(x3.data[j])
    x2 = csc_matrix((td, (tx, ty)), shape=x3.shape, dtype='float')
    x4 = sparse_reset(x1)
    
    res = 0
    for i in range(m):
        res += sparse_grad_product(x4, x2, i, i)
    return res


def sparse_all_plus_eps(xi):
    """
    plus a very small number for every nonzero
    :param xi:
    :return:
    """
    assert isinstance(xi, csr_matrix)
    m, n = xi.shape
    tx = []
    ty = []
    td = []

    for i in range(m):
        for j in range(n):
            tx.append(i)
            ty.append(j)
            td.append(1e-8)
    yi = csr_matrix((td, (tx, ty)), shape=(m, n), dtype='float')
    return xi + yi


def sparse_difference_sum(x1, x2):
    """
    First, obtain the difference of sparse matrix x1 and x2
    Second, sum the abs of each element in dense difference
    :param x1: 
    :param x2: 
    :return: 
    """
    m1, n1 = x1.shape
    m2, n2 = x2.shape
    if m1 != m2 or n1 != n2:
        print('Wrong dimension from sparse_difference_sum: (%d, %d), (%d, %d)' % (m1, n1, m2, n2))
    t = (x1-x2).toarray()
    res = 0.0
    for i in range(m1):
        for j in range(n1):
            res += abs(t[i][j])
    return res


def sparse_reset(xi):
    """
    After the library function of csr_matrix, its elements may be disorder. 
    So, before the operation which need order of csr_matrix, it must be reseted
    :param xi:
    :return:
    """
    assert isinstance(xi, csr_matrix)
    tx = []
    ty = []
    td = []
    for kk in range(len(xi.indptr) - 1):
        for j in range(xi.indptr[kk], xi.indptr[kk + 1]):
            tx.append(kk)
            ty.append(xi.indices[j])
            td.append(xi.data[j])
    return csr_matrix((td, (tx, ty)), shape=xi.shape, dtype='float')


def sparse_write_detail(xi, fn):
    assert isinstance(xi, csr_matrix)
    f = open(fn, 'w')
    for kk in range(len(xi.indptr)-1):
        for j in range(xi.indptr[kk], xi.indptr[kk+1]):
            print('[%d, %d, %.8f]' % (kk, xi.indices[j], xi.data[j]), file=f, end='\t')
        print(file=f)
    f.close()


def sparse_magnification(xi, times):
    assert isinstance(xi, csr_matrix)
    tx = []
    ty = []
    td = []
    for kk in range(len(xi.indptr) - 1):
        for j in range(xi.indptr[kk], xi.indptr[kk + 1]):
            tx.append(kk)
            ty.append(xi.indices[j])
            td.append(xi.data[j] * times)
    return csr_matrix((td, (tx, ty)), shape=xi.shape, dtype='float')

