import time
import utils
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import norm
from scipy.sparse import rand
import matplotlib.pyplot as plt
import logging
NUM_CID = 0

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

def tran(xi):
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

def div(x1, x2):
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

def prod(x1, x2):
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

def trace(x1, x3):
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

def load_all_data(fnxi, fnxo, fnxc, fnyib, fnyic, fnidi, fnidb, fnidc):
    # read the id of sku/brand/cid3
    fidi = open(fnidi, 'r')     # file of sku to id
    fidb = open(fnidb, 'r')     # file of brand to id
    fidc = open(fnidc, 'r')     # file of cid3 to id

    total_sku_to_id = {}
    for line in fidi:
        tlist = line.strip().split('\t')
        total_sku_to_id[int(tlist[0])] = int(tlist[1])

    total_brand_to_id = {}
    for line in fidb:
        tlist = line.strip().split('\t')
        total_brand_to_id[int(tlist[0])] = int(tlist[1])

    total_cid3_to_id = {}
    for line in fidc:
        tlist = line.strip().split('\t')
        total_cid3_to_id[int(tlist[0])] = int(tlist[1])

    fidi.close()
    fidb.close()
    fidc.close()

    # read the relationships matrix
    xi = utils.sparse_read(fnxi, len(total_sku_to_id), len(total_sku_to_id))
    xo = utils.sparse_read(fnxo, len(total_sku_to_id), len(total_sku_to_id))
    xc = utils.sparse_read(fnxc, len(total_cid3_to_id), len(total_cid3_to_id))
    yib = utils.sparse_read(fnyib, len(total_sku_to_id), len(total_brand_to_id))
    yic = utils.sparse_read(fnyic, len(total_sku_to_id), len(total_cid3_to_id))

    return xi, xo, xc, yib, yic, total_sku_to_id, total_brand_to_id, total_cid3_to_id


def objective(xi, xo, xc, yib, yic, vi, vo, vb, vc, li, lo, lc, xiT, xoT, xcT, yibT, yicT, a, b):
    """
    The objective function of multi dimensional matrix factorization
    miss Xb, Lb
    :param xi: item*item
    :param xo: item*item (order)
    :param xb: brand*brand
    :param xc: cid3*cid3
    :param yib: item*brand
    :param yic: item*cid3
    :param vi: factorization matrix of item
    :param vo: factorization matrix of item (order)
    :param vb: factorization matrix of brand
    :param vc: factorization matrix of cid3
    :param li: regularized laplace matrix of x_i
    :param lo: regularized laplace matrix of x_o
    :param lb: regularized laplace matrix of x_b
    :param lc: regularized laplace matrix of x_c
    :return: the result of objective function
    """
    viT = tran(vi)
    voT = tran(vo)
    vbT = tran(vb)
    vcT = tran(vc)


    res = 0
    res += 0.5 * ( np.square(norm(xi, 'fro')) - 2 * trace(xiT.dot(vi), viT) + trace(vi.dot(viT.dot(vi)), viT) )
    res += 0.5 * ( np.square(norm(xo, 'fro')) - 2 * trace(xoT.dot(vo), voT) + trace(vo.dot(voT.dot(vo)), voT) )
    res += 0.5 * ( np.square(norm(xc, 'fro')) - 2 * trace(xcT.dot(vc), vcT) + trace(vc.dot(vcT.dot(vc)), vcT) )

    res += 0.5 * ( np.square(norm(yib, 'fro')) - 2 * trace(yibT.dot(vi), vbT) + trace(vb.dot(viT.dot(vi)), vbT) )
    res += 0.5 * ( np.square(norm(yic, 'fro')) - 2 * trace(yicT.dot(vi), vcT) + trace(vc.dot(viT.dot(vi)), vcT) )
    res += 0.5 * ( np.square(norm(yib, 'fro')) - 2 * trace(yibT.dot(vo), vbT) + trace(vb.dot(voT.dot(vo)), vbT) )
    res += 0.5 * ( np.square(norm(yic, 'fro')) - 2 * trace(yicT.dot(vo), vcT) + trace(vc.dot(voT.dot(vo)), vcT) )

    res += a * 0.5 * (np.square(norm(vi, 'fro')))
    res += a * 0.5 * (np.square(norm(vo, 'fro')))
    res += a * 0.5 * (np.square(norm(vb, 'fro')))
    res += a * 0.5 * (np.square(norm(vc, 'fro')))

    res += b * utils.sparse_tl21_norm(vc)
    return res


def basic_nmf(xi, xo, xc, yib, yic, r, max_iter, stop_condition, a, b, outdirpath):
    """

    :param xi:
    :param xo:
    :param xc:
    :param yib:
    :param yic:
    :param r:
    :param max_iter:
    :param stop_condition: Minimal required improvement of the residuals from the previous iteration.
    :param a:
    :param b:
    :param outdirpath:
    :return:
    """
    parapath = '_a' + str(a) + '_b' + str(b) + '_d' + str(r)
    di, db = yib.shape
    _, dc = yic.shape
    vi = rand(di, r, density=1, format='csr', dtype='float') #  density equal to one means a full matrix, density of 0 means a matrix with no non-zero items.
    vo = rand(di, r, density=1, format='csr', dtype='float')
    vb = rand(db, r, density=1, format='csr', dtype='float')
    vc = rand(dc, r, density=1, format='csr', dtype='float')

    li = utils.sparse_regularized_laplace(xi)
    lo = utils.sparse_regularized_laplace(xo)
    lc = utils.sparse_regularized_laplace(xc)
    print('Vi dimension is (%d, %d)' % (vi.shape))
    print('Vo dimension is (%d, %d)' % (vo.shape))
    print('Vb dimension is (%d, %d)' % (vb.shape))
    print('Vc dimension is (%d, %d)' % (vc.shape))
    xiT = tran(xi)
    xoT = tran(xo)
    xcT = tran(xc)
    yibT = tran(yib)
    yicT = tran(yic)
    f_loss = open(outdirpath + '/loss_curve' + parapath + '.txt', "w")

    eps = 1e-10
    pdist = 1e8
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=outdirpath + '/my' + parapath +'.log', level=logging.DEBUG, format=LOG_FORMAT)

    for i in range(max_iter):
        start = time.process_time()
        print('The iteration %d is running...(vi:%d, vo:%d, vb:%d, vc:%d)' % (i, len(vi.data), len(vo.data), len(vb.data), len(vc.data)))

        # Vi
        numerator = yib.dot(vb) + yic.dot(vc) + xi.dot(vi) + xiT.dot(vi)   # + eps
        denominator = vi.dot(tran(vb).dot(vb)) + a*vi + vi.dot(tran(vc).dot(vc)) + 2 * vi.dot(tran(vi).dot(vi))
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        vi = prod(vi, div(numerator, denominator))


        numerator = xo.dot(vo) + xoT.dot(vo) + yib.dot(vb) + yic.dot(vc)
        denominator = 2 * vo.dot(tran(vo).dot(vo)) + vo.dot(tran(vb).dot(vb)) + \
            vo.dot(tran(vc).dot(vc)) + a*vo
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        vo = prod(vo, div(numerator, denominator))


        numerator = yibT.dot(vi) + yibT.dot(vo)
        denominator = vb.dot(tran(vi).dot(vi)) + a*vb + vb.dot(tran(vo).dot(vo))
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        vb = prod(vb, div(numerator, denominator))


        numerator = yicT.dot(vi) + xc.dot(vc) + xcT.dot(vc) + yicT.dot(vo)
        denominator = vc.dot(tran(vi).dot(vi)) + a*vc + 2 * vc.dot(tran(vc).dot(vc)) \
                      + vc.dot(tran(vo).dot(vo)) + b*utils.sparse_derivative_tl21_norm(vc)
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        vc = prod(vc, div(numerator, denominator))


        end = time.process_time()
        print('The iteration end, time is %f.(vi:%d, vo:%d, vb:%d, vc:%d)' % (end-start, len(vi.data), len(vo.data), len(vb.data), len(vc.data)))

        if i % 20 == 0:
            print('Output results after iteration %d' % (i))
            np.savetxt(outdirpath + '/4. vc' + parapath + '.txt', vc.toarray(), fmt="%.10f", delimiter=" ")

            print("Computing loss...")
            dist = objective(xi, xo, xc, yib, yic, vi, vo, vb, vc, li, lo, lc, xiT, xoT, xcT, yibT, yicT, a, b)
            print("Distance (%d) is %.10f, absolute distance is %.10f" % (i, pdist - dist, dist))
            if i > 0:
                print("%d %.10f" % (i, dist), file=f_loss)
                logging.info("%s\t%d\t%.10f\t%.10f" % (parapath, i, dist, pdist - dist))
                if 0 < pdist-dist < stop_condition:
                    break
            pdist = dist

    # plot the convergence curve
    f_loss.close()
    curve = np.loadtxt(outdirpath + '/loss_curve' + parapath +'.txt')
    plt.clf()
    plt.plot(curve[:, 0], curve[:, 1])
    plt.savefig(outdirpath + '/loss_curve' + parapath +'.png')

    # output the results
    np.savetxt(outdirpath + '/4. vc' + parapath + '.txt', vc.toarray(), fmt="%.10f", delimiter=" ")


def construct_scene_from_array(array, thr, min_construct_scene):
    m, n = array.shape
    res = []
    for j in range(n):
        tl = []
        for i in range(m):
            if array[i][j] > thr:
                tl.append(i)
        if len(tl) >= min_construct_scene:
            res.append(tl)
    return res

def saveSense(scenes_list, fno):
    fo = open(fno, 'w')
    for scene in scenes_list:
        first = True
        for cate in scene:
            if first:
                print(cate, end='', file=fo)
                first = False
            else:
                print('\t' + str(cate), end='', file=fo)
        print('', end='\n', file=fo)

    fo.close()

def load_ground_truth(fn):
    gt = []
    f = open(fn, 'r')
    for line in f:
        tlist = line.strip().split('\t')
        tl = []
        for t in tlist:
            tl.append(int(t))
        gt.append(tl)
    f.close()
    return gt

def f1(p, g):
    """
    calculate the F1-score
    :param p: prediction list
    :param g: ground-truth list
    :return: F1-score
    """
    intersection_num = 0
    for e in p:
        if e in g:
            intersection_num += 1
    precision = float(intersection_num) / len(p)
    recall = float(intersection_num) / len(g)
    eps = 1e-8
    if intersection_num > eps:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0

def average_f1(predictions, groundtruths):
    """
    Calculate the Average F1 score
    :param predictions: the list of the list of cid in the same scene produced by program
    :param groundtruths: the list of the list of cid in the same scene in ground-truth
    :return:
    """
    # eps = 1e-8
    # if len(predictions) < eps:
    #     return 0

    f1pg = 0.0  # sum of F1 score for prediction to ground-truth
    for prediction in predictions:
        inres = 0.0
        for groundtruth in groundtruths:
            tres = f1(prediction, groundtruth)
            if tres > inres:
                inres = tres
        f1pg += inres
    f1pg /= len(predictions)

    f1gp = 0.0  # sum of F1 score for ground-truth to prediction
    for groundtruth in groundtruths:
        inres = 0.0
        for prediction in predictions:
            tres = f1(prediction, groundtruth)
            if tres > inres:
                inres = tres
        f1gp += inres
    f1gp /= len(groundtruths)

    return 0.5 * (f1pg + f1gp)

def omega_index(predictions, groundtruths, n):
    """
    Calculate the Omega Index
    :param predictions: the list of the list of cid in the same scene produced by program
    :param groundtruths: the list of the list of cid in the same scene in ground-truth
    :param n: the number of elements
    :return:
    """
    cp = np.zeros((n, n), dtype=np.int)  # Co-occur array of elements in predictions
    for prediction in predictions:
        for i in prediction:
            for j in prediction:
                cp[i][j] += 1

    cg = np.zeros((n, n), dtype=np.int)
    for groundtruth in groundtruths:
        for i in groundtruth:
            for j in groundtruth:
                cg[i][j] += 1

    eps = 1e-8
    res = 0
    for i in range(n):
        for j in range(n):
            if cp[i][j] == cg[i][j] and cp[i][j] > eps:
                res += 1
    return float(res) / np.square(n)

# -------------------------------------------------------------
# The method of calculating NMI is proposed by Andrea Lancichinetti, Santo Fortunato and János Kertész in paper： Detecting the overlapping and
# hierarchical community structure in complex networks (shorted by NMI_LFK)

def h(x):
    if x > 0:
        return -1 * x * math.log(x, 2)
    else:
        return 0

# H(Xi)
def H(Xi):
    global NUM_CID
    p1 = len(Xi) / NUM_CID
    p0 = 1 - p1
    # print('#Xi: %d, NUM_CID: %d, p1: %f, p0: %f' % (len(Xi), NUM_CID, p1, p0))
    eps = 1e-8
    sum = h(p0) + h(p1)
    if sum >= eps:
        return h(p0) + h(p1)
    else:
        return eps

# H(Xi, Yj)
def H_Xi_joint_Yj(Xi, Yj):
    Xi_set = set(Xi)
    Yj_set = set(Yj)
    global NUM_CID
    P11 = len(Xi_set & Yj_set) / NUM_CID  # intersection(Xi, Yj).size()
    P10 = len(Xi_set - Yj_set) / NUM_CID  # difference(Xi, Yj).size()
    P01 = len(Yj_set - Xi_set) / NUM_CID  # difference(Yj, Xi).size()
    P00 = 1 - P11 - P10 - P01

    if h(P11) + h(P00) >= h(P01) + h(P10):
        return h(P11) + h(P10) + h(P01) + h(P00)
    else:
        return H(Xi) + H(Yj)

# H(Xi|Yj)
def H_Xi_given_Yj(Xi, Yj):
    return H_Xi_joint_Yj(Xi, Yj) - H(Yj)

# H(Xi|Y) return min{H(Xi|Yj)} for all j
def H_Xi_given_Y(Xi, Y):
    res = H_Xi_given_Yj(Xi, Y[0])
    for j in range(1, len(Y)):
        res = min(res, H_Xi_given_Yj(Xi, Y[j]))
    return res

# H(Xi|Y)_norm
def H_Xi_given_Y_norm(Xi, Y):
    return H_Xi_given_Y(Xi, Y) / H(Xi)

# H(X|Y)_norm
def H_X_given_Y_norm(X, Y):
    res = 0
    for Xi in X:
        res += H_Xi_given_Y_norm(Xi, Y)
    return res / len(X)

def nmi_lfk(X, Y):
    return 1 - 0.5 * (H_X_given_Y_norm(X, Y) + H_X_given_Y_norm(Y, X))
# -------------------------------------------------------------

def main():
    global NUM_CID
    max_iter = 500
    stop_condition = 0.000001
    min_construct_scene = 1
    dimentions = [64, 64, 64, 64] # [128, 128, 128, 128]
    indirpaths = ['../in_baby_toy',
              '../in_electronics',
              '../in_fashion',
              '../in_food_drink'
                ]
    outdirpaths = ['../out_baby_toy',
              '../out_electronics',
              '../out_fashion',
               '../out_food_drink'
                ]
    num_cids = [103, 78, 91, 105]
    a = 1
    b = 5
    thr = 0.1
    # k = 0
    for k in range(len(indirpaths)):
        r = dimentions[k]
        indirpath = indirpaths[k]
        outdirpath = outdirpaths[k]

        xi, xo, xc, yib, yic, total_sku_to_id, total_brand_to_id, total_cid3_to_id = \
            load_all_data(indirpath + '/xi_log10_min_max_normalized_new.txt', indirpath + '/xo_log10_min_max_normalized_new.txt',
                          indirpath + '/xc.txt', indirpath + '/yib.txt',
                          indirpath + '/yic.txt', indirpath + '/ID_sku_to_id.txt', indirpath + '/ID_brand_to_id.txt',
                          indirpath + '/ID_cid3_to_id.txt')

        print('The relation matrices and ID Indexes have been read. (sku: %d, brand: %d, cid3: %d)' % (len(total_sku_to_id), len(total_brand_to_id), len(total_cid3_to_id)))
        basic_nmf(xi, xo, xc, yib, yic, r, max_iter, stop_condition, a, b, outdirpath)

        parapath = '_a' + str(a) + '_b' + str(b) + '_d' + str(r)
        original_array = np.loadtxt(outdirpath + '/4. vc' + parapath + '.txt', dtype=np.float32)
        scenes_list = construct_scene_from_array(array=original_array, thr=thr, min_construct_scene=min_construct_scene)
        thrVal_str = '%.2f' % thr
        saveSense(scenes_list, fno=outdirpath + '/scene_thr' + thrVal_str + '.txt')

        scene = load_ground_truth(outdirpath + '/sense_merge_allow_subset_delete_single.txt')
        NUM_CID = num_cids[k]
        a = average_f1(predictions=scene, groundtruths=scenes_list)
        b = omega_index(predictions=scene, groundtruths=scenes_list, n=num_cids[k])
        c = nmi_lfk(X=scene, Y=scenes_list)
        print('average_f1:%f\tomega_index:%f\tnmi:%f' % (a, b, c))



if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(end - start)

