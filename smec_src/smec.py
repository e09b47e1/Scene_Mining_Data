import os
import time
import utils
import numpy as np

from scipy.sparse.linalg import norm
from scipy.sparse import rand


def load_all_data(fnWii, fnWoo, fnWcc, fnWib, fnWic, fnidi, fnidb, fnidc):
    fidi = open(fnidi, 'r')
    fidb = open(fnidb, 'r')
    fidc = open(fnidc, 'r')

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

    Wii = utils.sparse_read(fnWii, len(total_sku_to_id), len(total_sku_to_id))
    Woo = utils.sparse_read(fnWoo, len(total_sku_to_id), len(total_sku_to_id))
    Wcc = utils.sparse_read(fnWcc, len(total_cid3_to_id), len(total_cid3_to_id))
    Wib = utils.sparse_read(fnWib, len(total_sku_to_id), len(total_brand_to_id))
    Wic = utils.sparse_read(fnWic, len(total_sku_to_id), len(total_cid3_to_id))

    return Wii, Woo, Wcc, Wib, Wic, total_sku_to_id, total_brand_to_id, total_cid3_to_id

# Line 8: Compute loss
def objective(Wii, Woo, Wcc, Wib, Wic, Hi, Ho, Hb, Hc, WiiT, WooT, WccT, WibT, WicT, a, b):
    HiT = utils.sparse_transpose(Hi)
    HoT = utils.sparse_transpose(Ho)
    HbT = utils.sparse_transpose(Hb)
    HcT = utils.sparse_transpose(Hc)

    res = 0
    res += 0.5 * ( np.square(norm(Wii, 'fro')) - 2 * utils.sparse_trace(WiiT.dot(Hi), HiT) + utils.sparse_trace(Hi.dot(HiT.dot(Hi)), HiT) )
    res += 0.5 * ( np.square(norm(Woo, 'fro')) - 2 * utils.sparse_trace(WooT.dot(Ho), HoT) + utils.sparse_trace(Ho.dot(HoT.dot(Ho)), HoT) )
    res += 0.5 * ( np.square(norm(Wcc, 'fro')) - 2 * utils.sparse_trace(WccT.dot(Hc), HcT) + utils.sparse_trace(Hc.dot(HcT.dot(Hc)), HcT) )

    res += 0.5 * ( np.square(norm(Wib, 'fro')) - 2 * utils.sparse_trace(WibT.dot(Hi), HbT) + utils.sparse_trace(Hb.dot(HiT.dot(Hi)), HbT) )
    res += 0.5 * ( np.square(norm(Wic, 'fro')) - 2 * utils.sparse_trace(WicT.dot(Hi), HcT) + utils.sparse_trace(Hc.dot(HiT.dot(Hi)), HcT) )
    res += 0.5 * ( np.square(norm(Wib, 'fro')) - 2 * utils.sparse_trace(WibT.dot(Ho), HbT) + utils.sparse_trace(Hb.dot(HoT.dot(Ho)), HbT) )
    res += 0.5 * ( np.square(norm(Wic, 'fro')) - 2 * utils.sparse_trace(WicT.dot(Ho), HcT) + utils.sparse_trace(Hc.dot(HoT.dot(Ho)), HcT) )

    res += a * 0.5 * (np.square(norm(Hi, 'fro')))
    res += a * 0.5 * (np.square(norm(Ho, 'fro')))
    res += a * 0.5 * (np.square(norm(Hb, 'fro')))
    res += a * 0.5 * (np.square(norm(Hc, 'fro')))
    res += b * utils.sparse_tl21_norm(Hc)

    return res

# Line 1-8: factorizing the adjacency matrices to obtain membership matices
def basic_nmf(Wii, Woo, Wcc, Wib, Wic, r, max_iter, stop_condition, a, b, outdirpath):
    parapath = '_a' + str(a) + '_b' + str(b)
    di, db = Wib.shape
    _, dc = Wic.shape

    # Line 1: Initialize representation matrices
    Hi = rand(di, r, density=1, format='csr', dtype='float')
    Ho = rand(di, r, density=1, format='csr', dtype='float')
    Hb = rand(db, r, density=1, format='csr', dtype='float')
    Hc = rand(dc, r, density=1, format='csr', dtype='float')

    WiiT = utils.sparse_transpose(Wii)
    WooT = utils.sparse_transpose(Woo)
    WccT = utils.sparse_transpose(Wcc)
    WibT = utils.sparse_transpose(Wib)
    WicT = utils.sparse_transpose(Wic)

    eps = 1e-10
    pdist = 1e8

    # Line 2: Iteration begins...
    for i in range(max_iter):

        # Line 3-4: Update representation whose type belong to user behaviors
        numerator = Wib.dot(Hb) + Wic.dot(Hc) + Wii.dot(Hi) + WiiT.dot(Hi)
        denominator = Hi.dot(utils.sparse_transpose(Hb).dot(Hb)) + a*Hi + Hi.dot(utils.sparse_transpose(Hc).dot(Hc)) + 2 * Hi.dot(utils.sparse_transpose(Hi).dot(Hi))
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        Hi = utils.sparse_elementwise_product(Hi, utils.sparse_elementwise_division(numerator, denominator))

        # Line 3-4: Update representation whose type belong to user behaviors
        numerator = Woo.dot(Ho) + WooT.dot(Ho) + Wib.dot(Hb) + Wic.dot(Hc)
        denominator = 2 * Ho.dot(utils.sparse_transpose(Ho).dot(Ho)) + Ho.dot(utils.sparse_transpose(Hb).dot(Hb)) + \
            Ho.dot(utils.sparse_transpose(Hc).dot(Hc)) + a*Ho
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        Ho = utils.sparse_elementwise_product(Ho, utils.sparse_elementwise_division(numerator, denominator))

        # Line 5-6: Update representation whose type belong to item attributes
        numerator = WibT.dot(Hi) + WibT.dot(Ho)
        denominator = Hb.dot(utils.sparse_transpose(Hi).dot(Hi)) + a*Hb + Hb.dot(utils.sparse_transpose(Ho).dot(Ho))
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        Hb = utils.sparse_elementwise_product(Hb, utils.sparse_elementwise_division(numerator, denominator))

        # Line 7: Update representation whose type is categories
        numerator = WicT.dot(Hi) + Wcc.dot(Hc) + WccT.dot(Hc) + WicT.dot(Ho)
        denominator = Hc.dot(utils.sparse_transpose(Hi).dot(Hi)) + a*Hc + 2 * Hc.dot(utils.sparse_transpose(Hc).dot(Hc)) \
                      + Hc.dot(utils.sparse_transpose(Ho).dot(Ho)) + b*utils.sparse_derivative_tl21_norm(Hc)
        numerator = utils.sparse_all_plus_eps(numerator)
        denominator = utils.sparse_all_plus_eps(denominator)
        numerator = utils.sparse_reset(numerator)
        denominator = utils.sparse_reset(denominator)
        Hc = utils.sparse_elementwise_product(Hc, utils.sparse_elementwise_division(numerator, denominator))


        if i % 25 == 0:

            # Line 8: Compute loss
            dist = objective(Wii, Woo, Wcc, Wib, Wic, Hi, Ho, Hb, Hc, WiiT, WooT, WccT, WibT, WicT, a, b)
            if i > 0:
                if 0 < pdist-dist < stop_condition:
                    break
            pdist = dist

    return Hc.toarray()

def save_scene(lists, fn):
    f = open(fn, 'w')
    for list in lists:
        flag = False
        for v in list:
            if flag:
                print('\t' + str(v), file=f, end='')
            else:
                flag = True
                print(str(v), file=f, end='')
        print(file=f)
    f.close()

def construct_scene_from_array(array, thr, fn):
    min_construct_scene = 1
    m, n = array.shape
    res = []
    for j in range(n):
        tl = []
        for i in range(m):
            if array[i][j] > thr:
                tl.append(i)
        if len(tl) >= min_construct_scene:
            res.append(tl)

    save_scene(res, fn)
    return res

def main():
    # Setting parameters
    max_iter = 156
    stop_condition = 0.01
    dimention = 16
    thr = 0.1
    a_para = 1.0
    b_para = 1.0
    dataset = 'baby_toy'

    print('SMEC: ' + 'dataset-' + str(dataset) + ', max_iter-'  + str(max_iter) + ', stop_condition-' + str(stop_condition) + ', dimention-' + str(dimention))

    # Input
    in_dir = '../in_' + dataset
    Wii, Woo, Wcc, Wib, Wic, total_sku_to_id, total_brand_to_id, total_cid3_to_id = \
        load_all_data(in_dir + '/item_item (co_viewed).txt', in_dir + '/item_item (co_bought).txt',
                      in_dir + '/category_category.txt', in_dir + '/item_brand.txt',
                      in_dir + '/item_cate.txt', in_dir + '/ID_item_to_id.txt', in_dir + '/ID_brand_to_id.txt',
                      in_dir + '/ID_cate_to_id.txt')
    print('Data has been load.')

    out_dir = '../out_' + dataset
    if os.path.exists(out_dir):
        pass
    else:
        os.mkdir(out_dir)
    parapath = '_a' + str(a_para) + '_b' + str(b_para)
    print(dataset + ' (' + parapath + ') begins...')
    start = time.process_time()

    # Line 1-8: factorizing the adjacency matrices to obtain membership matices
    Hc = basic_nmf(Wii, Woo, Wcc, Wib, Wic, dimention, max_iter, stop_condition, a_para, b_para, out_dir)

    # Line 9-14: selecting the candidate to generate the final scenes.
    construct_scene_from_array(Hc, thr, out_dir + '/Scene' + parapath + '.txt')

    end = time.process_time()
    print(dataset + ' (' + parapath + ') is done...\n' + 'RunningTime: %f' % (end - start))


if __name__ == "__main__":
    main()


