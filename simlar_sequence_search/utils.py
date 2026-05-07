import math
import numpy as np

def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[ np.nonzero(l2 < 0) ] = 0.0
    return np.sqrt(l2)

def cos_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == x.shape[1]

    x = x.T
    sqr_q = np.sqrt(np.sum(q ** 2, axis=1, keepdims=True))
    sqr_x = np.sqrt(np.sum(x ** 2, axis=0, keepdims=True))

    cos = q @ x / sqr_q / sqr_x

    return 1.0 - cos

def arg_sort(q, x):
    dists = cos_dist(q, x)
    return np.argsort(dists)

def intersect(gs, ids):
    return np.mean([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])

def intersect_sizes(gs, ids):
    return np.array([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])


def test_recall(X, Q, G):
    ks = [1, 5, 10, 50]
    Ts = [1, 5, 10, 50]
    ks2 = [1, 10]
    Ts2 = [2 ** i for i in range(2 + int(math.log2(len(X))))]

    sort_idx = arg_sort(Q, X)

    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        tps = [intersect_sizes(G[:, :top_k], ids) / float(top_k) for top_k in ks]
        rcs = [np.mean(t) for t in tps]
        vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()

    print("# Probed \t Items \t", end="")
    for top_k in ks2:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts2:
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        tps = [intersect_sizes(G[:, :top_k], ids) / float(top_k) for top_k in ks2]
        rcs = [np.mean(t) for t in tps]
        vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()

def test_recall2(D, G):
    ks = [1, 5, 10, 50]
    Ts = [1, 5, 10, 50]
    ks2 = [1, 10]
    Ts2 = [2 ** i for i in range(17)]

    #sort_idx = arg_sort(Q, X)
    print(D.shape)
    sort_idx = np.argsort(D)
    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        tps = [intersect_sizes(G[:, :top_k], ids) / float(top_k) for top_k in ks]
        rcs = [np.mean(t) for t in tps]
        vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()

    print("# Probed \t Items \t", end="")
    for top_k in ks2:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts2:
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        tps = [intersect_sizes(G[:, :top_k], ids) / float(top_k) for top_k in ks2]
        rcs = [np.mean(t) for t in tps]
        vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()