# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse
from sklearn.metrics import mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score
from metric import jaccard, Dice, F_measure


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    # if num_class1 != numclass2:
    #     print('error')
    #     return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eval(label, pred):
    acc, f1 = cluster_acc(label, pred)
    print("")
    print(f"our         (ACC): {acc:.6f}")

    Our_Jaccard = jaccard(label, pred)
    print(f"our         jaccard: {Our_Jaccard:.6f}")

    Our_F_measure = F_measure(label, pred)
    print(f"our         F_measure: {Our_F_measure:.6f}")

    Our_mutual_info = mutual_info_score(pred, label)
    print(f"our         Mutual Information: {Our_mutual_info:.6f}")

    Our_nmi = nmi_score(label, pred, average_method='arithmetic')
    print(f"Our         (NMI): {Our_nmi:.6f}")

    Our_ami = adjusted_mutual_info_score(pred, label)
    print(f"Our         (AMI): {Our_ami:.6f}")

    Our_V = v_measure_score(pred, label)
    print(f"Our         V-measure: {Our_V:.6f}")

    Our_homogeneity = homogeneity_score(pred, label)
    Our_completeness = completeness_score(pred, label)
    print(f"Our         Homogeneity: {Our_homogeneity:.6f} Completeness: {Our_completeness:.6f}")

    Our_ari = adjusted_rand_score(pred, label)
    print(f"Our         (ARI): {Our_ari:.6f}")

    Our_fmi = fowlkes_mallows_score(pred, label)
    print(f"Our         (FMI): {Our_fmi:.6f}")


# Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
def eva_no_labels(z, y_pred):
    z = z.detach().cpu().numpy()
    asw = silhouette_score(z, y_pred)
    chi = calinski_harabasz_score(z, y_pred)
    dbi = davies_bouldin_score(z, y_pred)

    print("")
    print(f"Our         (ASW): {asw:.6f}")
    print(f"Our         (CHI): {chi:.6f}")
    print(f"Our         (DBI): {dbi:.6f}")
