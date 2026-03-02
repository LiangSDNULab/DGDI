import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import sklearn.neighbors
import pandas as pd
import scipy.sparse as sp
import opt
import scanpy as sc
import torch.nn as nn
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn import cluster
from termcolor import colored
from processing import pca
from scipy.sparse import coo_matrix, csr_matrix

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def reconstruction_loss(X, A_norm, X_hat, A_hat):
    loss_w = F.mse_loss(X_hat, torch.mm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm)
    loss_igae = loss_w + opt.args.alpha_value * loss_a
    return loss_igae


def cos_sim(A, B):
    # cosine similarity
    A_norm = torch.norm(A, dim=1, keepdim=True)  # L2
    B_norm = torch.norm(B, dim=1, keepdim=True)  # L2
    similarity_matrix = torch.matmul(A, B.T) / (torch.matmul(A_norm, B_norm.T) + 1e-8)
    return similarity_matrix


def contrastive_loss(A, B, adj_graph, temperature):
    similarity_matrix1 = torch.exp(cos_sim(A, B) / temperature)
    positive_score_array1 = torch.sum(similarity_matrix1 * adj_graph[0], dim=1)
    negative_score_array1 = torch.sum(similarity_matrix1 * adj_graph[1], dim=1)

    similarity_matrix2 = torch.exp(cos_sim(A, A) / temperature)
    similarity_matrix2 = similarity_matrix2 - torch.diag(torch.diag(similarity_matrix2))
    positive_score_array2 = torch.sum(similarity_matrix2 * adj_graph[0], dim=1)
    negative_score_array2 = torch.sum(similarity_matrix2 * adj_graph[1], dim=1)

    positive_score_array = positive_score_array1 + positive_score_array2
    negative_score_array = negative_score_array1 + negative_score_array2
    positive_score_array = positive_score_array[positive_score_array > 0]
    negative_score_array = negative_score_array[negative_score_array > 0]

    positive_score = torch.log(positive_score_array).sum()
    negative_score = torch.log(negative_score_array).sum()
    constract_loss = -(positive_score - negative_score)
    return constract_loss


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def construct_graph(count, k=10, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()

    adj = (adj.T + adj) / 2
    adj_n = norm_adj(adj)

    return adj, adj_n


def refine_adj_spatial(feature_graph, spatial_graph):
    mask = np.logical_and(feature_graph > 0, spatial_graph > 0)
    spatial_graph_refine = np.where(mask, spatial_graph, 0)

    return norm_adj(spatial_graph_refine)


def convert_to_tensor(arrays, device=opt.args.device):
    return [torch.FloatTensor(arr).to(device) for arr in arrays]


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=11, alpha=4):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def print_metrics(ACC, F1, NMI, ARI, AMI):
    metrics_str = f"| ACC: {ACC:.4f} | NMI: {NMI:.4f} | ARI: {ARI:.4f} | F1: {F1:.4f} | AMI: {AMI:.4f} |"
    border = "=" * len(metrics_str)

    colored_border = colored(border, 'white', attrs=['bold'])
    colored_metrics = colored(metrics_str, 'red', attrs=['bold'])

    print(colored_border)
    print(colored_metrics)
    print(colored_border)


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, use_global=False):
    if verbose:
        print('------Calculating spatial graph...')
    if use_global:
        coor = pd.DataFrame(adata.obsm['spatial_global'])
    else:
        coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.close('all')


def get_negative_graph(positive_pairs_graph, n_samples):
    negative_pairs_graph = np.ones((n_samples, n_samples))
    negative_pairs_graph[positive_pairs_graph > 0] = 0
    return negative_pairs_graph


def reformulate_positive_graph(positive_pairs_graph, n_samples):
    positive_pairs_graph = positive_pairs_graph - np.eye(n_samples)
    positive_pairs_graph = positive_pairs_graph / np.sum(positive_pairs_graph, axis=0)
    positive_pairs_graph = positive_pairs_graph + np.eye(n_samples)
    return positive_pairs_graph


def fused_adj_graph(positive_adj_graphs, n_samples, n_views):
    fused_positive_pairs_graph = np.zeros((n_samples, n_samples))
    for i in range(n_views):
        if isinstance(positive_adj_graphs[i], torch.Tensor):
            # 将张量移动到 CPU 并转换为 NumPy 数组
            positive_adj_graphs[i] = positive_adj_graphs[i].cpu().numpy()
        fused_positive_pairs_graph = np.maximum(positive_adj_graphs[i], fused_positive_pairs_graph)

    fused_positive_pairs_graph[fused_positive_pairs_graph > 0] = 1
    fused_positive_pairs_graph = reformulate_positive_graph(fused_positive_pairs_graph, n_samples)
    fused_negative_pairs_graph = get_negative_graph(fused_positive_pairs_graph, n_samples)

    adj_graph = np.stack((fused_positive_pairs_graph.T, fused_negative_pairs_graph.T), axis=0)
    return adj_graph


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0,
               increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """

    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == 'mclust':
        if use_pca:
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res