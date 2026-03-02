# -*- coding:utf-8 -*-
import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch.backends import cudnn
import episcanpy.api as epi
import ot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.special import softmax
from anndata import AnnData
from scipy.linalg import block_diag



def preprocess(adata_omics1, adata_omics2, datatype='10x'):
    # configure random seed
    random_seed = 2022
    fix_seed(random_seed)

    if datatype not in ['10x', 'MISAR', 'Simulation']:
        raise ValueError(
            "The datatype is not supported now. SMODICS supports '10x', 'MISAR'. We would extend SpaMICS for more data types. ")

    if datatype == '10x':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=100)

        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars)

    elif datatype == 'MISAR':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        # ATAC
        epi.pp.filter_features(adata_omics2, min_cells=int(adata_omics2.shape[0] * 0.06))  # 0.05
        print(adata_omics2.X.shape)
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=21)

        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

    return [adata_omics1, adata_omics2]


def pca(adata, use_reps=None, n_comps=10):
    """修改后的PCA函数：保留载荷矩阵、中心化均值，存入adata的uns和varm中"""
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix

    pca_model = PCA(n_components=n_comps)

    # 1. 提取输入数据（3000维标准化后的基因表达矩阵）
    if use_reps is not None:
        input_data = adata.obsm[use_reps]
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            input_data = adata.X.toarray()
        else:
            input_data = adata.X  # 此时adata.X已完成scale，是3000维标准化后的数据

    # 2. 执行PCA并保留结果
    feat_pca = pca_model.fit_transform(input_data)

    # 3. 存储反向PCA所需的关键参数到adata.uns和varm中
    adata.uns['pca_params'] = {
        'mean_': pca_model.mean_,  # PCA中心化时使用的均值（3000维，对应每个基因的均值）
        'components_': pca_model.components_,  # 载荷矩阵（150×3000，每行=1个主成分，每列=1个基因的系数）
        'n_comps': n_comps
    }
    adata.varm['pca_loadings'] = pca_model.components_.T  # 转置为3000×150，方便后续计算

    # 4. 存储预处理阶段的关键参数（用于还原基因表达量）
    # （1）sc.pp.scale的均值和标准差（scale后的数据满足均值=0，标准差=1，需保留原始均值/标准差）
    adata.uns['scale_params'] = {
        'mean': adata.X.mean(axis=0).reshape(1, -1),  # 3000维，每个基因的原始均值（scale前）
        'std': adata.X.std(axis=0).reshape(1, -1)  # 3000维，每个基因的原始标准差（scale前）
    }
    # （2）sc.pp.log1p的标记（log1p是log(1+x)，反向是expm1(x)）
    adata.uns['preprocess_steps'] = ['normalize_total', 'log1p', 'scale']

    return feat_pca


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    # X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    # adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class LoadBatch10xAdata:
    def __init__(self, dataset_path: str, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
                 image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
        self.dataset_path = dataset_path  # until dataset path (like ./Dataset/DLPFC)
        self.file_list = file_list  # slice name list
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata_list = []
        self.adata_len = []
        self.merged_adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.do_log = do_log

    def construct_interaction(self, input_adata):
        input_adata.var_names_make_unique()
        sc.pp.highly_variable_genes(input_adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(input_adata, target_sum=1e4)
        sc.pp.log1p(input_adata)
        sc.pp.scale(input_adata)

        adata_Vars = input_adata[:, input_adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        input_adata.obsm['feat'] = feat

        position = input_adata.obsm['spatial']
        feature = input_adata.obsm['feat']

        distance_matrix = ot.dist(position, position, metric='euclidean')
        feature_matrix = ot.dist(feature, feature, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        feature_interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1

        for i in range(n_spot):
            vec = feature_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.n_neighbors + 1):
                y = distance[t]
                feature_interaction[i, y] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)

        adj_feature = interaction + interaction.T
        adj_feature = np.where(adj_feature > 1, 1, adj_feature)

        input_adata.obsm['local_graph'] = adj
        input_adata.obsm['local_feature_graph'] = adj
        return input_adata

    def load_data(self):
        for i in self.file_list:
            print('now load: ' + i)
            load_path = os.path.join(self.dataset_path, i)
            adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
            adata.var_names_make_unique()
            if self.label:
                df_meta = pd.read_csv(os.path.join(load_path, 'truth.txt'), sep='\t', header=None)
                df_meta_layer = df_meta[1]
                adata.obs['ground_truth'] = df_meta_layer.values
                print(i + ' load label done')
                if self.filter_na:
                    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
                    print(i + ' filter NA done')
            if self.image_emb:
                data = np.load(os.path.join(load_path, 'embeddings.npy'))
                data = data.reshape(data.shape[0], -1)
                scaler = StandardScaler()
                embedding = scaler.fit_transform(data)
                pca = PCA(n_components=128, random_state=42)
                embedding = pca.fit_transform(embedding)
                adata.obsm['img_emb'] = embedding
                print(i + ' load img embedding done')
            adata = self.construct_interaction(input_adata=adata)
            print(i + ' build local graph done')
            self.adata_list.append(adata)
            self.adata_len.append(adata.X.shape[0])
            print(i + ' added to list')
        print('load all slices done')

        return self.adata_list

    def concatenate_slices(self):
        adata = AnnData.concatenate(*self.adata_list, join='outer')

        self.merged_adata = adata
        print('merge done')
        return self.merged_adata

    def construct_whole_graph(self):
        matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
        adjacency = block_diag(*matrix_list)
        self.merged_adata.obsm['graph_neigh'] = adjacency

        feature_matrix_list = [i.obsm['local_feature_graph'] for i in self.adata_list]
        fearure_adjacency = block_diag(*feature_matrix_list)
        self.merged_adata.obsm['feature_neigh'] = fearure_adjacency
        return self.merged_adata

    def calculate_edge_weights(self):
        # 获取现有的邻接矩阵和节点 embedding
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['img_emb']

        # 计算所有节点之间的欧氏距离
        euclidean_distances = cdist(node_emb, node_emb, metric='euclidean')

        # 计算邻边权重矩阵
        edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)

        # 将邻边权重转换为概率（用 softmax 函数）
        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(edge_weights.shape[0]):
            # edge_probabilities[i] = softmax(-edge_weights[i]) # 注意：这里用负号，使得距离近的节点具有较低的概率被删除。
            non_zero_indices = edge_weights[i] != 0
            non_zero_weights = np.log(edge_weights[i][non_zero_indices] + 1)  # 使用对数函数进行缩放，+1是为了避免对零取对数
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i][non_zero_indices] = softmax_weights
        # 将概率矩阵存储到 adata
        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        self.concatenate_slices()
        self.construct_whole_graph()
        if self.image_emb:
            self.calculate_edge_weights()
        return self.merged_adata
