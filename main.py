import numpy as np
import pandas as pd
import torch
import opt
from utils import post_proC, print_metrics, set_seed
from model import AllModel
from evaluation import eval
from load_data import load_data
import tqdm
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import networkx as nx

def build_spatial_knn(adata, k=10):

    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")

    coords = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    n = coords.shape[0]
    A = np.zeros((n, n))

    for i in range(n):
        A[i, indices[i]] = 1

    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    return csr_matrix(A)


def build_dgdi_graph(S, target_degree):

    S = (S - S.min()) / (S.max() - S.min() + 1e-8)

    n = S.shape[0]
    total_edges = int(n * target_degree / 2)

    triu_idx = np.triu_indices(n, k=1)
    values = S[triu_idx]

    threshold = np.sort(values)[-total_edges]

    A = np.zeros_like(S)
    A[S >= threshold] = 1

    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    return csr_matrix(A)


def graph_statistics(A1, A2):

    G1 = nx.from_scipy_sparse_array(A1)
    G2 = nx.from_scipy_sparse_array(A2)

    edges1 = set(G1.edges())
    edges2 = set(G2.edges())

    overlap = len(edges1 & edges2)
    union = len(edges1 | edges2)

    print("Spatial edges:", G1.number_of_edges())
    print("DGDI edges:", G2.number_of_edges())
    print("Edge overlap ratio:", overlap / union)

    from networkx.algorithms.community import greedy_modularity_communities

    comm1 = greedy_modularity_communities(G1)
    comm2 = greedy_modularity_communities(G2)

    mod1 = nx.algorithms.community.quality.modularity(G1, comm1)
    mod2 = nx.algorithms.community.quality.modularity(G2, comm2)

    print("Spatial modularity:", mod1)
    print("DGDI modularity:", mod2)

if __name__ == '__main__':

    set_seed(seed=opt.args.seed)

    # Load data
    X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, label, adj_spatial_omics1, adj_spatial_omics2, a_omics1, data, raw_data = load_data()

    opt.args.n_omics1 = X_omics1.shape[1]
    opt.args.n_omics2 = X_omics2.shape[1]
    opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 1, 3

    if opt.args.name == 'Human_tonsil':
        opt.args.n_cluster = 7
        label = None
    else:
        opt.args.n_cluster = len(np.unique(label))

    print("=" * 10 + " Pretraining has begun! " + "=" * 10)

    model = AllModel(X_omics2.shape[0]).cuda(opt.args.device)
    optimizer0 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(31), ncols=200)

    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2)

        pretrain_loss = loss_rec
        optimizer0.zero_grad()
        pretrain_loss.backward()
        optimizer0.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(opt.args.pretrain_epoch + 1), ncols=200)
    for epoch in pbar:
        loss_rec = model.forward1(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2)

        pretrain_loss = loss_rec
        optimizer.zero_grad()
        pretrain_loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.train_lr)
    pbar2 = tqdm.tqdm(range(900 + 1), ncols=200)
    for epoch in pbar2:

        loss_rec, loss_selfExp, S = model.forward2(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2,
                                                           adj_spatial_omics1, adj_spatial_omics2)

        total_loss =  loss_rec + loss_selfExp

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        pbar2.set_postfix({'loss': '{0:1.4f}'.format(total_loss)})
        if epoch % 900 == 0 and epoch != 0:
            S_cpu = S.cpu().detach().numpy()
            pred, _ = post_proC(S_cpu, opt.args.n_cluster)

            if label is not None:
                metrics = eval(label, pred)  
                # data.uns['clustering_metrics'] = metrics
            print("===== Dynamic Graph Comparison =====")

            # Spatial graph
            A_spatial = build_spatial_knn(data, k=10)
            avg_degree = np.array(A_spatial.sum(axis=1)).mean()

            A_dgdi = build_dgdi_graph(S_cpu, target_degree=int(avg_degree))

            graph_statistics(A_spatial, A_dgdi)

            print("===== Exporting Expression + Clusters =====")

            pred_str = pred.astype(str)

            if hasattr(raw_data.X, "toarray"):
                expr = raw_data.X.toarray()
            else:
                expr = raw_data.X

            expr_df = pd.DataFrame(
                expr,
                index=raw_data.obs_names,
                columns=raw_data.var_names
            )

            meta_df = pd.DataFrame(
                {"clusters": pred_str},
                index=raw_data.obs_names
            )

            expr_df.to_csv("./results/MBE15/spatial_expression_matrix.csv")
            meta_df.to_csv("./results/MBE15/spatial_metadata.csv")

            expr_df.to_csv("./results/MBE15/dgdi_expression_matrix.csv")
            meta_df.to_csv("./results/MBE15/dgdi_metadata.csv")

            print("CSV files saved for Seurat.")



