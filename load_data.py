from utils import construct_graph, refine_adj_spatial, convert_to_tensor
import scanpy as sc
import numpy as np
import opt


def load_data():
    """
        Load and preprocess data from specified datasets.

        Constructs feature graphs and spatial graphs, refines the spatial graphs,
        and returns PCA features, normalized graphs, and labels.

        Returns:
            tuple: Contains PCA features of each omics data, normalized feature graphs,
                   labels, and refined spatial graphs for both omics datasets.
    """

    if opt.args.name in ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1", "HLN"]:
        # 人类淋巴
        adata_omics1, adata_omics2, label = load_human_lymph_node(opt.args.name)
        # opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 10, 10
        datatype = '10x'
    elif opt.args.name in ["Human_tonsil_1", "Human_tonsil_3"]:
        # 人类扁桃体
        adata_omics1, adata_omics2, label = load_human_tonsil(opt.args.name)
        datatype = '10x'
        opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 2, 1
    elif opt.args.name in ["Mouse_Brain_E15", "Mouse_Brain_E18"]:
        adata_omics1, adata_omics2, label = load_mouse_brain(opt.args.name)
        datatype = 'MISAR'
        opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 1, 2
    else:
        raise ValueError(f"Dataset {opt.args.name} not supported")

    from processing import preprocess
    # Preprocessing data
    data = preprocess(adata_omics1, adata_omics2, datatype=datatype)
    X_pca_omics1, X_pca_omics2 = data[0].obsm['feat'].copy(), data[1].obsm['feat'].copy()
    a_omics1 = data[0].copy()

    # Feature Graph Construction
    adj_omics1, adj_norm_omics1 = construct_graph(X_pca_omics1, k=opt.args.k)
    adj_omics2, adj_norm_omics2 = construct_graph(X_pca_omics2, k=opt.args.k)

    # Spatial Graph Construction
    adj_spatial, adj_spatial_norm = construct_graph(adata_omics1.obsm['spatial'], k=opt.args.spatial_k)

    # # Refine the spatial graphs using the feature graphs
    adj_spatial_refine_omics1 = refine_adj_spatial(adj_omics1, adj_spatial)
    adj_spatial_refine_omics2 = refine_adj_spatial(adj_omics2, adj_spatial)

    # Convert to tensor and move to the correct device
    X_pca_omics1, X_pca_omics2 = convert_to_tensor([X_pca_omics1, X_pca_omics2])
    adj_norm_omics1, adj_norm_omics2, adj_spatial_refine_omics1, adj_spatial_refine_omics2 = convert_to_tensor(
        [adj_norm_omics1, adj_norm_omics2, adj_spatial_refine_omics1, adj_spatial_refine_omics2])

    if opt.args.show:
        print("-------------Details Of The Dataset------------")
        print("-----------------------------------------------")
        print("Dataset name         :", opt.args.name)
        print("Omics1 shape         :", X_pca_omics1.shape)
        print("Omics2 shape         :", X_pca_omics2.shape)
        if label is not None:
            print("Category num         :", max(label) - min(label) + 1)
            print("Category distribution:")
            for i in range(max(label + 1)):
                print("Label", i, end=":")
                print(len(label[np.where(label == i)]))
        print("-----------------------------------------------")

    return X_pca_omics1, X_pca_omics2, adj_norm_omics1, adj_norm_omics2, label, adj_spatial_refine_omics1, adj_spatial_refine_omics2, a_omics1, data[0], adata_omics1


def load_human_lymph_node(name):
    """
        Load human lymph node data from .h5ad files.

        Args:
            name (str): Dataset name indicating which files to load.

        Returns:
            tuple: Contains RNA and ADT AnnData objects and their labels.
    """
    if name in ["Human_Lymph_Node_A1"]:
        adata_rna = sc.read_h5ad(f'./data/10X/{name}/adata_RNA.h5ad')
        adata_adt = sc.read_h5ad(f'./data/10X/{name}/adata_ADT.h5ad')

        adata_rna.var_names_make_unique()
        adata_adt.var_names_make_unique()

        # label = np.loadtxt(f'./data/10X/{name}/GT_Mouse_Brain_E15labels.txt')

        label = np.load(f'./data/10X/{name}/label.npy')
    else:
        adata_rna = sc.read_h5ad(f'./data/10X/{name}/adata_rna.h5ad')
        adata_adt = sc.read_h5ad(f'./data/10X/{name}/adata_adt.h5ad')
        adata_rna.var_names_make_unique()
        adata_adt.var_names_make_unique()

        label = adata_rna.obs['final_annot']
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        labels_numeric = label_encoder.fit_transform(label)
        adata_rna.obs['lab_numeric'] = labels_numeric
        label = adata_rna.obs['lab_numeric']

    return adata_rna, adata_adt, label


def load_mouse_brain(name):
    """
    Load mouse brain data from .h5ad files.

    Args:
        name (str): Dataset name indicating which files to load.

    Returns:
        tuple: Contains RNA and ATAC AnnData objects and their labels (None for this dataset).
    """
    adata_rna = sc.read_h5ad(f'./data/MISAR/{name}/adata_rna.h5ad')
    # print(adata_rna.X.shape)
    adata_atac = sc.read_h5ad(f'./data/MISAR/{name}/adata_atac.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()

    label = adata_rna.obs['Combined_Clusters']
    print(adata_rna, adata_atac, adata_rna.obs['Combined_Clusters'])

    return adata_rna, adata_atac, label


def load_human_tonsil(name):
    """
    Load human tonsil data from .h5ad files.

    Args:
        name (str): Dataset name indicating which files to load.

    Returns:
        tuple: Contains RNA and ADT AnnData objects and their labels (None for this dataset).
    """
    adata_rna = sc.read_h5ad(f'./data/10X/{name}/adata_rna.h5ad')
    adata_adt = sc.read_h5ad(f'./data/10X/{name}/adata_adt.h5ad')
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()

    label = adata_rna.obs['final_annot']
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_numeric = label_encoder.fit_transform(label)
    adata_rna.obs['lab_numeric'] = labels_numeric
    print(adata_rna, adata_adt, adata_rna.obs['final_annot'])

    return adata_rna, adata_adt, adata_rna.obs['lab_numeric']

