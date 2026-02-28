import torch.autograd
import torch.nn as nn
from torch.cuda import device
import opt
from utils import he_init_weights, reconstruction_loss, contrastive_loss, fused_adj_graph
import torch.nn.functional as F


class AllModel(nn.Module):
    def __init__(self, num_sample):
        super().__init__()

        """
        A model for spatial domain identification in spatial multi-omics.

        The model has three training stages:
        - forward: Pre-training of GAEs(no fusion).
        - forward1: Pre-training of GAEs and fusion layers.
        - forward2: Full model training.
    
        Parameters
        ----------
        num_sample : int
            The number of samples in the dataset

        Attributes
        ----------
        gae_feature_omics1 : GAE
            Graph autoencoder for feature representation of omics1(future graph).
        gae_feature_omics2 : GAE
            Graph autoencoder for feature representation of omics2(future graph).
        gae_spatial_omics1 : GAE
            Graph autoencoder for spatial representation of omics1(spatial graph).
        gae_spatial_omics2 : GAE
            Graph autoencoder for spatial representation of omics2(spatial graph).
        D_omics1 : nn.Parameter
            Self-representation matrix for omics1, learnable parameter.
        D_omics2 : nn.Parameter
            Self-representation matrix for omics2, learnable parameter.
        C1 : nn.Parameter
            Low-rank matrix shared across omics, learnable parameter.
        fusion_1 : WeightFusion
            Fusion layer for feature and spatial representations of omics1.
        fusion_2 : WeightFusion
            Fusion layer for feature and spatial representations of omics2.
        beta_1 : float
            Weight factor for reconstruction loss of omics1.
        beta_2 : float
            Weight factor for reconstruction loss of omics2.
        s : nn.Sigmoid
            Sigmoid activation function for adjacency matrix calculation.

        Returns
        -------
        loss_rec : FloatTensor
            The reconstruction loss.
        loss_selfExp : FloatTensor
            All losses caused by the self-expression layer
        S : Tensor
            The combined similarity matrix.
        """

        self.num_sample = num_sample

        # Initialize graph autoencoders (GAE) for spatial multi-omics data
        self.gae_feature_omics1 = build_gae(n_input=opt.args.n_omics1, num_sample=num_sample)
        self.gae_feature_omics2 = build_gae(n_input=opt.args.n_omics2, num_sample=num_sample)
        self.gae_spatial_omics1 = build_gae(n_input=opt.args.n_omics1, num_sample=num_sample)
        self.gae_spatial_omics2 = build_gae(n_input=opt.args.n_omics2, num_sample=num_sample)

        # self.self_expression = SelfExpression(self.num_sample)

        # Initialize the private self-representation matrix (D_omics1、D_omics2) for each omics.
        self.D_omics1 = nn.Parameter(1.0e-4 * torch.ones(num_sample, num_sample), requires_grad=True)
        self.D_omics2 = nn.Parameter(1.0e-4 * torch.ones(num_sample, num_sample), requires_grad=True)

        # Initialize the low-rank matrix (C1), shared across omics.
        self.C1 = nn.Parameter(1.0e-4 * torch.ones(num_sample, opt.args.ranks), requires_grad=True)

        # Weight fusion layers for feature-latent and spatial-latent integration
        self.fusion_1 = CAM(n_input=opt.args.n_omics1, num_sample=num_sample)
        self.fusion_2 = CAM(n_input=opt.args.n_omics2, num_sample=num_sample)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=opt.args.gcn_z, nhead=4, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)

        # Initialize weights using He initialization
        self.apply(he_init_weights)

        # Balance parameter for omics reconstruction loss
        self.beta_1, self.beta_2 = self.compute_weights(opt.args.n_omics1, opt.args.n_omics2)
        self.temperature = 1

        self.s = nn.Sigmoid()

    def forward(self, omics_1, omics_2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1, adj_spatial_omics2):

        """
                The forward pass of the model.
                Pretrain GAEs (no fusion)
                Stage 1: Pretrain GAEs + Fusion layers
                Stage 2: Full model training
        """

        z_omics1_feature, adj_omics1_feature_hat = self.gae_feature_omics1.encoder(omics_1, adj_feature_omics1)
        z_omics2_feature, adj_omics2_feature_hat = self.gae_feature_omics2.encoder(omics_2, adj_feature_omics2)
        z_omics1_spatial, adj_omics1_spatial_hat = self.gae_spatial_omics1.encoder(omics_1, adj_spatial_omics1)
        z_omics2_spatial, adj_omics2_spatial_hat = self.gae_spatial_omics2.encoder(omics_2, adj_spatial_omics2)

        loss_rec = self._reconstruction_loss_only(omics_1, omics_2, adj_feature_omics1, adj_feature_omics2,
                                                    adj_spatial_omics1, adj_spatial_omics2, z_omics1_feature,
                                                    adj_omics1_feature_hat,
                                                    z_omics2_feature, adj_omics2_feature_hat, z_omics1_spatial,
                                                    adj_omics1_spatial_hat, z_omics2_spatial, adj_omics2_spatial_hat)

        return loss_rec

    def forward1(self, omics_1, omics_2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1, adj_spatial_omics2):


        """
                The forward pass of the model.
                Pretrain GAEs + Fusion layers
        """

        z_omics1_feature, adj_omics1_feature_hat = self.gae_feature_omics1.encoder(omics_1, adj_feature_omics1)
        z_omics2_feature, adj_omics2_feature_hat = self.gae_feature_omics2.encoder(omics_2, adj_feature_omics2)
        z_omics1_spatial, adj_omics1_spatial_hat = self.gae_spatial_omics1.encoder(omics_1, adj_spatial_omics1)
        z_omics2_spatial, adj_omics2_spatial_hat = self.gae_spatial_omics2.encoder(omics_2, adj_spatial_omics2)

        H_omics1 = self.fusion_1(z_omics1_feature, z_omics1_spatial, adj_feature_omics1, adj_spatial_omics1, omics_1)
        H_omics2 = self.fusion_2(z_omics2_feature, z_omics2_spatial, adj_feature_omics2, adj_spatial_omics2, omics_2)

        adj_fusion_hat_omics1 = self.s(torch.mm(H_omics1, H_omics1.t()))
        adj_fusion_hat_omics2 = self.s(torch.mm(H_omics2, H_omics2.t()))

        loss_rec = self._reconstruction_loss_only(omics_1, omics_2, adj_feature_omics1, adj_feature_omics2,
                                                    adj_spatial_omics1, adj_spatial_omics2, H_omics1, adj_fusion_hat_omics1, H_omics2,
                                                    adj_fusion_hat_omics2, H_omics1, adj_fusion_hat_omics1, H_omics2, adj_fusion_hat_omics2)
        return loss_rec

    def forward2(self, omics_1, omics_2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1, adj_spatial_omics2):


        """
                The forward pass of the model.
                Full model training
        """
        adj_feature = []
        adj_feature.append(adj_feature_omics1)
        adj_feature.append(adj_feature_omics2)

        fused_adj = fused_adj_graph(adj_feature, self.num_sample, 2)
        adj_graph = torch.tensor(fused_adj, dtype=torch.float32, device=opt.args.device)

        z_omics1_feature, adj_omics1_feature_hat = self.gae_feature_omics1.encoder(omics_1, adj_feature_omics1)
        z_omics2_feature, adj_omics2_feature_hat = self.gae_feature_omics2.encoder(omics_2, adj_feature_omics2)
        z_omics1_spatial, adj_omics1_spatial_hat = self.gae_spatial_omics1.encoder(omics_1, adj_spatial_omics1)
        z_omics2_spatial, adj_omics2_spatial_hat = self.gae_spatial_omics2.encoder(omics_2, adj_spatial_omics2)

        H_omics1 = self.fusion_1(z_omics1_feature, z_omics1_spatial, adj_feature_omics1, adj_spatial_omics1, omics_1)
        H_omics2 = self.fusion_2(z_omics2_feature, z_omics2_spatial, adj_feature_omics2, adj_spatial_omics2, omics_2)

        # 在第三维拼接（深度拼接）
        CH_omics = torch.stack((H_omics1, H_omics2), dim=2)  # 形状: [batch, num_features, 2]
        CH_omics = CH_omics.permute(0, 2, 1)  # 调整维度顺序为 [batch, 2, num_features]

        # 通过Transformer（自动处理序列长度）
        AH_omics = self.TransformerEncoderLayer(CH_omics)  # 输出形状: [batch, 2, num_features]

        # 恢复原始维度顺序并拆分
        AH_omics = AH_omics.permute(0, 2, 1)  # 形状: [batch, num_features, 2]
        AH_omics1, AH_omics2 = AH_omics.chunk(2, dim=2)  # 沿第三维拆分为两个张量
        AH_omics1 = AH_omics1.squeeze(2)  # 移除第三维 [batch, num_features]
        AH_omics2 = AH_omics2.squeeze(2)  # 移除第三维 [batch, num_features]

        C = torch.mm(self.C1, self.C1.T)
        D_omics1 = self.D_omics1 - torch.diag(torch.diag(self.D_omics1))
        D_omics2 = self.D_omics2 - torch.diag(torch.diag(self.D_omics2))

        S_omics1 = D_omics1 + C
        S_omics2 = D_omics2 + C

        # # Apply the self-representation to the latent fusion features.
        SH_omics1 = torch.matmul(S_omics1, AH_omics1)
        SH_omics2 = torch.matmul(S_omics2, AH_omics2)
        # SH_omics1, SH_omics2, loss_selfExp, S = self.self_expression(AH_omics1, AH_omics2)
        # # S = S + (encoder1_c + encoder2_c + encoder3_c + encoder4_c) / 4

        adj_fusion_hat_omics1 = self.s(torch.mm(SH_omics1, SH_omics1.t()))
        adj_fusion_hat_omics2 = self.s(torch.mm(SH_omics2, SH_omics2.t()))

        loss_rec = self._reconstruction_loss_only(omics_1, omics_2, adj_feature_omics1, adj_feature_omics2,
                                                    adj_spatial_omics1, adj_spatial_omics2, SH_omics1, adj_fusion_hat_omics1,
                                                    SH_omics2, adj_fusion_hat_omics2, SH_omics1, adj_fusion_hat_omics1, SH_omics2,
                                                    adj_fusion_hat_omics2)

        loss_reg = self._regularization_loss(C, D_omics1, D_omics2)
        loss_self = self._self_expression_loss(SH_omics1, SH_omics2, AH_omics1, AH_omics2)
        loss_dis = self._discriminative_constraint(D_omics1, D_omics2)

        loss_contrastive = (contrastive_loss(SH_omics1, AH_omics1, adj_graph, self.temperature)
                     + contrastive_loss(SH_omics2, AH_omics2, adj_graph, self.temperature))

        loss_selfExp = opt.args.lambda_1 * loss_self + opt.args.lambda_2 * loss_reg + opt.args.lambda_3 * loss_dis + loss_contrastive * 0.0001

        S = C + (1 / opt.args.num_views) * (0.5 * (D_omics1 + D_omics1.T) + 0.5 * (D_omics2 + D_omics2.T))

        return loss_rec, loss_selfExp, S



    def _reconstruction_loss_only(self, omics_1, omics_2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                                  adj_spatial_omics2, z_omics1_feature, a_1, z_omics2_feature, a_2, z_omics1_spatial,
                                  a_3, z_omics2_spatial, a_4):
        """
        Compute the reconstruction loss for all omics.
        """

        X_omics1_feature_hat, A1_hat = self.gae_feature_omics1.decoder(z_omics1_feature, adj_feature_omics1)
        X_omics2_feature_hat, A2_hat = self.gae_feature_omics2.decoder(z_omics2_feature, adj_feature_omics2)
        X_omics1_spatial_hat, A3_hat = self.gae_spatial_omics1.decoder(z_omics1_spatial, adj_spatial_omics1)
        X_omics2_spatial_hat, A4_hat = self.gae_spatial_omics2.decoder(z_omics2_spatial, adj_spatial_omics2)

        Z_omics1_feature_hat, a1_hat = self.gae_feature_omics1.encoder(X_omics1_feature_hat, adj_feature_omics1)
        Z_omics2_feature_hat, a2_hat = self.gae_feature_omics2.encoder(X_omics2_feature_hat, adj_feature_omics2)
        Z_omics1_spatial_hat, a3_hat = self.gae_spatial_omics1.encoder(X_omics1_spatial_hat, adj_spatial_omics1)
        Z_omics2_spatial_hat, a4_hat = self.gae_spatial_omics2.encoder(X_omics2_spatial_hat, adj_spatial_omics2)


        loss_rec_omics1 = reconstruction_loss(omics_1, adj_feature_omics1, X_omics1_feature_hat, (A1_hat + a_1) / 2) + \
                          reconstruction_loss(omics_1, adj_spatial_omics1, X_omics1_spatial_hat, (A3_hat + a_3) / 2) + \
                          reconstruction_loss(z_omics1_feature, adj_feature_omics1, Z_omics1_feature_hat, (A1_hat + a1_hat) / 2) + \
                          reconstruction_loss(z_omics1_spatial, adj_spatial_omics1, Z_omics1_spatial_hat, (A3_hat + a3_hat) / 2)
        loss_rec_omics2 = reconstruction_loss(omics_2, adj_feature_omics2, X_omics2_feature_hat, (A2_hat + a_2) / 2) + \
                          reconstruction_loss(omics_2, adj_spatial_omics2, X_omics2_spatial_hat, (A4_hat + a_4) / 2) + \
                          reconstruction_loss(z_omics2_feature, adj_feature_omics2, Z_omics2_feature_hat, (A2_hat + a2_hat) / 2) + \
                          reconstruction_loss(z_omics2_spatial, adj_spatial_omics2, Z_omics2_spatial_hat, (A4_hat + a4_hat) / 2)

        return self.beta_1 * loss_rec_omics1 + self.beta_2 * loss_rec_omics2


    def _regularization_loss(self, C, D_omics1, D_omics2):

        """
        Compute the regularization loss for the shared matrix C and private self-representation matrices (D_omics1, D_omics2).
        """

        loss_reg = (1 / (opt.args.num_views + 1)) * (F.mse_loss(C,
                                                           torch.zeros(self.num_sample, self.num_sample,
                                                                       requires_grad=True).to(
                                                               opt.args.device), reduction='sum') +
                                                F.mse_loss(D_omics1,
                                                           torch.zeros(self.num_sample, self.num_sample,
                                                                       requires_grad=True).to(
                                                               opt.args.device), reduction='sum') +
                                                F.mse_loss(D_omics2,
                                                           torch.zeros(self.num_sample, self.num_sample,
                                                                       requires_grad=True).to(
                                                               opt.args.device), reduction='sum'))

        return loss_reg

    @staticmethod
    def _self_expression_loss(SZ_1, SZ_2, z_omics1, z_omics2):

        """
        Compute the self-expression loss between the self-represented omics latent features and the original latent features.
        """

        return (1 / opt.args.num_views) * F.mse_loss(SZ_1, z_omics1) + (1 / opt.args.num_views) * F.mse_loss(SZ_2, z_omics2)

    @staticmethod
    def _discriminative_constraint(D_omics1, D_omics2):

        """
        Apply a discriminative constraint between the omics-specific self-representation matrices.
        """

        return torch.norm(torch.mul(D_omics1, D_omics2).view(-1), p=1)


    def compute_weights(self, n_omics1, n_omics2):

        """
        Calculate the balance parameters for weighting the reconstruction losses for omics1 and omics2 based on feature dimensions
        """

        denominator = n_omics1 + n_omics2
        if denominator == 0:
            raise ValueError("The sum of n_omics1 and n_omics2 should not be zero.")
        return n_omics2 / denominator, n_omics1 / denominator

class GNNLayer(nn.Module):
    """
    A single GNN layer that performs a linear transformation of the input
    features followed by matrix multiplication with the adjacency matrix (adj).
    """

    def __init__(self, in_features, out_features, activation=nn.Tanh()):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.act = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights using Xavier initialization.
        """
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        """
        Forward pass through the GNN layer.
        - features: Input node features.
        - adj: Adjacency matrix representing graph structure.
        - apply_activation: Whether to apply the activation function.
        """
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        return output


class GAE_encoder(nn.Module):
    """
        Encoder part of the Graph Autoencoder.
        Encodes input features into a latent space representation using multiple GNN layers.
    """
    def __init__(self, gae_n_enc_1, gae_n_enc_2, n_input, n_z, dropout, num_sample):
        super(GAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, n_z)
        self.dropout = nn.Dropout(dropout)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        """
        Forward pass through the encoder.
        - x: Input features.
        - adj: Adjacency matrix.
        Returns latent space representation and reconstructed adjacency matrix.
        """

        z = self.gnn_1(x, adj, active=True)
        z = self.dropout(z)
        z = self.gnn_2(z, adj, active=True)
        z = self.dropout(z)
        z_igae = self.gnn_3(z, adj, active=False)

        # Reconstruct adjacency matrix from latent space representation
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj



class GAE_decoder(nn.Module):
    """
    Decoder part of the Graph Autoencoder.
    Reconstructs the input features from the latent space representation using GNN layers.
    """
    def __init__(self, gae_n_dec_1, gae_n_dec_2, n_input, n_z):
        super(GAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_5 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_6 = GNNLayer(gae_n_dec_2, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        """
        Forward pass through the decoder.
        - z_igae: Latent space representation.
        - adj: Adjacency matrix.
        Returns reconstructed features and reconstructed adjacency matrix.
        """

        z2 = self.gnn_4(z_igae, adj, active=True)
        z1 = self.gnn_5(z2, adj, active=True)
        z_hat = self.gnn_6(z1, adj, active=True)

        # Reconstruct adjacency matrix from decoded features
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class GAE(nn.Module):
    """
    Full Graph Autoencoder (GAE) model that includes both the encoder and decoder.
    """
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_dec_1, gae_n_dec_2, n_input, n_z, dropout, num_sample):
        super(GAE, self).__init__()
        self.encoder = GAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            n_input=n_input,
            n_z=n_z,
            dropout=dropout,
            num_sample=num_sample)

        self.decoder = GAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            n_input=n_input,
            n_z=n_z)


def build_gae(n_input, num_sample):
    """
    Helper function to construct a GAE model.
    - n_input: The dimensionality of input features.
    """
    try:
        return GAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            n_input=n_input,
            n_z=opt.args.gcn_z,
            dropout=opt.args.dropout,
            num_sample=num_sample
        ).to(opt.args.device)
    except AttributeError as e:
        raise ValueError(f"Missing argument in opt.args: {e}")


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = nn.Linear(n_mlp, opt.args.gcn_z)

    def forward(self, mlp_in):
        weight_output =self.wl(mlp_in)

        return weight_output


class CAM(nn.Module):
    def __init__(self, n_input, num_sample):
        super(CAM, self).__init__()
        self.gea = build_gae(n_input, num_sample)
        self.meta = nn.Parameter(torch.Tensor([0.1]))
        self.MLP_L = MLP_L(opt.args.gcn_z)
        self.MLP = nn.Sequential(
            nn.Linear(opt.args.gcn_z * 3, opt.args.gcn_z)
        )

    def forward(self, z_feature, z_spatial, adj_feature, adj_spatial, x):
        con_adj = self.meta * adj_feature + (1 - self.meta) * adj_spatial
        com , con_adj_hat = self.gea.encoder(x, con_adj)

        emb = torch.stack([z_feature, com, z_spatial], dim=1)
        a = self.MLP_L(emb)
        emb = F.normalize(a, p=2)

        emb = torch.cat((emb[:, 0].mul(z_feature), emb[:, 1].mul(com), emb[:, 2].mul(z_spatial)), 1)
        emb = self.MLP(emb)

        return emb


