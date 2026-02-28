import argparse

parser = argparse.ArgumentParser(description='SpaMICS', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# setting
parser.add_argument('--name', type=str, default="Human_Lympha_Node_A1")
parser.add_argument('--show', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--pretrain_epoch', type=int, default=300)
parser.add_argument('--alpha_value', type=float, default=1)
parser.add_argument('--num_views', type=int, default=2, help='number of omics')

# parameters
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--spatial_k', type=int, default=10)
parser.add_argument('--pretrain_lr', type=float, default=1e-3)
parser.add_argument('--train_lr', type=float, default=1e-3)

# dimension of input and latent representations
parser.add_argument('--n_omics1', type=int, default=100)
parser.add_argument('--n_omics2', type=int, default=31)
parser.add_argument('--gcn_z', type=int, default=20)
parser.add_argument('--ranks', type=int, default=70)

parser.add_argument('--lambda_1', type=float, default=0.2)
parser.add_argument('--lambda_2', type=float, default=0.5)
parser.add_argument('--lambda_3', type=float, default=0.5)
parser.add_argument('--beta_1', type=float, default=0.0001)
parser.add_argument('--beta_2', type=float, default=0.005)
parser.add_argument('--device', type=str, default='cuda:0')

# GAE structure parameters
parser.add_argument('--gae_n_enc_1', type=int, default=800)
parser.add_argument('--gae_n_enc_2', type=int, default=400)
parser.add_argument('--gae_n_dec_1', type=int, default=400)
parser.add_argument('--gae_n_dec_2', type=int, default=800)
parser.add_argument('--dropout', type=float, default=0.3)

parser.add_argument('--slices', nargs='+', help='切片名称列表')
parser.add_argument('--output', default='./results/integration_results', help='输出目录')

args = parser.parse_args()
