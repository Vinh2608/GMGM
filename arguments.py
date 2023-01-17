import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_fpath", help="file path of dude data", type=str, default='./datasets/iuphar/processed')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = './ckpts')
parser.add_argument("--log_dir", help="logging directory", type=str, default = './logs')
parser.add_argument("--result_dir", help="result directory", type=str, default = './results')

# General training configurations
parser.add_argument("--ckpt", help="Load ckpt file", type=str, default = "")
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epochs", help="epochs", type=int, default = 5)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 1)
parser.add_argument("--num_workers", help="number of workers", type=int, default = os.cpu_count())

# Active Leanring
parser.add_argument("--ensemble_size", help="number of ensemble models", type=int, default = 4)
parser.add_argument("--n_iter", help="global iterations", type=int, default = 1000)
parser.add_argument("--test_interval", help="number of epochs between testing", type=int, default = 5)
parser.add_argument("--sampling", help="sampling complexes for subset training data", type=int, default = 100)
parser.add_argument("--unobserved_ratio", help="ratio of unobserved families", type=float, default = 0.95)
parser.add_argument("--n_acq", help="number of complexes to be accquire each iteration", type=int, default = 50)

# Model parameters
parser.add_argument("--tatic", help="tactic of defining number of hops", type=str, default = "static", choices=["static", "cont", "jump"])
parser.add_argument("--nhop", help="number of hops", type=int, default = 3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 72)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 64)
parser.add_argument("--n_att_heads", help="number of attention heads", type=int, default = 1)

parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

# Dataset configurations
parser.add_argument("--train_keys", help="train keys", type=str, default='./datasets/iuphar/keys/train_keys.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='./datasets/iuphar/keys/test_keys.pkl')
parser.add_argument("--n_families", help="number of receptors", type=int, default=150)

args = parser.parse_args()