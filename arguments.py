import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_fpath", help="file path of dude data", type=str, default='./datasets/iuphar/processed')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = './src/GMGM/save/')
parser.add_argument("--log_dir", help="logging directory", type=str, default = './src/GMGM/log/')

parser.add_argument("--ckpt", help="Load ckpt file", type=str, default = "")
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epochs", help="epochs", type=int, default = 10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 2)
parser.add_argument("--num_workers", help="number of workers", type=int, default = os.cpu_count())

parser.add_argument("--tatic", help="tactic of defining number of hops", type=str, default = "static", choices=["static", "cont", "jump"])
parser.add_argument("--nhop", help="number of hops", type=int, default = 3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)

parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

parser.add_argument("--train_keys", help="train keys", type=str, default='./datasets/iuphar/keys/train_keys.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='./datasets/iuphar/keys/test_keys.pkl')
parser.add_argument("--n_recpts", help="number of receptors", type=int, default=None)

args = parser.parse_args()