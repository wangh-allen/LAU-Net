import argparse
import utility
import numpy as np

parser = argparse.ArgumentParser(description='LAU')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_dir', type=str, default='./dataset',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='odisr',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',
                    help='test dataset name')
parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--pre_train', type=str, default=r'./results/model/model_best.pt',
                    help='pre-trained model directory')
parser.add_argument('--n_blocks', type=int, default=8,
                    help='number of CA Dense blocks in each level')
parser.add_argument('--n_evaluator', type=int, default=12,
                    help='number of evaluators in the network')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--negval', type=float, default=0.2,
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration, L1|MSE')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--save', type=str, default='./results/',
                    help='file name to save')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

args = parser.parse_args()
args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

