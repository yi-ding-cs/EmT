from base.cross_validation import CrossValidation
from datasets.FACED import FACED
from base.utils import seed_all, set_gpu
import numpy as np
import os
import torch
import argparse
current_dir = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--ROOT', type=str, default=current_dir)
    parser.add_argument('--dataset', type=str, default='FACED')
    parser.add_argument('--data-path', type=str, default=r"D:\DingYi\Dataset\FACED")    # change this accordingly
    parser.add_argument('--subjects', type=int, default=123)
    parser.add_argument('--data-exist', action='store_true', default=False) # skip data preparation by default

    parser.add_argument('--fold-to-run', type=int, default=10)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V', 'D', 'L'])
    parser.add_argument('--segment', type=int, default=20)
    parser.add_argument('--overlap', type=float, default=0.8)
    parser.add_argument('--sub-segment', type=int, default=4, help="Window length of each time sequence")
    parser.add_argument('--sub-overlap', type=float, default=0.75)
    parser.add_argument('--sampling-rate', type=int, default=250)
    parser.add_argument('--num-channel', type=int, default=32)
    parser.add_argument('--num-time', type=int, default=5000)
    parser.add_argument('--num-feature', type=int, default=7)
    parser.add_argument('--data-format', type=str, default='rPSD', choices=['DE', 'Hjorth', 'PSD', 'rPSD',
                                                                           'sta', 'multi-view', 'raw'])
    parser.add_argument('--split', type=int, default=1, help="To split the data into smaller trials")
    parser.add_argument('--sub-split', type=int, default=1,  help="To split each smaller trials into time sequence")
    parser.add_argument('--extract-feature', type=int, default=1)
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2022)
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--patient-cmb', type=int, default=2)
    parser.add_argument('--max-epoch-cmb', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--LS', type=int, default=1, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)

    parser.add_argument('--save-path', default=os.path.join(os.getcwd(), 'save'))
    parser.add_argument('--load-path', default=os.path.join(os.getcwd(), 'save', 'candidate.pth'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=int, default=1)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='EmT')
    parser.add_argument('--graph-type', type=str, default='BL')
    parser.add_argument('--layers-graph', type=int, default=[1, 2])
    parser.add_argument('--layers-transformer', type=int, default=4)
    parser.add_argument('--num-adj', type=int, default=2)
    parser.add_argument('--hidden-graph', type=int, default=32)
    parser.add_argument('--num_head', type=int, default=16)
    parser.add_argument('--dim-head', type=int, default=32)
    parser.add_argument('--graph2token', type=str, default='Linear', choices=['Linear', 'AvgPool', 'MaxPool', 'Flatten'])
    parser.add_argument('--encoder-type', type=str, default='Cheby', choices=['GCN', 'Cheby'])
    parser.add_argument('--K', type=int, default=3, help='K for ChebyNet')

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', type=int, default=0)
    args = parser.parse_args()
    sub_to_run = np.arange(args.subjects)

    if not args.data_exist:
        pd = FACED(args)
        pd.create_dataset(
            sub_to_run, split=args.split, sub_split=args.sub_split,
            feature=args.extract_feature, band_pass_first=False if args.data_format in ['PSD', 'rPSD'] else True
        )

    if torch.cuda.is_available():
        set_gpu(args.gpu)
    seed_all(args.random_seed)
    cv = CrossValidation(args)
    cv.leave_n_sub_out(n=12, subject=sub_to_run, shuffle=False, reproduce=args.reproduce)
