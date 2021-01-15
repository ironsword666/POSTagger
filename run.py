import os
import random
import argparse
from datetime import datetime, timedelta

from config import Config
from src.learner.tagger import Tagger 

import torch

if __name__ == '__main__':

    print('program start time: {} \n'.format(datetime.now()))

    args = Config()

    parser = argparse.ArgumentParser(description='A POS Tagger, may used to chunk, NER, ...')
    
    parser.add_argument('--device', default='-1',
                        help='ID of GPU to use')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for pytorch')
    parser.add_argument('--feat', choices=['char', 'bert'],
                        help='use which feature as extra input')
    parser.add_argument('--use_crf', action='store_true',
                        help='whether use crf to calculate loss') 
    parser.add_argument('--ftrain', default=None,
                        help='path to train file')
    parser.add_argument('--fdev', default=None,
                        help='path to dev file')
    parser.add_argument('--ftest', default=None,
                        help='path to test file')
    parser.add_argument('--w2v', default=None,
                        help='where to load pretrained embeddings')  
    parser.add_argument('--unk', default=None,
                        help='what is the label of unknown word in w2v') 
    parser.add_argument('--save_dir', default=None,
                        help='path to directory used to save model or fields')
    
    # TODO parse_known_args()
    parser.parse_args(namespace=args)

    # random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # TODO whether needed ?
    torch.cuda.manual_seed(args.seed)

    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('CUDA_VISIBLE_DEVICES: {} \n'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device mode: {} \n'.format(args.device))

    tagger = Tagger.build_tagger(args)
    tagger.train(tagger.args)
    
    print('program end time: {} '.format(datetime.now()))


