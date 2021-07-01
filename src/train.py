import argparse
import math
import os
import random
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from model import GeneratorModule
from dataloader import DataModule


def main(args):
    dm = DataModule(args)
    pl_module = GeneratorModule(args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(pl_module, dm)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Important args gpu_list train/valid_file '
    )
    parser.add_argument('gpus', type=str)
    parser.add_argument('--cpu', action='store_true')
    
    parser.add_argument("--progress_bar_refresh_rate", default=100, type=int)
    # files
    parser.add_argument('--train_file', type=str, default='data/train.jsonl')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')

    # training scheme
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--max_epochs', type=int, default=10)

    # save
    parser.add_argument('--seed', type=int, default=42)

    # model
    parser.add_argument('--bert_config', type=str, default='pretrained-models/bert_base_uncased')
    parser.add_argument('--gpt2_config', type=str, default='pretrained-models/gpt2')

    parser.add_argument('--gpt2_truncate', type=int, default=256) # for gpt2
    parser.add_argument('--knowledge_truncate', type=int, default=64) # for gpt2
    parser.add_argument('--text_truncate', type=int, default=64) # for gpt2
    parser.add_argument('--segment', action="store_true")

    parser.add_argument('--n_sent', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--min_length', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--lstm_hidden', type=int, default=256)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    main(args)  
        
