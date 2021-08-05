import argparse
import math
import os
import random
import numpy as np
from datetime import datetime
from pytorch_lightning.callbacks.base import Callback

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT2Summ, sequence_loss
from dataloader import get_train_dataloader, get_val_dataloader, GenBatcher



def main(args):
    ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
    gen_criterion = lambda logits, targets: sequence_loss(logits, targets, ce, pad_idx=-1)
    gen_batcher = GenBatcher(args.text_truncate, args.gpt2_truncate, args.gpt2_config)
    gen_model = GPT2Summ(tokenizer=gen_batcher.tokenizer, gpt2_config=args.gpt2_config, segment=args.segment)
    
    optim = torch.optim.Adam(gen_model.parameters(), lr = args.lr)
    
    train_dl = get_train_dataloader()
    val_dl = get_val_dataloader()
    
    global_step = 0
    for epoch in range(args.max_epochs):
        n_token_train, train_loss, n_token_valid, valid_loss = 0, 0.0, 0, 0.0
        gen_model.train()
        for batch in train_dl:
            optim.zero_grad()
            
            knowledges, histories, users, responses, knowledge_lens = batch
            histories = [his.split('\n\n') for his in histories]
            input_ids, token_type_ids, targets = gen_batcher(histories, users, responses, args.segment, training=True)
            
            outputs = gen_model(input_ids.to(args.device), token_type_ids=token_type_ids.to(args.device) if token_type_ids else None)
            loss = gen_criterion(outputs[0], targets.to(args.device))
            loss.mean().backward()
            optim.step()

            n_token_train += loss.size(0)
            train_loss += loss.sum().item()
            global_step += 1
        
        TrainMeanLoss = train_loss / n_token_train
        
        if (epoch % args.check_val_every_n_epoch) ==0:
            gen_model.eval()
            for batch in val_dl:
                with torch.no_grad():
                    knowledges, histories, users, responses, knowledge_lens = batch
                    histories = [his.split('\n\n') for his in histories]
                    input_ids, token_type_ids, targets = gen_batcher(histories, users, responses, args.segment, training=True)
                    
                    outputs = gen_model(input_ids.to(args.device), token_type_ids=token_type_ids.to(args.device) if token_type_ids else None)
                    loss = gen_criterion(outputs[0], targets.to(args.device))

                    n_token_valid += loss.size(0)
                    valid_loss += loss.sum().item()
                    return loss.mean()
            ValidMeanLoss = valid_loss / n_token_valid
        
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("**********************************")
        print("EPOCH: {} results.......... {}".format(epoch, time_str))
        print("Step: %d \t| train ppl: %.3f \t|valid ppl: %.3f" % (global_step, math.exp(TrainMeanLoss), math.exp(ValidMeanLoss)))
        print("**********************************")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Important args gpu_list train/valid_file '
    )
    parser.add_argument('gpus', type=str)
    parser.add_argument('--cpu', action='store_true')
    
    parser.add_argument("--progress_bar_refresh_rate", default=100, type=int)
    parser.add_argument("--fast_dev_run", action="store_true")
    # files
    parser.add_argument('--train_file', type=str, default='data/train.jsonl')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')

    # training scheme
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
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
    
    main(args)  
        
