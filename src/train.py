import argparse
import math
import os
import random
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT2Summ, sequence_loss
from dataloader import KGDataset, get_batch_loader, GenBatcher


def main(args):
    global_step = 0
    
    train_dataset = KGDataset(args.train_file, max_knowledge=999)
    train_loader = get_batch_loader(train_dataset, batch_size=args.batch_size, is_test=False)
    
    valid_dataset = KGDataset(args.valid_file, max_knowledge=999)
    valid_loader = get_batch_loader(valid_dataset, batch_size=args.batch_size, is_test=True)

    # Batcher
    gen_batcher = GenBatcher(args.text_truncate, args.gpt2_truncate, args.gpt2_config, args.cuda)
    print("Datasets & Dataloaders instantiated...")

    # model
    gen_model = GPT2Summ(tokenizer=gen_batcher.tokenizer, gpt2_config=args.gpt2_config, segment=args.segment)
    gen_model.to(args.cuda)
    print("Generater model instantiated...")
    
    # loss criterion
    ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
    gen_criterion = lambda logits, targets: sequence_loss(logits, targets, ce, pad_idx=-1)

    # optimizer
    optimizer = torch.optim.Adam(gen_model.parameters(), lr = args.lr)
    
    for epoch in range(args.epochs):
        print(f"Beginning epoch: {epoch}.....")
        #train step
        n_token, train_loss = 0, 0.0 # ppl
        for knowledges, histories, users, responses, knowledge_lens in train_loader:
            histories = [his.split('\n\n') for his in histories]
            gen_args = gen_batcher(histories, users, responses, args.segment, True)
            
            optimizer.zero_grad()

            outputs = gen_model(gen_args[0].to(args.cuda), token_type_ids=gen_args[1].to(args.cuda) if gen_args[1] else None)
            loss = gen_criterion(outputs[0], gen_args[2].to(args.cuda))
            # breakpoint()

            loss.mean().backward()
            optimizer.step()
            
            n_token += loss.size(0)
            train_loss += loss.sum().item()
            global_step += 1

        TrainMeanLoss = train_loss / n_token

        # valid step
        with torch.no_grad():
            n_token, valid_loss = 0, 0.0 # ppl
            for i, knowledges, histories, users, responses, knowledge_lens in enumerate(valid_loader):
                histories = [his.split('\n\n') for his in histories]
                gen_args = gen_batcher(histories, users, responses, args.segment, True)
                outputs = gen_model(gen_args[0].to(args.cuda), token_type_ids=gen_args[1].to(args.cuda) if gen_args[1] else None)
                loss = gen_criterion(outputs[0], gen_args[2].to(args.cuda))
                
                n_token += loss.size(0)
                valid_loss += loss.sum().item()

            ValidMeanLoss = valid_loss / n_token

        
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("**********************************")
        print("EPOCH: {} results.......... {}".format(epoch, time_str))
        print("Step: %d \t| train ppl: %.3f \t|valid ppl: %.3f" % (global_step, math.exp(TrainMeanLoss), math.exp(ValidMeanLoss)))
        print("**********************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Important args gpu_list train/valid_file '
    )

    # files
    parser.add_argument('--train_file', type=str, default='data/train.jsonl')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')

    # training scheme
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=10)

    # save
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--log', type=str, default='wizard_of_wikipedia/log')

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
    parser.add_argument('--lstm_layer', type=int, default=1)

    # gpu
    parser.add_argument('--gpu_list', type=str, default='2')
    parser.add_argument('--gpu_ratio', type=float, default=0.85)
    parser.add_argument('--no_cuda', action="store_true")

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.cuda = torch.device('cuda' if args.cuda else 'cpu')
    main(args)  
        
