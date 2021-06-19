import os
import torch
import json
import re

from torch.utils.data import DataLoader, Dataset

from itertools import cycle

import numpy as np
from transformers import BertTokenizer, GPT2Tokenizer

class KGDataset(Dataset):
    def __init__(self, data_path, max_knowledge=32):
        # load data
        self._data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self._data.append(json.loads(line))

        self._n_data = len(self._data)
        self._max_knowledge = max_knowledge

    def __len__(self):
        return self._n_data

    def __getitem__(self, i):
        knowledge = self._data[i]['knowledge']
        history = self._data[i]['history']
        user = self._data[i]['user'] # response: 1, another: 0
        response = self._data[i]['response']

        if len(knowledge) > self._max_knowledge:
            # wizard
            keepers = 1 + np.random.choice(len(knowledge) - 1, self._max_knowledge, False)
            keepers[0] = 0
            knowledge = [knowledge[id] for id in keepers]

        return ('\n\n'.join(knowledge), '\n\n'.join(history), np.array(user), response)
    
    
def collate_fn(batch):
    knowledges   = [item[0] for item in batch]
    histories    = [item[1] for item in batch]
    users        = [item[2] for item in batch]
    responses    = [item[3] for item in batch]
    knowledge_lens = [len(knowledge.strip().split('\n\n')) for knowledge in knowledges]

    max_user = max([u.shape[0] for u in users])
    users = [np.pad(u, (0, max_user - u.shape[0]), 'constant', constant_values=-1) for u in users]

    return knowledges, histories, users, responses, knowledge_lens


def get_batch_loader(dataset, collate_fn=collate_fn, batch_size=2, num_workers=0, is_test=True):
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(not is_test), num_workers=num_workers, collate_fn=collate_fn
    )
    return loader


class GenBatcher:
    def __init__(self, text_truncate, block_size, gpt2_config, cuda=True):
        self.text_truncate = text_truncate
        self.block_size = block_size

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_config)

        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<user1>", "<user2>"]}
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

        self.eos_id = self.tokenizer.eos_token_id
        self.device = torch.device('cuda' if cuda else 'cpu')
        # todo
        self.user_id = [self.tokenizer.convert_tokens_to_ids('<user1>'), self.tokenizer.convert_tokens_to_ids('<user2>')]

    def tokenize(self, text, text_pair=None):
        return self.tokenizer.encode(text, text_pair=text_pair, add_special_tokens=True)

    def __call__(self, histories, users, responses=None, segment=True, training=True):
        if training:
            assert responses is not None
            input_ids, targets, token_type_ids = [], [], []
            for his, user, resp in zip(histories, users, responses):
                user = [u for u in user.tolist() if u >= 0]
                history_input, history_type = [], []
                for h, u in zip(his, user):
                    tmp = [self.user_id[u]] + self.tokenize(h)[:self.text_truncate]
                    history_input += tmp
                    history_type += len(tmp) * [self.user_id[u]]

                response_input = [self.user_id[1]] + self.tokenize(resp)
                response_type = len(response_input) * [self.user_id[1]]

                ids = history_input + response_input
                type_ids = history_type + response_type
                tgt = [-1] * len(history_input) + response_input[1:] + [self.eos_id]

                ids = ids[-self.block_size:]
                type_ids = type_ids[-self.block_size:]
                tgt = tgt[-self.block_size:]

                ids = ids + [0] * (self.block_size - len(ids))
                type_ids = type_ids + [0] * (self.block_size - len(type_ids))
                tgt = tgt + [-1] * (self.block_size - len(tgt))

                input_ids.append(ids)
                token_type_ids.append(type_ids)
                targets.append(tgt)
            input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)
            if segment:
                return input_ids, token_type_ids, targets
            else:
                return input_ids, None, targets
        else:
            user = [u for u in users[0].tolist() if u >= 0]
            history_input = []
            for h, u in zip(histories[0], user):
                history_input += [self.user_id[u]] + self.tokenize(h)[:self.text_truncate]

            input_ids = history_input + [self.user_id[1]]
            input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.long).unsqueeze(0)
            return input_ids