import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config


class GPT2Summ(GPT2PreTrainedModel):
    '''succeed from GPT2PreTraninedModel which has implemented the 'generate' func'''

    def __init__(self, tokenizer, gpt2_config, segment=True):
        config = GPT2Config.from_pretrained(gpt2_config)
        super(GPT2Summ, self).__init__(config)
        self.transformer = GPT2Model.from_pretrained(gpt2_config)
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.user_id = [tokenizer.convert_tokens_to_ids('<user1>'),
                        tokenizer.convert_tokens_to_ids('<user2>')]
        self.segment = segment

        self.lm_head = nn.Linear(config.n_embd, len(tokenizer), bias=False)
        self.config.vocab_size = len(tokenizer)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        token_type_ids = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            type_ids = []
            type_ids.append(ids)
            token_type_ids.append(type_ids)
        token_type_ids = torch.tensor(token_type_ids).type_as(input_ids)

        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        if self.segment:
            inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids}
        else:
            inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return (lm_logits,) + transformer_outputs[1:]
    
    def batch_decode(self, input_ids, max_len, min_len, early_stopping, beam_size,
                     repetition_penalty, eos_id, length_penalty, no_repeat_ngram_size):
        # new-version
        output_sequences = self.generate(
            input_ids=input_ids,
            max_length=input_ids.size(1) + max_len,
            min_length=input_ids.size(1) + min_len,
            do_sample=False,
            early_stopping=early_stopping,
            num_beams=beam_size,
            repetition_penalty=repetition_penalty,
            pad_token_id=0,
            # pad_token_id=None,
            eos_token_id=eos_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return output_sequences


def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""

    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))

    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))

    return loss
