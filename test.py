from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import csv
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from typing import Union
import math

# how padding is handled by attention mask?
# how is padding handled in general
# how is mask used

data = []
with open('data/IMDB Dataset.csv') as f:
    r = csv.reader(f, delimiter=',', quotechar='"')
    for row in r:
        data.append(row)
texts, sentiment = list(zip(*data[1:]))

texts = list(texts)

vocab_size = 16000
emb_dim = 512
context_length = 1024

class LM_DataSet(Dataset):

    def __init__(
        self,
        tokenizer: Union[str, Tokenizer],
        texts: iter,
        vocab_size: int,
        context_length: int,
        tokenizer_batch_size: int = 512
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if isinstance(texts, str):
            texts = [texts]
        self.tokenizer = self.tokenizer.train_new_from_iterator(
            texts, vocab_size=vocab_size
        )

        idx = 0
        idx_next = tokenizer_batch_size
        self._tokenized_texts = []
        self._id_ar = []
        while idx <= len(texts):
            tokenized_batch = self.tokenizer(
                texts[idx:idx_next],
                padding = False,
                truncation = False,
                return_attention_mask = False,
                return_token_type_ids = False
            )
            tokenized_batch, id_ar = list(zip(*[(torch.tensor(x + [self.tokenizer.pad_token_id]), len(x)) for x in tokenized_batch['input_ids']]))
            self._tokenized_texts.extend(tokenized_batch)
            self._id_ar.extend(id_ar)
            idx += tokenizer_batch_size
            idx_next += tokenizer_batch_size
        self._id_ar = np.array(self._id_ar)
        self._id_ar_cumsum = np.insert(np.cumsum(self._id_ar), 0, 0)

    def __len__(self):
        return self._id_ar.sum()

    def __getitem__(self, idx):
        text_idx = (self._id_ar_cumsum[1:] >= idx).argmax()
        offset = self._id_ar_cumsum[text_idx]
        start_idx = max(idx-offset-self.context_length, 0)
        end_idx = idx - offset + 1

        tokens = self._tokenized_texts[text_idx][start_idx:end_idx+1]
        return tokens[:-1], tokens[-1]


ds = LM_DataSet(
    'gpt2',
    texts,
    vocab_size,
    context_length
)


def collate_fn(batch, pad_id, d_model):
    x, y = list(zip(*batch))
    x = torch.stack([F.pad(_x, (0, d_model - len(_x)), value=pad_id) for _x in x])
    y = torch.stack(y)
    mask = ~(x == pad_id)
    return x, y, mask


dl = DataLoader(
        ds, batch_size=64, shuffle=True,
        collate_fn = lambda x: collate_fn(x, pad_id=ds.tokenizer.pad_token_id, d_model=emb_dim)
)

class Embedding(nn.Module):
    
    def __init__(self, d_vocab, d_model):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


emb = Embedding(vocab_size, emb_dim)

x, y, mask = next(iter(dl))

import IPython; IPython.embed(); exit(1)