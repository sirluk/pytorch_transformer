from __future__ import annotations

import yaml
import argparse
import os
import torch
import math
import random
import numpy as np
import torch.distributed as dist
from tokenizer import Tokenizer
from itertools import chain, cycle
from functools import partial
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from torch.utils.data import IterableDataset, DataLoader

from typing import Union, Optional


class TokenizedDataset(IterableDataset):

    TOKENIZER = Tokenizer()

    @classmethod
    def map_tokenizer(cls, text):
        tokens = cls.TOKENIZER.encode(text, bos=True)
        return {"tokens": tokens, 'len': len(tokens)}
    
    @classmethod
    def preprocess_hf_dataset(cls, dataset, num_proc = 64, batch_size = 1024, text_col = 'text'):
        if isinstance(dataset, Dataset):
            cols_to_remove = dataset.features
        else:
            cols_to_remove = dataset[list(dataset.keys())[0]].features

        tokenized_dataset = dataset.map(
            lambda x: cls.map_tokenizer(x[text_col]),
            remove_columns=cols_to_remove,
            num_proc=num_proc
        )

        if isinstance(tokenized_dataset, Dataset):
            tokenized_dataset = DatasetDict({'data': tokenized_dataset})

        for split, sub_ds in tokenized_dataset.items():
            split_len = sum(sub_ds['len'])
            n_batches = math.ceil(len(sub_ds) / batch_size)
            ar = np.memmap(f'{split}.bin', dtype=np.uint16, mode='w+', shape=(split_len))

            ar_start_idx = 0
            for batch_idx in range(n_batches):
                shard = sub_ds.shard(num_shards=n_batches, index=batch_idx, contiguous=True).with_format('numpy')
                shard_tokens = np.concatenate(shard['tokens'])
                ar[ar_start_idx:ar_start_idx+len(shard_tokens)] = shard_tokens
                ar_start_idx += len(shard_tokens)
    

    def __init__(self, filepaths: Union[str, list[str]], context_length: int, predict_n_tokens: int = 1):
        super().__init__()
            
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = sorted(filepaths)

        self.context_length = context_length
        self.predict_n_tokens = predict_n_tokens


    def _process_data(self, data, rng: Optional[random.Random] = None):
        if rng is None:
            rng = random.Random(0)
        ar = np.memmap(data, dtype=np.uint16, mode='r')
        n_chunks, max_offset = divmod(len(ar)-self.predict_n_tokens, self.context_length)
        chunks = list(range(n_chunks))
        rng.shuffle(chunks)
        random_offset = rng.randint(0, max_offset)
        for chunk_id in chunks:
            start_idx = random_offset + self.context_length * chunk_id
            end_idx = start_idx + self.context_length
            chunk = torch.tensor(ar[start_idx:end_idx+self.predict_n_tokens].astype(np.int64))
            yield chunk[:-self.predict_n_tokens], chunk[1:]
    
    
    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        rng.shuffle(self.filepaths)
        return chain.from_iterable(map(partial(self._process_data, rng=rng), cycle(self.filepaths)))
    

    def __len__(self):
        return sum([len(np.memmap(f, dtype=np.uint16, mode='r')) for f in self.filepaths])



if __name__ == '__main__':

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['model_cfg'])

    # ds = load_dataset("cerebras/SlimPajama-627B")
    # ds = load_dataset("openwebtext", num_proc=4, split='train[:1%]')

    try:
        _ds = load_from_disk('test.hf')
    except FileNotFoundError:
        _ds = load_dataset(path = "wikimedia/wikipedia", name = "20231101.en", split = 'train[:10000]')
        _ds = _ds.train_test_split(0.3)
        _ds.save_to_disk("test.hf")

    if not os.path.exists('train.bin'):
        TokenizedDataset.preprocess_hf_dataset(_ds, num_proc=64, batch_size=1024, text_col='text')
    
    ds = TokenizedDataset(filepaths='train.bin', context_length=cfg.context_length)

    dl = DataLoader(ds, batch_size=4)

    test = next(iter(dl))

    len(ds)

    import IPython; IPython.embed(); exit(1)
