import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import json
import re
from functools import partial
from itertools import repeat
from multiprocessing import Pool
import pickle

folder = '../activations'

files = [f for f in os.listdir(folder) if f[-3:] == 'npy']

files = files[:10]

with open(os.path.join(folder, 'quantization_scalars.json'), 'r') as fp:
    quantization_scalars = json.load(fp)

d = defaultdict(list)
for f in files:
    parts = f.split('.')[0].split('_')
    tp = parts[0] if not 'embedding' in f else 'embedding'
    if tp == 'activations':
        block = parts[2][5:]
        act_type = '_'.join(parts[3:])
        k = f'{block}_{act_type}'
        d[k].append((k, f))
    else:
        d['other'].append((tp, f))


def get_running_stats(l, quantization_scalars, folder):
    d = {}
    lengths = {}
    for (tp, f) in l:
        data = np.load(os.path.join(folder, f))
        nd = quantization_scalars[f]
        x = data / nd
        x2 = np.square(x)
        try:
            tmp = np.concatenate([d[tp][2:], np.absolute(x)[np.newaxis]], axis=0)
            max_vals = tmp.max(axis=0)
            min_vals = tmp.min(axis=0)  
            d[tp][:2] += np.stack([x, x2], axis=0)
            d[tp][2:] = np.stack([max_vals, min_vals], axis=0)
            lengths[tp] += 1
        except KeyError:
            x_abs = np.absolute(x)
            d[tp] = np.stack([x, x2, x_abs, x_abs], axis=0)
            lengths[tp] = 1
    return d, lengths


def get_running_stats_batched(l, quantization_scalars, folder, batch_size = 124):
    i = 0
    stats = {}
    while True:
        out = {}
        for (tp, f) in l:
            data = np.load(os.path.join(folder, f))
            nd = quantization_scalars[f]
            x = data / nd
            x_batch = x[i:i+batch_size]
            try:
                out[tp] = np.concatenate([out[tp], x_batch[np.newaxis]], axis=0)
            except KeyError:
                out[tp] = x_batch[np.newaxis]
        for k in list(out.keys()):
            if x_batch.shape[0] < batch_size: # take any x_batch
                out[k] = out[k][:,:x_batch.shape[0]]
            mean = out[k].mean(axis=0)
            std = out[k].std(axis=0)
            max_vals = np.absolute(out[k]).max(axis=0)
            min_vals = np.absolute(out[k]).min(axis=0)
            p50 = np.median(out[k], axis=0)
            p25 = np.percentile(out[k], 25, axis=0)
            p75 = np.percentile(out[k], 75, axis=0)
            try:
                stats[k] = np.concatenate([stats[k], np.stack([mean, std, max_vals, min_vals, p25, p50, p75], axis=0)], axis=1)
            except KeyError:
                stats[k] = np.stack([mean, std, max_vals, min_vals, p25, p50, p75], axis=0)
        i += batch_size
        if i >= x.shape[0]:
            break


def all_cell_values(l, quantization_scalars, folder, i = None):
    i = 0
    d = defaultdict(list)
    for (tp, f) in l:
        data = np.load(os.path.join(folder, f))
        nd = quantization_scalars[f]
        x = data / nd
        d[tp].append(x[i])
    return d



pf = partial(get_running_stats, quantization_scalars=quantization_scalars, folder=folder)

pfb = partial(get_running_stats_batched, quantization_scalars=quantization_scalars, folder=folder, batch_size=16)

acv = partial(all_cell_values, quantization_scalars=quantization_scalars, folder=folder)

res = []
with Pool(6) as pool:
    for x in tqdm(pool.imap_unordered(pfb, list(d.values())), total=len(d)):
        res.append(x)

# with open("activations_detailed.pkl","wb") as f:
with open("test.pkl","wb") as f:
    pickle.dump(res, f)