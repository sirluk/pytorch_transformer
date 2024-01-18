import os
import json
import functools
import pickle
from multiprocessing import Pool
from tokenizer import Tokenizer
from tqdm import tqdm

path = "/system/user/publicdata/slimpajama_sampled/train/chunk1"
num_proc = 64

def map_tokenizer(text, tokenizer):
    tokens = tokenizer.encode(text, bos=True)
    return {"tokens": tokens, 'len': len(tokens)}


if __name__ == '__main__':

    tokenized_data = []
    files = list(os.listdir(path))
    for file in tqdm(files, total=len(files)):

        with open(os.path.join(path, file), 'r') as f:
            data = [json.loads(line)['text'] for line in f]

        tokenizer = Tokenizer()

        with Pool(num_proc) as p:
            result = p.map(functools.partial(map_tokenizer, tokenizer=tokenizer), data)

        tokenized_data.extend(result)
        
    with open('/system/user/publicwork/hauzenbe/git_repos/pytorch_transformer/data/slimpajama/train_raw.pkl', 'wb') as f:
        pickle.dump(tokenized_data, file)
        
    import IPython; IPython.embed(); exit(1)