import time
import torch

x = torch.randn(size=(10,))
x.to('cuda:0')
time.sleep(100)