import torch
import torch.nn as nn

a = torch.tensor([[1, 2, 3, 4],
                  [2, 3, 4, 5]])

mask = torch.tensor([[True, True, True, True],
                  [True, True, True, True]])

b = a.masked_fill(mask, 1e-9)
print(b)