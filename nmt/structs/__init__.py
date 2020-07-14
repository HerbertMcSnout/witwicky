import sys
import os
import importlib
import struct

exclude = ['__init__.py', 'struct.py']

cd = os.path.dirname(__file__)
for fn in os.listdir(cd):
  if fn.endswith('.py') and fn not in exclude:
    importlib.import_module('nmt.structs.' + fn[:-3])

"""
Each struct is a module with               type
- a 'parse' function                       str -> Struct implementation
- a 'get_params' function                  config -> {name1: torch.Tensor(*), ...}
- (optional) a get_reg_penalty function    torch.Tensor(batch_size, max_len, embed_dim) -> torch.Tensor()
"""
