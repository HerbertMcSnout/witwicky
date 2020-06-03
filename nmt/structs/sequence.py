import torch
from nmt.utils import get_positional_encoding
from .struct import Struct

class SequenceStruct(Struct):

  def __init__(self, data):
    self.data = data

  def flatten(self):
    return self.data

  def map(self, f):
    return SequenceStruct([f(x) for x in self.data])

  def get_pos_embedding(self, embed_dim, params):
    if len(params) == 0:
      return SequenceStruct(get_positional_encoding(embed_dim, self.size()))
    else:
      pos_seq = params[0]
      return SequenceStruct([pos_seq[i, :] for i in range(self.size())])

def parse(s):
  return SequenceStruct(s.strip().split())

def get_params(config):
  if config['learn_pos']:
    return []
  else:
    embed_dim = config['embed_dim']
    max_len = config['max_train_length']
    pos_seq = torch.Tensor(max_len, embed_dim)
    torch.nn.init.normal_(pos_seq, mean=0, std=embed_dim ** -0.5)
    return [pos_seq]
