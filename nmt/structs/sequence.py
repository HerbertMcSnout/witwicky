import torch
from nmt.utils import get_position_encoding, get_float_type
from .struct import Struct

class SequenceStruct(Struct):

  def __init__(self, data):
    self.data = data

  def __str__(self):
    return " ".join([str(x) for x in self.data])

  def flatten(self):
    return self.data

  def map(self, f):
    return SequenceStruct([f(x) for x in self.data])

  def get_pos_embedding(self, embed_dim, params):
    if len(params) == 0:
      return SequenceStruct(get_position_encoding(embed_dim, self.size()) * ((embed_dim / 2) ** -0.5))
    else:
      pos_seq = params[0]
      return SequenceStruct(pos_seq[:self.size(), :])

def parse(s):
  return SequenceStruct(s.strip().split())

def get_params(config):
  if config['learned_pos']:
    embed_dim = config['embed_dim']
    max_len = config['max_train_length']
    pos_seq = torch.empty(max_len, embed_dim, dtype=get_float_type())
    torch.nn.init.normal_(pos_seq, mean=0, std=embed_dim ** -0.5)
    return {"pos_seq":pos_seq}
  else:
    return {}
