import torch
from nmt.utils import get_position_encoding, get_float_type
from nmt.structs.struct import Struct

class SequenceStruct(Struct):

  def __init__(self, data):
    self.data = data

  def __str__(self):
    return ' '.join([str(x) for x in self.data])

  def flatten(self):
    return self.data

  def map(self, f):
    return SequenceStruct([f(x) for x in self.data])

  def set_clip_length(self, clip):
    self.data = self.data[:clip]

  def get_pos_embedding(self, embed_dim, pos_seq=None):
    size = self.size()
    if pos_seq is None:
      return SequenceStruct(get_position_encoding(embed_dim, size) * ((embed_dim / 2) ** -0.5))
    else:
      return SequenceStruct(pos_seq[:size, :])

  def maybe_add_eos(self, EOS_ID):
    self.data.append(EOS_ID)

def parse(s, clip=None):
  return SequenceStruct(s.strip().split(maxsplit=(clip or -1))[slice(clip)])

def get_params(config):
  #if config['learned_pos']:
  #  embed_dim = config['embed_dim']
  #  # TODO: if you ever switch to using a struct for trg, make sure to somehow use max_trg_length here
  #  max_len = config['max_src_length']
  #  pos_seq = torch.empty(max_len, embed_dim, dtype=get_float_type())
  #  torch.nn.init.normal_(pos_seq, mean=0, std=embed_dim ** -0.5)
  #  return {'pos_seq':pos_seq}
  #else:
  #  return {}
  return {}
