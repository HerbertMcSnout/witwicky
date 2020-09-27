import torch
from nmt.utils import get_device
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

  def get_pos_embedding(self, embed_dim):
    return SequenceStruct(torch.zeros(len(self.data), embed_dim, device=get_device()))

def parse(s, clip=None):
  return SequenceStruct(s.strip().split(maxsplit=(clip or -1))[slice(clip)])

def get_params(config):
  return {}
