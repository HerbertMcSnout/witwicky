import torch
from nmt.utils import get_position_embedding, get_device
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

  def get_pos_embedding(self, embed_dim, attL, attR):
    return SequenceStruct(get_position_embedding(embed_dim, self.size()) * ((embed_dim / 2) ** -0.5))

def parse(s, clip=None):
  return SequenceStruct(s.strip().split(maxsplit=(clip or -1))[slice(clip)])

def get_params(config):
  device = get_device()
  num_heads = config['num_enc_heads']
  return dict(
    attL = torch.zeros(num_heads, device=device),
    attR = torch.zeros(num_heads, device=device)
  )

def get_enc_mask(toks, structs, num_heads, attL, attR):
  bsz, src_len = toks.size()
  device = get_device()

  diagL = torch.diag(torch.ones(src_len, dtype=torch.float, device=device), diagonal=-1)[:src_len, :src_len]
  diagR = torch.diag(torch.ones(src_len, dtype=torch.float, device=device), diagonal=1)[:src_len, :src_len]
  maskL = (diagL.unsqueeze(0) * attL.reshape(-1, 1, 1)).unsqueeze(0)
  maskR = (diagR.unsqueeze(0) * attR.reshape(-1, 1, 1)).unsqueeze(0)
  return maskL + maskR, toks.type(torch.bool).unsqueeze(1).unsqueeze(2)
