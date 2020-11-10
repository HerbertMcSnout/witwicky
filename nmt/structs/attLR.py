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

  bmask = torch.zeros(bsz, src_len, dtype=torch.bool, device=device)
  for line in range(bsz):
    bmask[line, :structs[line].size()] = True

  fmask = torch.zeros(bsz, num_heads, src_len, src_len, dtype=torch.float, device=device)
  for head in range(num_heads):
    for line in range(bsz):
      size = structs[line].size()
      for word in range(size):
        if word > 0:
          fmask[line, head, word, word - 1] = attL[head]
        if word + 1 < size:
          fmask[line, head, word, word + 1] = attR[head]
  
  return fmask, bmask.unsqueeze(1).unsqueeze(2)
