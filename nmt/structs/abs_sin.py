import torch
import numpy as np
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut

def get_position_encoding_row(d, t):
  c = 100
  k = (torch.arange(d, dtype=torch.int, device=ut.get_device()) // 2).type(torch.float)
  w_k = torch.pow(float(c), -2*k/d)
  p_t = torch.empty_like(w_k)
  p_t[0::2] = torch.sin(w_k[0::2] * t)
  p_t[1::2] = torch.cos(w_k[1::2] * t)
  return p_t

class Tree(tree_utils.Tree):

  def label(self, i=0):
    l = self.r and self.l.label(2*i + 1)
    r = self.r and self.r.label(2*i + 2)
    return self.new(i, l, r)

  def get_pos_embedding_h(self, embed_dim):
    self.v = get_position_encoding_row(embed_dim, self.v)
    if self.l: self.l.get_pos_embedding_h(embed_dim)
    if self.r: self.r.get_pos_embedding_h(embed_dim)

  def get_pos_embedding(self, embed_dim):
    x = self.label()
    x.get_pos_embedding_h(embed_dim)
    return x

def parse(fun_str, clip=None):
  return tree_utils.parse(fun_str, cls=Tree, clip=clip)
