import torch
from nmt.structs.struct import Struct
import nmt.structs.tree_utils as tree_utils
import nmt.utils as ut
from collections import Counter


# TODO: Make these get saved!

max_src_length = -1

common_map = {}
most_common = Counter()

def get_next_if(i):
  return i if i in common_map else (i - 1) >> 1

def record_common(tree, i=0):
  most_common[i] += 1
  if tree.l: record_common(tree.l, (i << 1) + 1)
  if tree.r: record_common(tree.r, (i << 1) + 2)

def finalize_common():
  if not common_map:
    i = 0
    for k, _ in most_common.most_common(max_src_length):
      common_map[k] = i
      i += 1

class Tree(tree_utils.Tree):

  def get_pe_h(self, abs_pe, i):
    l = self.l.get_pe_h(abs_pe, get_next_if((i << 1) + 1)) if self.l else None
    r = self.r.get_pe_h(abs_pe, get_next_if((i << 1) + 2)) if self.r else None
    return self.new(abs_pe[common_map[i], :], l, r)

  def get_pos_embedding(self, embed_dim, abs_pe):
    finalize_common()
    return self.get_pe_h(abs_pe, 0)

def parse(fun_str, clip=None):
  t = tree_utils.parse(fun_str, cls=Tree, clip=clip)
  if not common_map: record_common(t)
  return t

def get_params(config):
  global max_src_length
  max_src_length = config['max_src_length']
  embed_dim = config['embed_dim']
  abs_pe = torch.empty(max_src_length, embed_dim, dtype=torch.float, device=ut.get_device())
  torch.nn.init.normal_(abs_pe, mean=0, std=embed_dim ** -0.5)

  return dict(
    abs_pe = abs_pe,
  )
